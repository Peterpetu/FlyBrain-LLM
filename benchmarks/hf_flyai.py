"""HuggingFace-transformers wrapper for FlyAI hybrid LMs, so a trained `flyai.lm.HybridFlyAILM`
can be scored by lm-eval-harness (`--model hf --model_args pretrained=<dir>`) and used with
`generate()`.

    FlyAIHFConfig      PretrainedConfig mirroring flyai.lm.LMConfig (model_type "flyai")
    FlyAIForCausalLM   PreTrainedModel + GenerationMixin around HybridFlyAILM
    export_checkpoint  save config + weights (+ tokenizer) in HF format
    register           AutoConfig / AutoModelForCausalLM registration (also run at import time)

Incremental decoding. `generate()` carries a `FlyAIState` in `past_key_values`. Recurrent (flyai)
layers advance their block state one token at a time. The attention layers (every `attn_every`-th)
have no KV-cache API, so their *input* residual stream over the whole prefix is cached and the
causal attention is recomputed over it: exact, O(T d^2) per step per attention layer (a KV cache
would be O(T d)). Models with a Gated-DeltaNet mixer fall back to recomputing the whole prefix
through every layer (exact but O(T^2) over a generation), because the GDN mixer's causal
depthwise conv over time has no streaming API. `attention_mask`: right padding is accepted (and
ignored: pad tokens after the real ones do not affect the real positions' logits); LEFT padding is
rejected with an error because the recurrent layers would consume the pad tokens first. lm-eval's
loglikelihood tasks right-pad; run generate_until tasks with batch_size 1.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:                      # importable from any cwd (lm-eval, tests)
    sys.path.insert(0, str(_REPO))

from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from flyai.lm import HybridFlyAILM, LMConfig, make_lm


class FlyAIHFConfig(PretrainedConfig):
    """`transformers` config with the fields of `flyai.lm.LMConfig` (`vocab` is called `vocab_size`)."""

    model_type = "flyai"
    attribute_map = {"hidden_size": "d", "num_hidden_layers": "n_layers", "num_attention_heads": "n_heads"}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size: int = 50257, d: int = 768, n_layers: int = 12, n_heads: int = 12, mixer: str = "flyai",
                 attn_every: int = 4, mlp_mult: int = 4, tie_weights: Optional[bool] = None, block: Optional[dict] = None,
                 gdn_key: int = 128, seed: int = 0, max_position_embeddings: int = 2048,
                 pad_token_id=None, bos_token_id=None, eos_token_id=None, **kwargs):
        tw = kwargs.pop("tie_word_embeddings", None)          # HF spelling of tie_weights; tie_weights wins if both given
        if tie_weights is None:
            tie_weights = True if tw is None else tw
        self.vocab_size, self.d, self.n_layers, self.n_heads = vocab_size, d, n_layers, n_heads
        self.mixer, self.attn_every, self.mlp_mult, self.gdn_key, self.seed = mixer, attn_every, mlp_mult, gdn_key, seed
        self.block = dict(block or {})
        self.tie_weights = bool(tie_weights)
        self.max_position_embeddings = max_position_embeddings   # lm-eval reads this as the context length
        super().__init__(**kwargs)
        # Set after super().__init__: transformers v4 re-reads these from kwargs, v5 pops them as generation params.
        self.tie_word_embeddings = self.tie_weights
        self.pad_token_id, self.bos_token_id, self.eos_token_id = pad_token_id, bos_token_id, eos_token_id

    def to_lm_config(self) -> LMConfig:
        return LMConfig(vocab=self.vocab_size, d=self.d, n_layers=self.n_layers, n_heads=self.n_heads, mixer=self.mixer,
                        attn_every=self.attn_every, mlp_mult=self.mlp_mult, tie_weights=self.tie_weights,
                        block=dict(self.block), gdn_key=self.gdn_key, seed=self.seed)

    @classmethod
    def from_lm_config(cls, cfg: LMConfig, **kwargs) -> "FlyAIHFConfig":
        return cls(vocab_size=cfg.vocab, d=cfg.d, n_layers=cfg.n_layers, n_heads=cfg.n_heads, mixer=cfg.mixer,
                   attn_every=cfg.attn_every, mlp_mult=cfg.mlp_mult, tie_weights=cfg.tie_weights, block=dict(cfg.block),
                   gdn_key=cfg.gdn_key, seed=cfg.seed, **kwargs)

    @classmethod
    def from_size(cls, size: str = "125m", mixer: str = "flyai", vocab: int = 50257, **block) -> "FlyAIHFConfig":
        """Config for a `flyai.lm.make_lm(size, mixer, vocab, **block)` model, without allocating its weights."""
        with torch.device("meta"):
            cfg = make_lm(size, mixer, vocab, **block).cfg
        return cls.from_lm_config(cfg)


def _tree_map(fn, obj):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_map(fn, v) for v in obj)
    return obj


class FlyAIState:
    """Decoding state carried in `past_key_values` between `generate()` steps.

    `states[i]` is layer i's recurrent block state (None for attention layers); `attn_inputs[i]` is the
    residual stream entering attention layer i over the whole prefix, (B, T, d), so its causal attention
    can be recomputed exactly; `tokens` is the full prefix (B, T) in full-recompute mode. Not a
    `transformers.Cache`: the model marks itself stateful and opts out of the library cache classes."""

    is_compileable = False
    is_croppable = False

    def __init__(self, states: list, attn_inputs: dict, tokens: Optional[torch.Tensor], seq_len: int):
        self.states, self.attn_inputs, self.tokens, self.seq_len = states, attn_inputs, tokens, seq_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_len

    def reorder_cache(self, beam_idx: torch.Tensor) -> "FlyAIState":     # beam search
        sel = lambda t: t.index_select(0, beam_idx.to(t.device))
        self.states, self.attn_inputs = _tree_map(sel, self.states), _tree_map(sel, self.attn_inputs)
        if self.tokens is not None:
            self.tokens = sel(self.tokens)
        return self


def _cache_len(cache) -> int:
    get = getattr(cache, "get_seq_length", None)
    return int(get()) if callable(get) else len(cache)


class FlyAIForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FlyAIHFConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _is_stateful = True                                     # recurrent state cannot be rolled back (no assisted decoding)
    _tied_weights_keys = {"model.head.weight": "model.emb.weight"}
    _no_split_modules = ["Layer"]

    def __init__(self, config: FlyAIHFConfig, model: Optional[HybridFlyAILM] = None):
        super().__init__(config)
        self.model = model if model is not None else HybridFlyAILM(config.to_lm_config())
        self.post_init()

    @classmethod
    def from_flyai(cls, model: HybridFlyAILM, cfg: Optional[LMConfig] = None, **config_kwargs) -> "FlyAIForCausalLM":
        """Wrap an existing (e.g. trained) HybridFlyAILM in place; parameters are shared, not copied."""
        return cls(FlyAIHFConfig.from_lm_config(cfg or model.cfg, **config_kwargs), model=model)

    # --- weights: keep whatever HybridFlyAILM initialised / whatever was loaded ------------------------
    def _init_weights(self, module):
        pass

    def get_input_embeddings(self):
        return self.model.emb

    def set_input_embeddings(self, value):
        self.model.emb = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, value):
        self.model.head = value

    # --- generation plumbing ---------------------------------------------------------------------------
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False                                        # generate() must not inject a DynamicCache

    @property
    def incremental(self) -> bool:
        """True when every layer can be advanced token by token (flyai / attention); GDN layers cannot."""
        return all(l.kind in ("flyai", "attention") for l in self.model.layers)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        if isinstance(past_key_values, FlyAIState):
            input_ids = input_ids[:, past_key_values.seq_len:]        # only the tokens the state has not consumed
        elif past_key_values is not None and _cache_len(past_key_values) == 0:
            past_key_values = None                                    # empty library cache -> no past
        return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask,
                "use_cache": use_cache}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values.reorder_cache(beam_idx) if isinstance(past_key_values, FlyAIState) else past_key_values

    # --- forward ---------------------------------------------------------------------------------------
    def _run(self, tokens: torch.Tensor, past: Optional[FlyAIState], use_cache: bool):
        m = self.model
        if past is None and not use_cache:                            # plain forward: the model's own path
            return m(tokens)[0], None
        T_new = tokens.shape[1]
        if not self.incremental:                                      # exact full-prefix recompute (O(T^2) per generation)
            full = tokens if past is None else torch.cat([past.tokens, tokens], 1)
            logits = m(full)[0][:, -T_new:]
            return logits, (FlyAIState([], {}, full, full.shape[1]) if use_cache else None)
        x = m.emb(tokens); states, attn_inputs = [], {}
        for i, layer in enumerate(m.layers):
            if layer.kind == "attention":                             # same ops as Layer.forward, over the cached prefix
                x_full = x if past is None else torch.cat([past.attn_inputs[i], x], 1)
                if use_cache:
                    attn_inputs[i] = x_full
                x = x + layer.mix(layer.n1(x_full))[0][:, -T_new:]
                x, st = x + layer.mlp(layer.n2(x)), None
            else:
                x, st = layer(x, None if past is None else past.states[i])
            states.append(st)
        logits = m.head(m.norm(x))
        seq_len = T_new + (0 if past is None else past.seq_len)
        return logits, (FlyAIState(states, attn_inputs, None, seq_len) if use_cache else None)

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, use_cache=None,
                return_dict=None, inputs_embeds=None, **kw):
        if input_ids is None or inputs_embeds is not None:
            raise NotImplementedError("FlyAIForCausalLM takes `input_ids` only")
        if attention_mask is not None and not bool(attention_mask.all()):
            m = attention_mask.bool()
            if bool((((~m).cumsum(1) > 0) & m).any()):                      # a real token AFTER a pad: left (or interior) padding
                raise ValueError("FlyAIForCausalLM cannot take LEFT-padded batches: the recurrent layers consume every position. "
                                 "Use batch_size 1 (generate_until tasks) or right padding.")
            warnings.warn("FlyAIForCausalLM ignores attention_mask: right-padded positions are fed to the recurrent layers after the "
                          "real tokens (harmless for the logits of the real positions)", stacklevel=2)
        past = past_key_values
        if past is not None and not isinstance(past, FlyAIState):
            if _cache_len(past) != 0:
                raise TypeError(f"past_key_values must be a FlyAIState, got {type(past).__name__}")
            past = None
        logits, new_past = self._run(input_ids, past, bool(use_cache))
        loss = None
        if labels is not None:                                        # GPT-2 convention: position t predicts token t+1
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        if return_dict is False:
            return tuple(v for v in (loss, logits, new_past) if v is not None)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=new_past)


def register(exist_ok: bool = True) -> None:
    """Make `AutoConfig` / `AutoModelForCausalLM.from_pretrained(<export dir>)` resolve model_type "flyai"."""
    AutoConfig.register("flyai", FlyAIHFConfig, exist_ok=exist_ok)
    AutoModelForCausalLM.register(FlyAIHFConfig, FlyAIForCausalLM, exist_ok=exist_ok)


register()


def export_checkpoint(model: HybridFlyAILM, cfg: LMConfig, tokenizer_name_or_path: Optional[str], out_dir,
                      **config_kwargs) -> Path:
    """Write config.json + model.safetensors (+ tokenizer files) so lm-eval can load `out_dir` with `--model hf`.

    The tokenizer is optional: when `tokenizer_name_or_path` is None or cannot be loaded (offline), only the model
    is written and the tokenizer files must be copied into `out_dir` before running lm-eval."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    config = FlyAIHFConfig.from_lm_config(cfg, **config_kwargs)
    if tokenizer_name_or_path is not None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
        except Exception as e:                                        # no hub access, missing files, ...
            warnings.warn(f"tokenizer {tokenizer_name_or_path!r} not exported ({type(e).__name__}: {e}); "
                          f"copy tokenizer files into {out} before running lm-eval", stacklevel=2)
        else:
            tok.save_pretrained(out)
            for k in ("pad_token_id", "bos_token_id", "eos_token_id"):
                if getattr(config, k, None) is None and getattr(tok, k, None) is not None:
                    setattr(config, k, getattr(tok, k))
    FlyAIForCausalLM(config, model=model).save_pretrained(out, safe_serialization=True)
    return out
