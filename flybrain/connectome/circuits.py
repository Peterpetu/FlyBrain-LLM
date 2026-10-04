"""Extraction and verification of identified sub-connectomes.

Two circuits with experimentally established function are extracted per hemisphere:

* Mushroom body (MB): olfactory projection neurons (PN) -> Kenyon cells (KC) -> mushroom
  body output neurons (MBON), with APL/DPM feedback and PAM/PPL1 dopaminergic neurons.
* Central complex (CX) heading circuit: EPG, PEN_a, PEN_b, PEG, Delta7 (plus EL, ExR).

Compartment (MB) and protocerebral-bridge glomerulus (CX) labels are parsed from the
official MaleCNS `instance` strings (e.g. "MBON01(y5B'2a)_R", "EPG(PB08)_L3",
"Delta7(PB15)_L8R1R9_L"). Every extraction returns a `verification` dict comparing measured
wiring numbers against published references (hemibrain: Li et al. 2020 eLife; Hulse et al.
2021 eLife; Caron 2013; Lin 2014), so a mislabelled circuit fails loudly.

All matrices are anatomical synaptic-contact counts (edges >= 3 contacts); see
LIMITATIONS.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .io import Connectome, load_official_annotations, DATA_DIR

CIRCUIT_DIR = DATA_DIR / "circuits"

_PAREN = re.compile(r"\((.*?)\)")
_GLOM = re.compile(r"_([LR])(\d)$")            # EPG(PB08)_L3 -> L, 3
_D7 = re.compile(r"_((?:[LR]\d)+)_([LR])$")     # Delta7(PB15)_L8R1R9_L -> L8R1R9, L


def _instances(c: Connectome, annotations=None) -> np.ndarray:
    a = annotations if annotations is not None else load_official_annotations()
    a = a.set_index("bodyId")
    m = a.reindex(c.body_id.astype(np.int64))
    return m["instance"].fillna("").values.astype(object)


def _block(c: Connectome, src: np.ndarray, dst: np.ndarray) -> sp.csr_matrix:
    return c.W[src][:, dst].tocsr()


# --------------------------------------------------------------------------------------
# Mushroom body
# --------------------------------------------------------------------------------------
def extract_mushroom_body(c: Connectome, side: str = "R", annotations=None) -> dict:
    inst = _instances(c, annotations)
    on_side = c.side == side
    KC = np.flatnonzero(on_side & np.char.startswith(c.cell_type.astype(str), "KC"))
    APL = np.flatnonzero(on_side & (c.cell_type == "APL"))
    DPM = np.flatnonzero(on_side & (c.cell_type == "DPM"))
    MBON = np.flatnonzero(on_side & np.char.startswith(c.cell_type.astype(str), "MBON"))
    PAM = np.flatnonzero(on_side & np.char.startswith(c.cell_type.astype(str), "PAM"))
    PPL1 = np.flatnonzero(on_side & np.char.startswith(c.cell_type.astype(str), "PPL1"))
    DAN = np.concatenate([PAM, PPL1])
    # PNs: any cell whose type ends in "PN" (antennal-lobe projection neuron nomenclature)
    # with at least one >=3-contact edge onto a KC of this hemisphere.
    pn_type = np.array([t.endswith("PN") for t in c.cell_type], dtype=bool)
    to_kc = np.asarray((c.W[:, KC] > 0).sum(1)).ravel() > 0
    PN = np.flatnonzero(pn_type & to_kc & on_side)

    W_pn_kc = _block(c, PN, KC)
    W_kc_apl, W_apl_kc = _block(c, KC, APL), _block(c, APL, KC)
    W_kc_dpm, W_dpm_kc = _block(c, KC, DPM), _block(c, DPM, KC)
    W_kc_kc = _block(c, KC, KC)
    W_kc_mbon = _block(c, KC, MBON)
    W_dan_kc = _block(c, DAN, KC)
    W_dan_mbon = _block(c, DAN, MBON)
    W_mbon_dan = _block(c, MBON, DAN)
    W_mbon_mbon = _block(c, MBON, MBON)
    W_mbon_kc = _block(c, MBON, KC)

    # compartments from instance strings, e.g. "MBON01(y5B'2a)_R" -> "y5B'2a"
    def comp(i):
        m = _PAREN.search(inst[i]); return m.group(1) if m else ""
    mbon_comp = np.array([comp(i) for i in MBON], dtype=object)
    dan_comp = np.array([comp(i) for i in DAN], dtype=object)
    pn_types = c.cell_type[PN]
    kc_types = c.cell_type[KC]

    fan_in_neurons = np.asarray((W_pn_kc > 0).sum(0)).ravel()
    # fan-in in PN *types* (glomeruli): count distinct types per KC
    T = sp.csr_matrix((np.ones(len(PN)), (np.searchsorted(np.unique(pn_types), pn_types), np.arange(len(PN)))),
                      shape=(len(np.unique(pn_types)), len(PN)))
    fan_in_types = np.asarray(((T @ (W_pn_kc > 0).astype(float)) > 0).sum(0)).ravel()
    fan_out = np.asarray((W_pn_kc > 0).sum(1)).ravel()
    apl_cov = float((np.asarray((W_apl_kc > 0).sum(0)).ravel() > 0).mean()) if len(APL) else 0.0
    mbon_fan_in = np.asarray((W_kc_mbon > 0).sum(0)).ravel()
    kc_with_pn = fan_in_neurons > 0

    stats = {
        "side": side, "n_PN": int(len(PN)), "n_PN_types": int(len(np.unique(pn_types))),
        "n_KC": int(len(KC)), "n_APL": int(len(APL)), "n_DPM": int(len(DPM)),
        "n_MBON": int(len(MBON)), "n_PAM": int(len(PAM)), "n_PPL1": int(len(PPL1)),
        "n_compartments_MBON": int(len(set(mbon_comp) - {""})),
        "frac_KC_with_PN_input": float(kc_with_pn.mean()),
        "KC_fan_in_PN_neurons_mean": float(fan_in_neurons[kc_with_pn].mean()),
        "KC_fan_in_PN_types_mean": float(fan_in_types[kc_with_pn].mean()),
        "KC_fan_in_PN_types_median": float(np.median(fan_in_types[kc_with_pn])),
        "PN_fan_out_KC_mean": float(fan_out.mean()),
        "PN_KC_contacts": int(W_pn_kc.sum()), "PN_KC_edges": int(W_pn_kc.nnz),
        "APL_covers_frac_KC": apl_cov,
        "KC_MBON_edges": int(W_kc_mbon.nnz), "MBON_fan_in_KC_mean": float(mbon_fan_in.mean()),
        "DAN_KC_edges": int(W_dan_kc.nnz), "DAN_MBON_edges": int(W_dan_mbon.nnz),
        "MBON_DAN_feedback_edges": int(W_mbon_dan.nnz), "MBON_DAN_feedback_contacts": int(W_mbon_dan.sum()),
        "MBON_MBON_edges": int(W_mbon_mbon.nnz), "KC_KC_edges": int(W_kc_kc.nnz),
        "APL_nt": list(map(str, c.nt[APL])), "DPM_nt": list(map(str, c.nt[DPM])),
        "MBON_nt_counts": {k: int(v) for k, v in zip(*np.unique(c.nt[MBON], return_counts=True))},
        "KC_types": {k: int(v) for k, v in zip(*np.unique(kc_types, return_counts=True))},
        "PN_nt_counts": {k: int(v) for k, v in zip(*np.unique(c.nt[PN], return_counts=True))},
        "frac_KC_inhibitory_flag": float(c.inhibitory[KC].mean()),
    }
    # --- verification against published wiring (hemibrain, Li et al. 2020; Caron 2013; Lin 2014)
    ver = {
        "KC_count_~2000_per_hemisphere": (1500 <= len(KC) <= 2600, len(KC)),
        "APL_single_neuron_GABA": (len(APL) == 1 and all(n == "gaba" for n in c.nt[APL]), list(map(str, c.nt[APL]))),
        "APL_inhibits_>95%_KC": (apl_cov > 0.95, apl_cov),
        "KC_fan_in_PN_types_5-7_(Caron2013)": (4.0 <= stats["KC_fan_in_PN_types_mean"] <= 8.0, stats["KC_fan_in_PN_types_mean"]),
        "MBON_~20-40_types_per_side": (15 <= len(MBON) <= 60, len(MBON)),
        "MBON_DAN_feedback_exists": (W_mbon_dan.nnz > 50, W_mbon_dan.nnz),
        "DAN_target_MBON_dendrites_(compartment_coupling)": (W_dan_mbon.nnz > 100, W_dan_mbon.nnz),
        "PAM_>>PPL1_in_count": (len(PAM) > 5 * len(PPL1), (len(PAM), len(PPL1))),
        "KC_excitatory_(cholinergic)": (stats["frac_KC_inhibitory_flag"] < 0.05, stats["frac_KC_inhibitory_flag"]),
    }
    return {
        "idx": {"PN": PN, "KC": KC, "APL": APL, "DPM": DPM, "MBON": MBON, "PAM": PAM, "PPL1": PPL1, "DAN": DAN},
        "W": {"pn_kc": W_pn_kc, "kc_apl": W_kc_apl, "apl_kc": W_apl_kc, "kc_dpm": W_kc_dpm, "dpm_kc": W_dpm_kc,
              "kc_kc": W_kc_kc, "kc_mbon": W_kc_mbon, "dan_kc": W_dan_kc, "dan_mbon": W_dan_mbon,
              "mbon_dan": W_mbon_dan, "mbon_mbon": W_mbon_mbon, "mbon_kc": W_mbon_kc},
        "labels": {"pn_type": pn_types, "kc_type": kc_types, "mbon_type": c.cell_type[MBON], "mbon_comp": mbon_comp,
                   "dan_type": c.cell_type[DAN], "dan_comp": dan_comp, "mbon_nt": c.nt[MBON], "dan_is_pam": np.r_[np.ones(len(PAM), bool), np.zeros(len(PPL1), bool)]},
        "stats": stats, "verification": ver,
    }


# --------------------------------------------------------------------------------------
# Central complex heading circuit
# --------------------------------------------------------------------------------------
def _pb_glomeruli(s: str) -> list[tuple[str, int]]:
    """Parse PB glomeruli from an instance string. Delta7 lists several."""
    m = _D7.search(s)
    if m:
        return [(g[0], int(g[1])) for g in re.findall(r"[LR]\d", m.group(1))]
    m = _GLOM.search(s)
    return [(m.group(1), int(m.group(2)))] if m else []


def pb_glomerulus_to_angle(side: str, g: int) -> float:
    """Heading angle (radians) represented by PB glomerulus `side g`.

    Convention (verified from the data, not assumed): the ellipsoid body is tiled by 8
    heading sectors and each PB half has 9 glomeruli with the end glomeruli duplicating
    (L9 = L1, R9 = R1). The two halves pair as L_i = R_(9-i): this is read directly off
    the Delta7 output labels (e.g. Delta7 "L8R1R9", "L1L9R8", "L4R5", "L5R4"), whose
    2-3 output glomeruli all represent one heading (Wolff & Rubin 2018; Hulse 2021).
    Angle = 2*pi*(i-1)/8 for L_i and 2*pi*(8-j)/8 for R_j, wrapped mod 2*pi."""
    i = g if side == "L" else 9 - g
    return (2 * np.pi * (i - 1) / 8.0) % (2 * np.pi)


def extract_central_complex(c: Connectome, annotations=None) -> dict:
    inst = _instances(c, annotations)
    types = c.cell_type.astype(str)
    pops = {
        "EPG": np.flatnonzero(types == "EPG"),
        "PEN_a": np.flatnonzero(np.char.startswith(types, "PEN_a")),
        "PEN_b": np.flatnonzero(np.char.startswith(types, "PEN_b")),
        "PEG": np.flatnonzero(types == "PEG"),
        "Delta7": np.flatnonzero(types == "Delta7"),
        "EL": np.flatnonzero(types == "EL"),
        "ExR": np.flatnonzero(np.char.startswith(types, "ExR")),
    }
    core = ["EPG", "PEN_a", "PEN_b", "PEG", "Delta7"]
    all_idx = np.concatenate([pops[p] for p in core])
    pop_of = np.concatenate([[p] * len(pops[p]) for p in core])
    W = _block(c, all_idx, all_idx)
    # angles
    ang = np.full(len(all_idx), np.nan)
    glom_side = np.array([""] * len(all_idx), dtype=object)
    for k, i in enumerate(all_idx):
        gl = _pb_glomeruli(inst[i])
        if gl:
            a = np.array([pb_glomerulus_to_angle(s, g) for s, g in gl])
            # circular mean (Delta7 spans 2-3 glomeruli that should share a heading)
            ang[k] = np.angle(np.exp(1j * a).mean()) % (2 * np.pi)
            glom_side[k] = "".join(f"{s}{g}" for s, g in gl)
    # ---- verification: connectivity as a function of angular offset
    def offset_profile(src_pop, dst_pop, nbins=8):
        s = np.flatnonzero(pop_of == src_pop); d = np.flatnonzero(pop_of == dst_pop)
        B = W[s][:, d].toarray()
        prof = np.zeros(nbins); cnt = np.zeros(nbins)
        for a_i, i in enumerate(s):
            for b_j, j in enumerate(d):
                if np.isnan(ang[i]) or np.isnan(ang[j]):
                    continue
                dlt = (ang[j] - ang[i]) % (2 * np.pi)
                b = int(np.round(dlt / (2 * np.pi / nbins))) % nbins
                prof[b] += B[a_i, b_j]; cnt[b] += 1
        return prof / np.maximum(cnt, 1)
    prof = {f"{a}->{b}": offset_profile(a, b).tolist() for a, b in
            [("EPG", "EPG"), ("EPG", "PEN_a"), ("EPG", "PEN_b"), ("PEN_a", "EPG"), ("PEN_b", "EPG"),
             ("EPG", "Delta7"), ("Delta7", "EPG"), ("Delta7", "PEN_a"), ("Delta7", "Delta7"), ("EPG", "PEG"), ("PEG", "EPG")]}
    # PEN neurons of the two PB halves should shift the bump in opposite directions
    side_of = np.array([("L" if "_L" in inst[i][-3:] else "R") for i in all_idx], dtype=object)
    def offset_profile_side(src_pop, dst_pop, side, nbins=8):
        keep = (pop_of == src_pop) & (side_of == side)
        s_ = np.flatnonzero(keep); d = np.flatnonzero(pop_of == dst_pop)
        B = W[s_][:, d].toarray(); pr = np.zeros(nbins); cn = np.zeros(nbins)
        for a_i, i in enumerate(s_):
            for b_j, j in enumerate(d):
                dlt = (ang[j] - ang[i]) % (2 * np.pi); b = int(np.round(dlt / (2 * np.pi / nbins))) % nbins
                pr[b] += B[a_i, b_j]; cn[b] += 1
        return pr / np.maximum(cn, 1)
    for pen in ("PEN_a", "PEN_b"):
        for sd in ("L", "R"):
            prof[f"{pen}[{sd}]->EPG"] = offset_profile_side(pen, "EPG", sd).tolist()
    p = {k: np.array(v) for k, v in prof.items()}
    d7 = p["Delta7->EPG"]
    # effective EPG -> Delta7 -> EPG inhibition profile (two-step), by angular offset
    e = np.flatnonzero(pop_of == "EPG"); d = np.flatnonzero(pop_of == "Delta7")
    two = (W[e][:, d] @ W[d][:, e]).toarray(); eff = np.zeros(8); cnt = np.zeros(8)
    for a_i, i in enumerate(e):
        for b_j, j in enumerate(e):
            dlt = (ang[j] - ang[i]) % (2 * np.pi); b = int(np.round(dlt / (2 * np.pi / 8))) % 8
            eff[b] += two[a_i, b_j]; cnt[b] += 1
    eff = eff / np.maximum(cnt, 1); prof["EPG->Delta7->EPG (effective inhibition)"] = eff.tolist()
    penL = p["PEN_a[L]->EPG"] + p["PEN_b[L]->EPG"]; penR = p["PEN_a[R]->EPG"] + p["PEN_b[R]->EPG"]
    # Data-driven PEN shift: angle of each PEN's EPG *inputs* vs angle of its EPG *outputs*
    # (independent of PEN glomerulus labels, which are numbered with a one-glomerulus
    # offset relative to EPG labels because PENs innervate PB glomeruli 2-9).
    pen_shift = {}
    for sd in ("L", "R"):
        sh = []
        for i in np.flatnonzero(((pop_of == "PEN_a") | (pop_of == "PEN_b")) & (side_of == sd)):
            win = W[e][:, i].toarray().ravel(); wout = W[i][:, e].toarray().ravel()
            if win.sum() == 0 or wout.sum() == 0:
                continue
            a_in = np.angle((win * np.exp(1j * ang[e])).sum()); a_out = np.angle((wout * np.exp(1j * ang[e])).sum())
            sh.append(np.angle(np.exp(1j * (a_out - a_in))))
        pen_shift[sd] = float(np.degrees(np.mean(sh)))
    prof["PEN output-minus-input shift (deg)"] = [pen_shift["L"], pen_shift["R"]]
    ver = {
        "EPG_46_(hemibrain 46)": (len(pops["EPG"]) == 46, len(pops["EPG"])),
        "PEN_42_(hemibrain 20 PEN_a + 22 PEN_b)": (len(pops["PEN_a"]) + len(pops["PEN_b"]) == 42, (len(pops["PEN_a"]), len(pops["PEN_b"]))),
        "Delta7_42_(hemibrain 42)": (len(pops["Delta7"]) == 42, len(pops["Delta7"])),
        "PEG_18_(hemibrain 18)": (len(pops["PEG"]) == 18, len(pops["PEG"])),
        "EPG_cholinergic": (all(n == "acetylcholine" for n in c.nt[pops["EPG"]]), None),
        "Delta7_glutamatergic_(inhibitory)": (all(n == "glutamate" for n in c.nt[pops["Delta7"]]), None),
        "EPG->EPG_peaks_at_offset0": (int(np.argmax(p["EPG->EPG"])) == 0, p["EPG->EPG"].round(1).tolist()),
        # The canonical ~45 deg PEN shift cannot be isolated here: EPG->PEN contacts occur both
        # in the PB (dendritic input) and in the EB (axo-axonal, at the shifted output
        # location) and this mirror carries no per-synapse ROI. We verify the sign structure
        # (opposite per side) and report the contact-weighted magnitude. See LIMITATIONS.md.
        "PEN_output_vs_input_shift_opposite_sign_per_side": (
            (np.sign(pen_shift["L"]) != np.sign(pen_shift["R"])) and all(abs(v) > 3 for v in pen_shift.values()), pen_shift),
        "EPG->Delta7_input_peaks_opposite_(offset4)": (int(np.argmax(p["EPG->Delta7"])) == 4, p["EPG->Delta7"].round(1).tolist()),
        "Delta7_output_local_(offset0_max)": (int(np.argmax(d7)) == 0, d7.round(1).tolist()),
        "effective_EPG->D7->EPG_inhibition_min_at_0_max_opposite": (int(np.argmin(eff)) == 0 and int(np.argmax(eff)) in (3, 4, 5), eff.round(0).tolist()),
        "PEN_L_and_PEN_R_shift_bump_opposite_ways": ((penL[1] > penL[7]) != (penR[1] > penR[7]), {"L": penL.round(1).tolist(), "R": penR.round(1).tolist()}),
    }
    stats = {k: int(len(v)) for k, v in pops.items()}
    stats.update({"n_core": int(len(all_idx)), "edges_core": int(W.nnz), "contacts_core": int(W.sum()),
                  "frac_with_angle": float(np.isfinite(ang).mean()), "nt": {p_: {k: int(v) for k, v in zip(*np.unique(c.nt[pops[p_]], return_counts=True))} for p_ in core}})
    return {"idx": {p_: pops[p_] for p_ in pops}, "core_idx": all_idx, "pop_of": pop_of, "W": W, "angle": ang,
            "glomeruli": glom_side, "sign": c.sign[all_idx], "nt": c.nt[all_idx], "profiles": prof, "stats": stats, "verification": ver}


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------
def _to_json(o):
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, tuple): return list(o)
    if isinstance(o, dict): return {str(k): _to_json(v) for k, v in o.items()}
    if isinstance(o, list): return [_to_json(v) for v in o]
    return o


def save_circuits(c: Connectome, out_dir: Optional[Path] = None, annotations=None) -> dict:
    out = Path(out_dir or CIRCUIT_DIR); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for side in ("L", "R"):
        mb = extract_mushroom_body(c, side, annotations)
        arrays = {f"W_{k}": v.toarray().astype(np.int32) for k, v in mb["W"].items()}
        arrays.update({f"idx_{k}": v for k, v in mb["idx"].items()})
        arrays.update({f"label_{k}": np.asarray(v).astype(str) if v.dtype == object else v for k, v in mb["labels"].items()})
        arrays["body_id"] = c.body_id
        np.savez_compressed(out / f"mushroom_body_{side}.npz", **arrays)
        report[f"mushroom_body_{side}"] = {"stats": mb["stats"], "verification": _to_json(mb["verification"])}
    cx = extract_central_complex(c, annotations)
    np.savez_compressed(out / "central_complex.npz", W=cx["W"].toarray().astype(np.int32), core_idx=cx["core_idx"],
                        pop_of=cx["pop_of"].astype(str), angle=cx["angle"], glomeruli=cx["glomeruli"].astype(str),
                        sign=cx["sign"], nt=cx["nt"].astype(str), body_id=c.body_id[cx["core_idx"]])
    report["central_complex"] = {"stats": cx["stats"], "verification": _to_json(cx["verification"]), "profiles": cx["profiles"]}
    (out / "circuit_report.json").write_text(json.dumps(_to_json(report), indent=1))
    return report


def load_circuit(name: str, circuit_dir: Optional[Path] = None) -> dict:
    d = Path(circuit_dir or CIRCUIT_DIR)
    z = np.load(d / f"{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}
