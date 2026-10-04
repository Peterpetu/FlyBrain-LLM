#!/usr/bin/env bash
# Fetch the raw MaleCNS v1.0 connectome (via the garyb9/fly-drone CC-BY-4.0 mirror),
# the official Janelia body-annotation table (for provenance cross-check), and the
# small public text corpora used by the CPU-scale experiments.
# Nothing here is committed; results in the repo reference these by SHA-256.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/malecns" "$ROOT/data/text"
M=https://raw.githubusercontent.com/garyb9/fly-drone/main/data/malecns
for f in neurons.bin graph.bin cells.json groups.json manifest.json ATTRIBUTION.md; do
  [ -f "$ROOT/data/malecns/$f" ] || curl -L --retry 4 -o "$ROOT/data/malecns/$f" "$M/$f"
done
# Official annotations (Berg et al. 2026, MaleCNS v1.0). SHA-256 is recorded in manifest.json.
A=body-annotations-male-cns-v1.0-minconf-0.5.feather
[ -f "$ROOT/data/malecns/$A" ] || curl -L --retry 4 -o "$ROOT/data/malecns/$A" \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/$A"
# Text corpora
T="$ROOT/data/text"
[ -f "$T/tinyshakespeare.txt" ] || curl -L -o "$T/tinyshakespeare.txt" \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
for s in train valid test; do
  [ -f "$T/wikitext2.$s.txt" ] || curl -L -o "$T/wikitext2.$s.txt" \
    https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$s.txt
  [ -f "$T/ptb.$s.txt" ] || curl -L -o "$T/ptb.$s.txt" \
    https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.$s.txt
done
# text8 / enwik8 (Hutter prize): http://mattmahoney.net/dc/text8.zip , enwik8.zip
# were not reachable from the authoring environment; fetch manually if you have access.
cd "$ROOT/data" && sha256sum malecns/*.bin malecns/*.feather text/*.txt > SHA256SUMS
echo "done; checksums in data/SHA256SUMS"
