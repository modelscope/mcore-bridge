#!/usr/bin/env bash
# Parity matrix for the MoE expert-export fast paths: every topology needs its own world size,
# so each case is one torchrun. Run this after touching gpt_bridge expert export.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

python -m pytest tests/test_moe_expert_export.py -q -p no:cacheprovider "$@"

for case in pp2:2 ep2:2 etp2:2 pp2ep2:4 etp2ep2:4; do
    topology=${case%%:*}
    world=${case##*:}
    echo "== MOE_EXPORT_TOPOLOGY=${topology} (world ${world})"
    # A topology that does not fit this machine skips, and a skip still exits 0: require a pass.
    out=$(MOE_EXPORT_TOPOLOGY="${topology}" torchrun --nproc_per_node="${world}" \
        -m pytest tests/test_moe_expert_export.py -q -rs -p no:cacheprovider -k parallel "$@" 2>&1) || {
        echo "${out}"
        exit 1
    }
    echo "${out}"
    grep -q '1 passed' <<< "${out}" || {
        echo "!! ${topology}: case did not run (skipped or deselected)"
        exit 1
    }
done
