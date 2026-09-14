# Copyright (c) ModelScope Contributors. All rights reserved.
"""Apply the Megatron-LM patch that GLM-5.3 needs to the installed megatron.

    pip install git+https://github.com/NVIDIA/Megatron-LM.git@dev
    python -m mcore_bridge.tools.apply_megatron_patch

The patch (`patches/megatron_glm53_dev.patch`, generated against dev ee743d3ef) backports the
unmerged Megatron-LM #7054 GLM numerics -- KDA two-stage gates, mHC precision, KPool DSA -- plus two
Megatron fixes that are not model specific (expert shards deduplicated over the expert TP group when
the gradient norm is computed, and `allreduce` preserved on fp32 master parameters). Both parts go
away once they land upstream.

#7054 is tracked by content, not by branch: the numerics come from its `be805e55`, and the chunked
index-score computation from its current head `3fceb0715`. The chunking matters at real scale --
with `index_n_heads=32` the un-chunked fp32 `[seqlen_q, batch, heads, seqlen_k]` tensor is 8 GiB at
sequence 8192 and 128 GiB when packed to 32768 -- and is bit-identical to the un-chunked version.
Deliberately not taken from that head: its context-parallel gather of the kpool gate score (this
model rejects CP) and main's different indexer-loss gradient formula (not a memory fix, and it would
change deepseek_v4 / glm_moe_dsa, which do use the indexer loss).

Idempotent. `git apply --3way` is used inside a git checkout, so an upstream edit outside the lines
we change does not block it and a real overlap is left as a visible conflict rather than dropped;
`patch(1)` is used for a pip-installed megatron, which is not a git tree.
"""
import importlib.util
import pathlib
import subprocess
import sys

PATCH = pathlib.Path(__file__).resolve().parent.parent / 'patches' / 'megatron_glm53_dev.patch'
MARKER_FILE = 'megatron/core/transformer/transformer_config.py'
MARKER_SYMBOL = 'kda_two_stage_gates'
INSTALL_HINT = 'pip install -U git+https://github.com/NVIDIA/Megatron-LM.git@dev'


def megatron_root():
    """The directory the patch's `megatron/core/...` paths are relative to."""
    spec = importlib.util.find_spec('megatron.core')
    if spec is None or not spec.origin:
        sys.exit(f'megatron is not installed; install it first:\n  {INSTALL_HINT}')
    return pathlib.Path(spec.origin).resolve().parents[2]  # <root>/megatron/core/__init__.py


def is_applied(root):
    target = pathlib.Path(root) / MARKER_FILE
    return target.is_file() and MARKER_SYMBOL in target.read_text(errors='ignore')


def main():
    root = megatron_root()
    if is_applied(root):
        print(f'already applied: {root}')
        return 0
    in_git = not subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                                capture_output=True).returncode
    command = (['git', 'apply', '--3way', str(PATCH)] if in_git else
               ['patch', '-p1', '--forward', '-i', str(PATCH)])
    proc = subprocess.run(command, cwd=root, capture_output=True, text=True)
    print(proc.stdout + proc.stderr, end='')
    if proc.returncode:
        print(f'could not apply {PATCH.name} to {root}. If megatron drifted from the commit above, '
              f'update it and retry:\n  {INSTALL_HINT}', file=sys.stderr)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
