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

Prepare an unused environment; do not apply from training ranks. Source checkouts retain
Git three-way merge support. Wheels only receive runtime files, not Megatron's unit tests.
Application is prepared and checked in a temporary directory before any target is written;
conflicts leave the checkout, including its index, unchanged. --check only verifies the patch.
"""
import argparse
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterator, List, Optional

PATCH = pathlib.Path(__file__).resolve().parent.parent / 'patches' / 'megatron_glm53_dev.patch'


def megatron_root() -> pathlib.Path:
    """Locate the package without importing megatron.core or initializing CUDA dependencies."""
    spec = importlib.util.find_spec('megatron')
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (pathlib.Path(location) / 'core' / '__init__.py').is_file():
                return pathlib.Path(location).resolve().parent
    raise RuntimeError('Megatron is not installed; install the supported Megatron version first.')


def git_root(root: pathlib.Path) -> bool:
    """A wheel inside another repository is not a Megatron source checkout."""
    if shutil.which('git') is None:
        return False
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
    return result.returncode == 0 and pathlib.Path(result.stdout.strip()).resolve() == root


def patch_files(source_checkout: bool) -> Dict[str, str]:
    """Select existing-file diffs, preserving test changes only for source checkouts."""
    entries = {}
    for section in re.split(r'(?=^diff --git )', PATCH.read_text(), flags=re.MULTILINE):
        if not section.strip():
            continue
        match = re.match(r'diff --git a/(\S+) b/\1\n', section)
        if match is None:
            raise RuntimeError('Unsupported patch header')
        name = match[1]
        if (not name.startswith(('megatron/core/', 'tests/')) or '..' in pathlib.PurePosixPath(name).parts
                or name in entries or f'\n--- a/{name}\n+++ b/{name}\n' not in section):
            raise RuntimeError(f'Unsupported patch target: {name}')
        if source_checkout or name.startswith('megatron/core/'):
            entries[name] = section
    if not entries:
        raise RuntimeError('Empty runtime patch')
    return entries


def run_patch(root: pathlib.Path,
              text: str,
              reverse: bool = False,
              dry_run: bool = True) -> subprocess.CompletedProcess:
    command = ['patch', '--batch', '--force', '--fuzz=0', '--no-backup-if-mismatch', '-p1']
    command += ['--reverse'] if reverse else ['--forward']
    if dry_run:
        command.append('--dry-run')
    return subprocess.run(command, cwd=root, input=text, capture_output=True, text=True)


def patch_hunks(entries: Dict[str, str]) -> Iterator[str]:
    """Check each hunk so a partially applied single file cannot pass as a fresh base."""
    for section in entries.values():
        parts = re.split(r'(?=^@@ )', section, flags=re.MULTILINE)
        if len(parts) < 2:
            raise RuntimeError('Patch target has no hunks')
        for hunk in parts[1:]:
            yield parts[0] + hunk


def prepare_patch(root: pathlib.Path, snapshot: pathlib.Path, entries: Dict[str, str], source_checkout: bool) -> None:
    """Apply off-target, allowing a clean Git three-way merge when context has drifted."""
    text = ''.join(entries.values())
    merged = False
    result = run_patch(snapshot, text)
    if result.returncode == 0:
        result = run_patch(snapshot, text, dry_run=False)
    elif source_checkout:
        merged = True
        # Use a private index and worktree; borrow only objects needed for the merge base.
        common = subprocess.run(['git', 'rev-parse', '--git-common-dir'],
                                cwd=root,
                                capture_output=True,
                                text=True,
                                check=True)
        objects = (root / common.stdout.strip() / 'objects').resolve()
        subprocess.run(['git', 'init', '-q', str(snapshot)], check=True)
        subprocess.run(['git', 'add', '--', *entries], cwd=snapshot, check=True)
        env = dict(os.environ, GIT_ALTERNATE_OBJECT_DIRECTORIES=str(objects))
        result = subprocess.run(['git', 'apply', '--3way', '--whitespace=nowarn', '-'],
                                cwd=snapshot,
                                input=text,
                                capture_output=True,
                                text=True,
                                env=env)
    if result.returncode:
        raise RuntimeError(f'Patch conflicts; no target files changed.\n{result.stdout}{result.stderr}')
    # Git verifies a three-way result itself; its merged context may differ from the diff.
    if not merged and run_patch(snapshot, text, reverse=True).returncode:
        raise RuntimeError('Incomplete patch result; no target files changed.')


def write_files(root: pathlib.Path, snapshot: pathlib.Path, originals: Dict[str, bytes]) -> None:
    """Restore original contents if publishing a prepared patch fails."""
    for name, content in originals.items():
        target = root / name
        if target.resolve() != target or target.read_bytes() != content:
            raise RuntimeError(f'Target changed during patch preparation: {name}')
    written = []
    try:
        for name in originals:
            written.append(name)
            (root / name).write_bytes((snapshot / name).read_bytes())
    except BaseException:
        for name in written:
            (root / name).write_bytes(originals[name])
        raise


def apply_patch(root: pathlib.Path, check_only: bool = False) -> None:
    root = root.resolve()
    source_checkout = git_root(root)
    entries = patch_files(source_checkout)
    originals = {}
    for name in entries:
        target = root / name
        if not target.is_file() or target.resolve() != target:
            raise RuntimeError(f'Missing or symlinked patch target: {target}')
        originals[name] = target.read_bytes()
    with tempfile.TemporaryDirectory(prefix='megatron-patch-') as directory:
        snapshot = pathlib.Path(directory)
        for name, content in originals.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if run_patch(snapshot, ''.join(entries.values()), reverse=True).returncode == 0:
            print(f'already applied (all {len(entries)} files verified): {root}')
            return
        partial = any(run_patch(snapshot, hunk, reverse=True).returncode == 0 for hunk in patch_hunks(entries))
        if not source_checkout:
            if check_only:
                raise RuntimeError(f'Patch not fully applied: {root}')
            if partial:
                raise RuntimeError(f'Partially applied patch; no target files changed: {root}')
        prepare_patch(root, snapshot, entries, source_checkout)
        # A three-way application to an already merged source tree is a no-op, even when
        # unrelated edits changed the patch context and reverse dry-run could not match it.
        if all((snapshot / name).read_bytes() == content for name, content in originals.items()):
            print(f'already applied (all {len(entries)} files verified): {root}')
            return
        if check_only:
            raise RuntimeError(f'Patch not fully applied: {root}')
        if partial:
            raise RuntimeError(f'Partially applied patch; no target files changed: {root}')
        write_files(root, snapshot, originals)
    print(f'applied and verified all {len(entries)} files: {root}')


def is_applied(root: pathlib.Path) -> bool:
    try:
        apply_patch(pathlib.Path(root), check_only=True)
    except (RuntimeError, OSError, subprocess.CalledProcessError):
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, help='Megatron source root; defaults to installed package')
    parser.add_argument('--check', action='store_true', help='Verify all selected patch hunks without writing')
    args = parser.parse_args(argv)
    try:
        apply_patch(args.root if args.root is not None else megatron_root(), args.check)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
