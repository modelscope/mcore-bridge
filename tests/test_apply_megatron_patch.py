# Copyright (c) ModelScope Contributors. All rights reserved.
"""Installer regressions using real patch/git executables, without Torch or CUDA."""
import difflib
import hashlib
import importlib.util
import os
import pathlib
import subprocess
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

TOOL = pathlib.Path(__file__).resolve().parents[1] / 'src/mcore_bridge/tools/apply_megatron_patch.py'
spec = importlib.util.spec_from_file_location('apply_megatron_patch', TOOL)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def blob(data: str) -> str:
    content = data.encode()
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def make_diff(name: str, before: str, after: str) -> str:
    body = ''.join(
        difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))
    return f'diff --git a/{name} b/{name}\nindex {blob(before)}..{blob(after)} 100644\n{body}'


class ApplyPatchTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.names = ['megatron/core/transformer_config.py', 'megatron/core/optimizer.py']
        self.before = 'before_context = 1\nvalue = 1\ncontext_a = 1\ncontext_b = 1\nafter_context = 1\n'
        self.after = self.before.replace('value = 1', 'value = 2')
        self.test_name = 'tests/unit_tests/test_example.py'
        for name in self.names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.before)
        self.patch_file = self.root / 'fixture.patch'
        self.patch_file.write_text(''.join(make_diff(name, self.before, self.after)
                                           for name in self.names) + make_diff(self.test_name, self.before, self.after))
        mock = patch.object(installer, 'PATCH', self.patch_file)
        mock.start()
        self.addCleanup(mock.stop)

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True).stdout

    def init_git(self):
        target = self.root / self.test_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.before)
        self.git('init', '-q')
        self.git('add', '--', *self.names, self.test_name)
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'base')

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_wheel_apply_check_repeat_without_test_tree(self):
        self.assertFalse(installer.is_applied(self.root))
        installer.apply_patch(self.root)
        for name in self.names:
            self.assertEqual((self.root / name).read_text(), self.after)
        self.assertFalse((self.root / 'tests').exists())
        expected = self.snapshot()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_marker_alone_is_not_a_complete_installation(self):
        (self.root / self.names[0]).write_text('kda_two_stage_gates = False\n')
        (self.root / self.names[1]).unlink()
        expected = self.snapshot()
        self.assertFalse(installer.is_applied(self.root))
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_patch_rejected(self):
        (self.root / self.names[0]).write_text(self.after)
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_hunks_in_one_file_rejected(self):
        before = self.before + '\n' * 10 + 'second = 1\n'
        after = self.after + '\n' * 10 + 'second = 2\n'
        self.patch_file.write_text(make_diff(self.names[0], before, after))
        (self.root / self.names[0]).write_text(before.replace('value = 1', 'value = 2'))
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_source_rejected_without_index_changes(self):
        self.init_git()
        (self.root / self.names[0]).write_text(self.after)
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_missing_source_test_file_rejected(self):
        self.init_git()
        (self.root / self.test_name).unlink()
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_failed_application_never_writes_target(self):
        expected = self.snapshot()
        original = installer.run_patch

        def fail_apply(root, text, reverse=False, dry_run=True):
            if not dry_run:
                (root / self.names[0]).write_text('partially applied')
                return subprocess.CompletedProcess(['patch'], 1, '', 'simulated failure')
            return original(root, text, reverse, dry_run)

        with patch.object(installer, 'run_patch', fail_apply):
            with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_unrelated_wheel_edits_preserved(self):
        for name in self.names:
            with (self.root / name).open('a') as target:
                target.write('\n# unrelated upstream addition\n')
        installer.apply_patch(self.root)
        self.assertTrue(installer.is_applied(self.root))
        for name in self.names:
            self.assertEqual((self.root / name).read_text(), self.after + '\n# unrelated upstream addition\n')

    def test_conflict_leaves_no_partial_files_or_rejects(self):
        (self.root / self.names[1]).write_text('conflicting content\n')
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_symlink_rejected(self):
        (self.root / self.names[0]).unlink()
        (self.root / self.names[0]).symlink_to(self.root / self.names[1])
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)

    def test_failed_publish_restores_originals(self):
        expected = self.snapshot()
        original = pathlib.Path.write_bytes
        failed = False

        def fail_once(path, data):
            nonlocal failed
            if path == self.root / self.names[1] and not failed:
                failed = True
                original(path, b'partial write')
                raise OSError('simulated write failure')
            return original(path, data)

        with patch.object(pathlib.Path, 'write_bytes', fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_source_applies_test_diff_without_staging(self):
        self.init_git()
        index = (self.root / '.git/index').read_bytes()
        installer.apply_patch(self.root)
        for name in self.names + [self.test_name]:
            self.assertEqual((self.root / name).read_text(), self.after)
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        self.assertEqual(self.git('diff', '--cached'), b'')
        self.assertTrue(installer.is_applied(self.root))

    def test_three_way_preserves_context_changes_and_index(self):
        self.init_git()
        # Change a context line: patch --fuzz=0 fails, while a three-way merge is clean.
        target = self.root / self.names[0]
        target.write_text(self.before.replace('after_context = 1', 'after_context = 9'))
        self.git('add', '--', self.names[0])
        index = (self.root / '.git/index').read_bytes()
        installer.apply_patch(self.root)
        self.assertEqual(target.read_text(), self.after.replace('after_context = 1', 'after_context = 9'))
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        expected = self.snapshot()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_three_way_conflict_preserves_worktree_and_index(self):
        self.init_git()
        (self.root / self.names[0]).write_text(self.before.replace('value = 1', 'value = 99'))
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_wheel_nested_in_git_does_not_require_tests(self):
        self.init_git()
        site = self.root / 'venv/lib/site-packages'
        for name in self.names:
            target = site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.before)
        self.assertFalse(installer.git_root(site))
        installer.apply_patch(site)
        self.assertFalse((site / 'tests').exists())

    def test_find_package_without_importing_core(self):
        location = self.root / 'megatron'
        (location / 'core/__init__.py').write_text('raise RuntimeError("must not import")\n')
        with patch.object(
                installer.importlib.util,
                'find_spec',
                return_value=SimpleNamespace(submodule_search_locations=[str(location)])) as find:
            self.assertEqual(installer.megatron_root(), self.root)
        find.assert_called_once_with('megatron')

    def test_readonly_check_returns_nonzero_for_unpatched_root(self):
        expected = self.snapshot()
        self.assertEqual(installer.main(['--root', str(self.root), '--check']), 1)
        self.assertEqual(expected, self.snapshot())

    def test_packaged_runtime_patch_parses(self):
        with patch.object(installer, 'PATCH', TOOL.parent.parent / 'patches/megatron_glm53_dev.patch'):
            runtime = installer.patch_files(False)
            source = installer.patch_files(True)
        self.assertTrue(all(name.startswith('megatron/core/') for name in runtime))
        self.assertIn('megatron/core/transformer/transformer_config.py', runtime)
        self.assertGreater(len(source), len(runtime))


class PackagedPatchIntegrationTest(unittest.TestCase):
    """Optionally exercise the entire bundled patch against real upstream artifacts."""

    @unittest.skipUnless(
        os.environ.get('MEGATRON_PATCH_BASE_WHEEL'), 'set MEGATRON_PATCH_BASE_WHEEL to the baseline wheel')
    def test_baseline_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with zipfile.ZipFile(os.environ['MEGATRON_PATCH_BASE_WHEEL']) as wheel:
                wheel.extractall(root)
            self.verify_complete_patch(root, source=False)

    @unittest.skipUnless(
        os.environ.get('MEGATRON_PATCH_BASE_SOURCE'), 'set MEGATRON_PATCH_BASE_SOURCE to baseline sources')
    def test_baseline_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            base = pathlib.Path(os.environ['MEGATRON_PATCH_BASE_SOURCE'])
            for name in installer.patch_files(True):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((base / name).read_bytes())
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            self.verify_complete_patch(root, source=True)

    def verify_complete_patch(self, root, source):
        entries = installer.patch_files(source)
        self.assertFalse(installer.is_applied(root))
        installer.apply_patch(root)
        installer.apply_patch(root, check_only=True)
        installer.apply_patch(root)
        for name, diff in entries.items():
            expected = diff.split('index ', 1)[1].split('..', 1)[1].split()[0]
            self.assertTrue(blob((root / name).read_text()).startswith(expected), name)


if __name__ == '__main__':
    unittest.main()
