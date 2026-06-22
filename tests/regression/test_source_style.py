import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("src", "scripts", "code")


def _project_sources():
    for root_name in SOURCE_ROOTS:
        yield from (PROJECT_ROOT / root_name).rglob("*.py")


class TestSourceStyle(unittest.TestCase):
    def test_modules_use_zero_argument_super(self):
        old_super = re.compile(r"super\([A-Za-z_][A-Za-z0-9_]*, self\)\.__init__")
        offenders = [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in _project_sources()
            if old_super.search(path.read_text(encoding="utf-8"))
        ]

        self.assertEqual([], offenders)

    def test_sources_omit_stale_commented_code_blocks(self):
        stale_patterns = (
            "self.adapters",
            "if adapter is not None",
            "# class Ours",
            "# class GraphNetBlock",
            "#         super(GraphNetBlock",
            "#         return X_new",
            "#         return X, E, None, None",
            "# pooler = Pooler",
            "# M = torch.scatter_add",
        )
        offenders = [
            f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}"
            for path in _project_sources()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if any(pattern in line for pattern in stale_patterns)
        ]

        self.assertEqual([], offenders)

    def test_sources_avoid_index_only_range_len_loops(self):
        indexed_filter_comprehension = re.compile(r"\[[^\n\]]+ for \w+ in range\(len\(")
        offenders = [
            f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}"
            for path in _project_sources()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if (
                indexed_filter_comprehension.search(line)
                or "for i in range(len(chunk))" in line
                or "steps = range(len(" in line
            )
        ]

        self.assertEqual([], offenders)

    def test_sources_iterate_dict_keys_directly(self):
        explicit_keys = re.compile(r"\.keys\(\)")
        offenders = [
            f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}"
            for path in _project_sources()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if explicit_keys.search(line)
        ]

        self.assertEqual([], offenders)

    def test_sources_use_lightweight_empty_checks_and_repeated_lists(self):
        verbose_empty_check = re.compile(r"len\([^)\n]+\)\s*(?:>|==)\s*0")
        repeated_value_comprehension = re.compile(
            r"\[(?:None|True|False|[-+]?\d+(?:\.\d+)?|['\"][^'\"]*['\"])\s+for _ in range\("
        )
        offenders = [
            f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}"
            for path in _project_sources()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if verbose_empty_check.search(line) or repeated_value_comprehension.search(line)
        ]

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
