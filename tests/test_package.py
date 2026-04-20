from __future__ import annotations

import importlib
import unittest


class PackageTests(unittest.TestCase):
    def test_package_exposes_version(self) -> None:
        module = importlib.import_module("template_repo")
        self.assertEqual(module.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
