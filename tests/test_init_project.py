from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InitProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module("init_project_test", "scripts/init_project.py")

    def test_normalize_project_name_slugifies_input(self) -> None:
        self.assertEqual(
            self.module.normalize_project_name("My Cool Project"),
            "my-cool-project",
        )

    def test_normalize_project_name_rejects_invalid_start(self) -> None:
        with self.assertRaisesRegex(Exception, "start with a letter"):
            self.module.normalize_project_name("123-template")

    def test_package_name_from_project_name_uses_underscores(self) -> None:
        self.assertEqual(
            self.module.package_name_from_project_name("my-cool-project"),
            "my_cool_project",
        )

    def test_prompt_project_name_accepts_folder_name(self) -> None:
        with patch("builtins.input", side_effect=["y"]):
            self.assertEqual(
                self.module.prompt_project_name("my-cool-project"),
                "my-cool-project",
            )

    def test_prompt_project_name_asks_for_replacement_and_confirms(self) -> None:
        with patch("builtins.input", side_effect=["n", "My Cool Project", "y"]):
            self.assertEqual(
                self.module.prompt_project_name("template-repo"),
                "my-cool-project",
            )


if __name__ == "__main__":
    unittest.main()
