from __future__ import annotations

import argparse
import json
import keyword
import re
import shutil
import subprocess
import sys
from pathlib import Path


TEMPLATE_PROJECT_NAME = "template-repo"
TEMPLATE_PACKAGE_NAME = "template_repo"
DEFAULT_DESCRIPTION = "Describe your project here."
DEFAULT_OWNER = "jakobmv"
DEFAULT_VISIBILITY = "private"


def normalize_project_name(value: str) -> str:
    project_name = value.strip().lower()
    project_name = re.sub(r"[\s_]+", "-", project_name)
    project_name = re.sub(r"[^a-z0-9-]+", "-", project_name)
    project_name = re.sub(r"-{2,}", "-", project_name).strip("-")
    if not project_name:
        raise argparse.ArgumentTypeError("Project name cannot be empty.")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", project_name):
        raise argparse.ArgumentTypeError(
            "Project name must start with a letter and contain only letters, numbers, or dashes."
        )
    return project_name


def package_name_from_project_name(project_name: str) -> str:
    package_name = project_name.replace("-", "_")
    if keyword.iskeyword(package_name) or not package_name.isidentifier():
        raise ValueError(
            "The derived package name is not a valid Python identifier. "
            "Choose a different project name."
        )
    return package_name


def prompt_yes_no(question: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_project_name(default_name: str) -> str:
    try:
        normalized_default = normalize_project_name(default_name)
    except argparse.ArgumentTypeError:
        normalized_default = None

    if normalized_default is not None and prompt_yes_no(
        f"Would you like to use '{default_name}' as the project name?",
        default=True,
    ):
        if normalized_default != default_name:
            print(f"Using normalized project name '{normalized_default}'.")
        return normalized_default

    if normalized_default is None:
        print(
            f"The current folder name '{default_name}' is not a valid project name."
        )

    while True:
        raw_name = input("Enter a new project name: ").strip()
        try:
            project_name = normalize_project_name(raw_name)
        except argparse.ArgumentTypeError as exc:
            print(f"Invalid project name: {exc}")
            continue

        if project_name != raw_name:
            print(f"'{raw_name}' will be normalized to '{project_name}'.")
        if prompt_yes_no(f"Is '{project_name}' correct?", default=True):
            return project_name


def prompt_description(default: str) -> str:
    description = input(f"Project description [{default}]: ").strip()
    return description or default


def prompt_visibility(default: str) -> str:
    while True:
        visibility = input(f"GitHub visibility [{default}]: ").strip().lower()
        if not visibility:
            return default
        if visibility in {"private", "public"}:
            return visibility
        print("Please enter 'private' or 'public'.")


def collect_interactive_settings(root: Path) -> tuple[str, str, bool, str]:
    project_name = prompt_project_name(root.name)
    description = prompt_description(DEFAULT_DESCRIPTION)
    create_github = prompt_yes_no(
        f"Create GitHub repo 'git@github.com:{DEFAULT_OWNER}/{project_name}.git' and point origin there?",
        default=True,
    )
    visibility = DEFAULT_VISIBILITY
    if create_github:
        visibility = prompt_visibility(DEFAULT_VISIBILITY)
    return project_name, description, create_github, visibility


def toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_pyproject(project_name: str, description: str) -> str:
    return "\n".join(
        [
            "[build-system]",
            'requires = ["setuptools>=68"]',
            'build-backend = "setuptools.build_meta"',
            "",
            "[project]",
            f"name = {toml_string(project_name)}",
            'version = "0.1.0"',
            f"description = {toml_string(description)}",
            'readme = "README.md"',
            'requires-python = ">=3.10"',
            "dependencies = []",
            "",
            "[tool.setuptools]",
            'package-dir = {"" = "src"}',
            "",
            "[tool.setuptools.packages.find]",
            'where = ["src"]',
            "",
        ]
    )


def render_project_readme(project_name: str, description: str) -> str:
    return "\n".join(
        [
            f"# {project_name}",
            "",
            description,
            "",
            "## Setup",
            "",
            "```bash",
            "uv sync",
            "```",
            "",
            "## Test",
            "",
            "```bash",
            "uv run python -m unittest",
            "```",
            "",
        ]
    )


def render_devcontainer(project_name: str) -> str:
    payload = {
        "name": f"{project_name} (py3.14-trixie + uv)",
        "image": "ghcr.io/astral-sh/uv:python3.14-trixie",
        "customizations": {
            "vscode": {
                "extensions": [
                    "ms-python.python",
                    "ms-python.vscode-pylance",
                ]
            }
        },
        "postCreateCommand": "uv sync --locked || uv sync",
    }
    return json.dumps(payload, indent=2) + "\n"


def render_package_init(project_name: str) -> str:
    return "\n".join(
        [
            f'"""Top-level package for {project_name}."""',
            "",
            '__all__ = ["__version__"]',
            "",
            '__version__ = "0.1.0"',
            "",
        ]
    )


def render_package_test(package_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import importlib",
            "import unittest",
            "",
            "",
            "class PackageTests(unittest.TestCase):",
            "    def test_package_exposes_version(self) -> None:",
            f'        module = importlib.import_module("{package_name}")',
            '        self.assertEqual(module.__version__, "0.1.0")',
            "",
            "",
            'if __name__ == "__main__":',
            "    unittest.main()",
            "",
        ]
    )


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def rename_package_directory(root: Path, package_name: str) -> None:
    src_dir = root / "src"
    template_dir = src_dir / TEMPLATE_PACKAGE_NAME
    target_dir = src_dir / package_name
    if template_dir == target_dir:
        return
    if not template_dir.exists():
        if target_dir.exists():
            raise RuntimeError("This template has already been initialized.")
        raise RuntimeError(f"Template package directory not found: {template_dir}")
    if target_dir.exists():
        raise RuntimeError(f"Target package directory already exists: {target_dir}")
    template_dir.rename(target_dir)


def write_project_files(root: Path, project_name: str, package_name: str, description: str) -> None:
    write_text(root / "pyproject.toml", render_pyproject(project_name, description))
    write_text(root / "README.md", render_project_readme(project_name, description))
    write_text(root / ".devcontainer" / "devcontainer.json", render_devcontainer(project_name))
    write_text(root / "src" / package_name / "__init__.py", render_package_init(project_name))
    write_text(root / "tests" / "test_package.py", render_package_test(package_name))


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def run_command(command: list[str], cwd: Path) -> None:
    require_command(command[0])
    subprocess.run(command, cwd=cwd, check=True)


def configure_github_remote(root: Path, owner: str, project_name: str, visibility: str) -> str:
    require_command("gh")
    require_command("git")
    repo = f"{owner}/{project_name}"
    remote_url = f"git@github.com:{repo}.git"

    repo_exists = subprocess.run(
        ["gh", "repo", "view", repo],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not repo_exists:
        run_command(["gh", "repo", "create", repo, f"--{visibility}"], cwd=root)

    origin_exists = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if origin_exists:
        run_command(["git", "remote", "set-url", "origin", remote_url], cwd=root)
    else:
        run_command(["git", "remote", "add", "origin", remote_url], cwd=root)

    return remote_url


def remove_stale_metadata(root: Path) -> None:
    for path in root.glob("src/*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn this template into a new Python project."
    )
    parser.add_argument(
        "--name",
        type=normalize_project_name,
        help="Project name to use for the repo and package derivation.",
    )
    parser.add_argument(
        "--description",
        help="Short project description for README and pyproject.toml.",
    )
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help="GitHub owner to use when creating a new remote repository.",
    )
    github_group = parser.add_mutually_exclusive_group()
    github_group.add_argument(
        "--create-github",
        action="store_true",
        dest="create_github",
        help="Create a GitHub repository and point origin at it.",
    )
    github_group.add_argument(
        "--no-github",
        action="store_false",
        dest="create_github",
        help="Skip GitHub repository creation.",
    )
    parser.set_defaults(create_github=None)
    parser.add_argument(
        "--visibility",
        choices=("private", "public"),
        help="Visibility to use for the GitHub repository.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip the final uv sync step.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        args = parse_args(argv)
        if args.name is None:
            project_name, description, create_github, visibility = (
                collect_interactive_settings(root)
            )
        else:
            project_name = args.name
            description = args.description or DEFAULT_DESCRIPTION
            create_github = bool(args.create_github)
            visibility = args.visibility or DEFAULT_VISIBILITY

        package_name = package_name_from_project_name(project_name)
        rename_package_directory(root, package_name)
        write_project_files(root, project_name, package_name, description)
        remove_stale_metadata(root)

        remote_url = None
        if create_github:
            remote_url = configure_github_remote(
                root=root,
                owner=args.owner,
                project_name=project_name,
                visibility=visibility,
            )

        if not args.skip_sync:
            run_command(["uv", "sync"], cwd=root)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    print(f"Initialized {project_name} with package {package_name}.")
    if remote_url is not None:
        print(f"Configured origin to use {remote_url}.")
    else:
        print("GitHub remote creation skipped.")
    if args.skip_sync:
        print("Skipped uv sync.")
    else:
        print("Created a fresh local environment with uv sync.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
