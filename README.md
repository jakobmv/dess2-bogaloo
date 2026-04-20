# template-repo

Lean Python starter repo with your default setup and conventions already in place.

This template keeps the parts you want to reuse:

- `uv` for environment management and commands
- a simple devcontainer based on `ghcr.io/astral-sh/uv:python3.14-trixie`
- `src/` package layout
- the repo conventions in `AGENTS.md`
- a one-command project initializer

## Use This Template

Clone the template, then run:

```bash
git clone git@github.com:jakobmv/template-repo.git my-new-project
cd my-new-project
make init
```

The setup will prompt you for:

- whether to use the current top-folder name as the project name
- a short project description
- whether to create `git@github.com:jakobmv/<project-name>.git` and point `origin` there
- GitHub visibility when you choose to create the repo

If you clone into the final folder name up front, you can usually just answer `y` to the first prompt.

What `make init` does:

- renames `src/template_repo` to match your new package name
- rewrites `pyproject.toml`, `README.md`, and `.devcontainer/devcontainer.json`
- keeps `AGENTS.md` and the rest of the repo structure intact
- runs `uv sync` to create a fresh local environment
- optionally creates `git@github.com:jakobmv/<project-name>.git` and points `origin` there

Defaults:

- GitHub owner: `jakobmv`
- GitHub visibility: `private`

You can still run the script directly for non-interactive setup:

```bash
uv run python scripts/init_project.py --name my-new-project --description "Short project description" --create-github
```

## Development

```bash
uv sync
uv run python -m unittest
```
