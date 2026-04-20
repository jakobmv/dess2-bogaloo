# Repo Conventions

Use this repo style as the default unless there is a clear reason not to.

- Use `uv` for environment management, dependency installation, and running commands.
- Keep a small `Makefile` with practical shortcuts for the workflows we actually use.
- Put importable Python code in `src/<project_name>/`.
- Put runnable task entrypoints in `scripts/`.
- Keep README files short, command-oriented, and focused on the main workflow.
- Keep repositories lean. Avoid scaffolding, abstractions, and folders that do not help right now.
- Do not commit large datasets, caches, or generated artifacts. Keep them local and gitignored.
- Prefer simple Python, clear names, and obvious file layout over clever patterns.
- When adding functionality, preserve the existing structure instead of inventing a new one.
