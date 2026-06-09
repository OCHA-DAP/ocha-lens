# Development

## Environment

Development is currently done using Python 3.12. We recommend using a virtual
environment such as ``venv``:

    python3.12 -m venv venv
    source venv/bin/activate

In your virtual environment, please install all packages for
development by running:

    pip install -r requirements.txt
    pip install -e ".[dev]"

## Pre-Commit

Also be sure to install `pre-commit`, which is run every time
you make a git commit:

    pre-commit install

With pre-commit, all code is formatted according to
[ruff](https://github.com/astral-sh/ruff) guidelines.

To check if your changes pass pre-commit without committing, run:

    pre-commit run --all-files

## Dependencies

[pip-tools](https://github.com/jazzband/pip-tools) is used for
package management.  If you’ve introduced a new package to the
source code (i.e.anywhere in `src/`), please add it to the
`project.dependencies` section of
`pyproject.toml` with any known version constraints.

For adding packages for testing or development, add them to
the `test` or `dev` sections under `[project.optional-dependencies]`.

Any changes to the dependencies will be automatically reflected in
`requirements.txt` with `pre-commit`, but you can re-generate
the file without committing by executing:

    pre-commit run pip-compile --all-files

## Documentation

Documentation is built using Sphinx with MyST for Markdown support. Build the docs locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

View the built documentation by opening `docs/_build/html/index.html` in your browser.

You can also use `sphinx-autobuild` to automatically rebuild the docs on changes:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

## Releasing

The package version is derived from the git tag via `hatch-vcs` — there is
**no version to bump in `pyproject.toml`**. Creating a GitHub Release is the
release: the `publish.yaml` workflow fires on `release: published` and
publishes the build to PyPI. PyPI will not let you reuse a version, so the
number must be right the first time.

Checklist:

1. **Make sure `main` is green** — CI passing, all intended PRs merged.
2. **Update the docs for any new/changed public API.** The Sphinx API
   reference (`docs/api.md`) is manually curated — new functions only appear
   once they're listed with an `autofunction` directive. For a new
   datasource, also add a `docs/datasets/<name>.md` page and wire it into
   `docs/datasets/index.md`. Build locally and confirm it's clean:

   ```bash
   sphinx-build -b html docs docs/_build/html
   ```

3. **Pick the version** (semver; the tag *is* the version). On the `0.x`
   line: bump the minor for new features (e.g. a new datasource or public
   function), the patch for fixes only. Use a bare tag like `0.5.0` — no `v`
   prefix.
4. **Draft release notes** summarizing user-facing changes, grouped by area,
   referencing the merged PRs.
5. **Create the GitHub Release** targeting `main` with that tag. This triggers
   the PyPI publish.
6. **Verify the publish** — confirm the `publish.yaml` run succeeded and the
   new version is live on [PyPI](https://pypi.org/project/ocha-lens/).
