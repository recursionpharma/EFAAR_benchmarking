:warning: **Note: Do not attempt to use this directly unless you
know what you are doing. The happy path is to let `roadie` interact with this
template for you.** :warning:

This is a template repository. It will be used by `roadie` to instantiate
instances. We will make an initial commit on those repos with this boilerplate,
so there is traceability as to the version of this template the repo was created
with. This could be useful in figuring out later upgrades necessary to keep
a repo up-to-date with compliance standards. Please use:

```bash
roadie quickstart package
```

# What's Inside?

## Project Configuration Files

<details>
<summary>

### `pyproject.toml`

</summary>
The newer standard for specifying python project configuration
that is meant to be a one-stop shop for specifying package metadata, third-party tools, 
and package requirements. Most formatting/linting/testing tools we use allow for configuration to be specified in this file including:
 - `bandit` - A static analyzer that checks the code for security risks
 - `black` - An opinionated, minimally configurable code formatter
 - `isort` - An import sorter that will alphabetize your import statements
 - `mypy` - A static analysis tool for type checking Python code
 - `pytest` - A testing framework
</details>

<details>
<summary>

### `setup.cfg`

</summary>

Required for configuring `flake8` in a format that is compatible with `black`.
We can remove this once https://github.com/PyCQA/flake8/issues/234 is addressed or
if we incur an extra dependency in [pyproject-flake8](https://pypi.org/project/pyproject-flake8/)

</details>

<details>
<summary>

### `setup.py`

</summary>

Invoked during CI in order to create a binary distribution and also locally when using `pip install .` (if you do this, we highly recommend using the `-e` flag to make the install editable, meaning it will essentially symlink to your dev environment).
If we switch to using [setuptools-scm](https://pypi.org/project/setuptools-scm/) for semantic versioning, this file can be omitted entirely.

</details>

## Provided Dependencies

<details>
<summary>click</summary>

[Click](https://click.palletsprojects.com/en/8.1.x/) is a nice library for constructing command line interfaces (CLIs)
without much pain. We recommend using it, if you intend on providing a command line entrypoint into your package.

</details>

## CI and Developer Tools

Upon creation of a new package, your project will be registered against a centralized CI build pipeline that is defined at [build-pipelines/python/library.yml](https://github.com/recursionpharma/build-pipelines/blob/trunk/python/library.yml).
The registering of the trigger back to your repository is beside the first file at [build-pipelines/python/library-spec.yml](https://github.com/recursionpharma/build-pipelines/blob/trunk/python/library-spec.yml).
This pipeline will run a bunch of checks against your code, and in this section we are going to elaborate on what each of those does, how it is configured, and any additional tooling that is at your disposal.

Our CI system uses a common `rxrx-python-builder` image for all pre-checks with all of the tools below baked into it.
The definition for this image is defined in [container-images/rxrx-python-builder](https://github.com/recursionpharma/container-images/tree/trunk/rxrx-python-builder)
where exact versions are pinned to ensure consistent builds. Note, the current version used by the CI system can be found by looking at [build-pipelines/python/library.yml](https://github.com/recursionpharma/build-pipelines/blob/trunk/python/library.yml).

### :fast_forward: Adopt :fast_forward:

<details>
<summary>black</summary>

[Black](https://black.readthedocs.io/en/stable/) is an opinonated, minimally configurable formatter that ensures that
with very little configuration all of our code adheres to a common Recursion standard. The only thing we currently
configure out of the box is the line length and that is derived from
[a constant](https://github.com/recursionpharma/roadie/blob/bb87651b769a9445484eb6e4ced5ff63029d307c/roadie/constants.py#L11)
in the `roadie` repository. `black` reads its configuration from the `pyproject.toml` file. Running `black` against your
code should automatically fix any errors, but if you want an all-in-one experience there is also
`roadie lint fix --black --isort` which will fix style and format anomalies.

</details>

<details>
<summary>codeql</summary>

[CodeQL](https://codeql.github.com/) is an advanced security offering from GitHub that will do a deep static analysis
of your codebase for common vulnerabilities and exposures. This is run as part of the CI system and will not fail the
main pipeline, but instead sends a report to GitHub which will trigger an additional "check" on your code. At this time,
we do not have an easy script for executing this on your local machine, but this is an improvement we expect to make in
the future.

</details>

<details>
<summary>coverage</summary>

[Coverage](https://coverage.readthedocs.io/en/7.2.1/) is a standalone tool that can be used to generate and combine
test coverage reports for your code. As noted above, we use `pytest-cov` to automatically generate coverage reports
for individual invocations of `pytest`, however with `tox` we usually end up running `pytest` in a variety of
configurations, and it would wrong for us to report any single run as the total coverage of our test suites. Therefore,
we use this tool in our default `tox` configuration to combine reports across all runs to generate a single coverage
report that we can then publish to the [build-reports](https://console.cloud.google.com/storage/browser/build-reports)
bucket.

</details>

<details>
<summary>isort</summary>

[isort](https://pycqa.github.io/isort/) sorts your import statements alphabetically. This is not a mandatory tool and
it will not be validated in the CI system, however it is a nice little convenience that can make wrangling your import
statements automatically consistent. There is a modest amount of configuration included in this template in the
`pyproject.toml` file to make `isort` play well with `black` and our default line length configuration defined in
[roadie/constants.py](https://github.com/recursionpharma/roadie/blob/bb87651b769a9445484eb6e4ced5ff63029d307c/roadie/constants.py#L15-L19).

</details>

<details>
<summary>mypy</summary>

[MyPy](https://mypy.readthedocs.io/en/stable/) is a type checking static analyzer that we run as part of our CI system.
This tool is configured in the `pyproject.toml`. As typing is optional in python, this tool supports gradual typing and
often depends on `types-` packages (can be handled automatically by including the `--install-types` flag
during invocation, e.g., `mypy --install-types --non-interactive`) or `-stubs` (e.g., `pandas-stubs`) being installed to
give a full report of your system. Without these extra packages installed, `mypy` may conservatively report success.

</details>

<details>
<summary>pytest</summary>

[PyTest](https://docs.pytest.org/en/7.2.x/) is a python testing framework we have chosen to standardize on. It utilizes
a plugin architecture that can extend the tool to suit your specific project's needs. By default, our project's ship
with the `pytest-cov` plugin which allows us to report on test coverage and publish results to a
[build-reports](https://console.cloud.google.com/storage/browser/build-reports) bucket during our CI builds.

</details>

<details>
<summary>tox</summary>

[Tox](https://tox.wiki/en/latest/) is a virtual environment managment and test tool that works well with `pytest` and
other tools by allowing you to define different environments under which to exercise your code. For example, we can use
`tox` to exercise our code under different minor versions of Python which is the default usage of it in this template,
but you can also use it to run pytest in environments both with and without optional dependencies of a project or
to setup configurations of pytest with different flags or markers enabled. We have elected to keep the configuration
for `tox` in the `tox.ini`, although it does have [support](https://tox.wiki/en/latest/config.html#pyproject-toml) for
consolidating this into a `pyproject.toml`.

</details>

### :arrow_forward: Trial :arrow_forward:

<details>
<summary>docstr-coverage</summary>

[Docstr-Coverage](https://docstr-coverage.readthedocs.io/en/latest/api_essentials.html) is a tool for reporting on the
percentage of public functions containing docstrings. This tool does not validate docstrings to a style guide, though
we have agreed as a company to use `numpy`-style docstrings. This latter functionality is deferred to
`pydocstyle` which is not currently a part of this template. If you use VS Code, we recommend using the
[autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) extension to easily
construct template docstrings that you can fill in with correct details.

</details>

<details>
<summary>radon</summary>

[Radon](https://radon.readthedocs.io/en/latest/) is a static analyzer that will report on your project's overall
complexity and maintainability. The usage of this tool is entirely optional, though it will run as part of the CI
system, and results will be published to a [build-reports](https://console.cloud.google.com/storage/browser/build-reports)
bucket if you ever wanted to look at them. We do not maintain any custom configuration of this tool, though you can
reference their [relevant documentation](https://radon.readthedocs.io/en/latest/commandline.html#radon-configuration-files)
on how to set it up.

</details>

<details>
<summary>pre-commit</summary>

[Pre-commit](https://pre-commit.com/) hooks are an optional feature that developers can choose to use on their own
machines that will run a collection of validation scripts against your code before allowing you to commit. Some of these
tools will modify your code in-place, but will fail on commit. This tool is configured in the `.pre-commit-config.yaml`
file of this template and it can be installed on local development environments with `pre-commit install`. Once
installed, you can bypass these checks if your house is burning down, but you really need to commit something with
`git commit --no-verify`.

</details>

<details>
<summary>pyupgrade</summary>

[Pyupgrade](https://github.com/asottile/pyupgrade) is one of the tools configured in the `.pre-commit-config.yaml` file
of this template that will run with `pre-commit`. Pyupgrade will automatically upgrade syntax to match a specified
version of python. It can also be invoked from the command line as something like `pyupgrade --py38-plus *.py`.

</details>

<details>
<summary>validate-pyproject</summary>

[Validate-pyproject](https://validate-pyproject.readthedocs.io/en/latest/) is one of the tools configured in the
`.pre-commit-config.yaml` file of this template that will run with `pre-commit`. This tool will parse and analyze your
`pyproject.toml` file for malformed configuration. It can also be invoked from the command line as something like
`validate-pyproject ./pyproject.toml`.

</details>

### :arrow_backward: Assess :arrow_backward:

<details>
<summary>bandit</summary>

[Bandit](https://bandit.readthedocs.io/en/latest/) is a legacy security static analyzer that we used to leverage on our
CI system that has been replaced by GitHub's CodeQL (see below). It remains a package we include in the optional dev
requirements as it can provide a quick response of possible security concerns when invoked from the CLI
(e.g., `bandit -r -c pyproject.toml .`). This tool is configured in `pyproject.toml`.

</details>

<details>

<summary>flake8</summary>

[Flake8](https://flake8.pycqa.org/en/latest/) is a PEP8-compliant style linter that will validate your code's adherence
to the [PEP 8](https://peps.python.org/pep-0008/) standard. Aside from ensuring consistent and Pythonic usage of the
language, this linter can catch common semantic errors and remove a lot of excess overhead in hastily written code.
To automatically fix stylistic problems in your code, run `roadie lint fix`. `flake8` is configured in the `setup.cfg`
file for now, and the template for its configuration on all new projects is housed in
[roadie/constants.py](https://github.com/recursionpharma/roadie/blob/bb87651b769a9445484eb6e4ced5ff63029d307c/roadie/constants.py#L15-L19).
There is again a very small amount of configuration needed in order to make this work kindly with `black` for an
up-to-date description, please refer to `black`'s [documentation](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html#flake8).
It may be worth looking at `pylint` or `ruff` as a replacement for this tool.

</details>

### :no_entry: Hold :no_entry:

<details>
<summary>dependabot</summary>

[Dependabot](https://docs.github.com/en/code-security/dependabot) is another security tool acquired by GitHub that will
examine your project's dependency list and flag any vulnerable dependencies. We are working on patterns that allow
`dependabot` to work nicely with our project structure in terms of automating pull requests, but the analysis that this
tool flag, should be heeded and remediated at your earliest convenience. We have our own system of monthly maintenance
that does this in a way that is guaranteed to not violate our lockfiles dependency graph.

</details>
