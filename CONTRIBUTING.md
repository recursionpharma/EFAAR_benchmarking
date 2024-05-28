# Contributing to EFAAR_benchmarking

Thank you for your interest in contributing to our project! From commenting on and triaging issues, to reviewing and sending Pull Requests, all contributions are welcome.
We aim to build a welcoming and friendly community for anyone who wants to use or contribute to methods for building and benchmarking perturbative maps of biology.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating in this project you agree to abide by its terms.

## Reporting bugs and feature requests

We use GitHub Issues to track bugs and features. Report them there.

## Development

### Prerequisites

* Python 3.10 or later

* Pip for installing Python dependencies

* A virtual environment tool such as `pyenv` or `conda`

### Installing

1. Clone the project to your machine.

2. Navigate to the project directory.

3. Setup the virtual environment and activate it.

4. Install the dependencies via `pip install -e .`.

### Making Changes

* Before making any changes, ensure you have the latest version of the project by pulling the changes from trunk.

* Create a new branch for your changes.

* While developing, make sure to to add a docstring to all new functions, that clearly yet briefly explains what the function does, what the inputs and outputs are, and what exceptions the function can raise.

* Update the `pyproject.toml` file with any new requirements. This ensures that others setting up the project environment will have all necessary dependencies installed.

* Make sure your changes are atomic (i.e., each commit should contain a single logical change), and the commit message clearly describes this change. This makes the commit history much easier to understand.

* If you use a new benchmark source to evaluate maps, make sure to add it to `benchmark_annotations` folder with license information added to the LICENSE file [here](https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/efaar_benchmarking/benchmark_annotations/LICENSE).

* New map building pipeline functions go into the `efaar.py` module. New map benchmarking functions go into the `benchmarking.py` module. Data loading functions for new perturbation datasets go into the `data_loading.py` module.

* If you are adding new functions to build and benchmark new maps, make sure to update the existing notebooks or add a new notebook to enable others to reproduce your results.

## Submitting Changes

* Make sure to run `pre-commit run --all-files` to find and fix simple issues before submission.

* Stage and commit your changes: `git add <modified files>` and `git commit -m "your message"`. Your commit message should include a brief description of what changes were made.

* Push your branch: `git push origin <branch-name>`.

* In the Github repository, create a new Pull Request.

* Describe your changes in the Pull Request and link any related issues.

* Wait for the review from maintainers and make any necessary changes after the review.

* After approval, your changes will be merged. Congratulations!

## License

By contributing, you agree that your contributions will be licensed under the license terms in [LICENSE](https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/LICENSE).
