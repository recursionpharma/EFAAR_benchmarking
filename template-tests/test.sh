#!/bin/bash

set -e
set -x

# Most developers will have their pip config set to look at nexus, so this
# conditional is to capture the case of the CI system where teh PYPI_DOWNLOAD_*
# env vars should be set. This should also work locally if you set these
# variables
if [[ $(pip config list) != *"nexus.rxrx.io"* ]]
then
    export PIP_INDEX_URL="https://${PYPI_DOWNLOAD_USERNAME}:${PYPI_DOWNLOAD_PASSWORD}@nexus.rxrx.io/repository/pypi-all/simple"
fi

python -m pip install -r requirements.txt

################################################################################
#### Test Case: Package With CLI
rm -rf test-package || true
python test_generate.py y

pushd test-package

roadie lock --all
roadie venv --python 3.9 --use-venv
source venv/bin/activate

flake8
black --diff --check .
mypy
bandit -r . -x /tests/,/venv/
validate-pyproject ./pyproject.toml 
CONFIGOME_ENV=test pytest
pip install --upgrade 'tox<4.0'  # https://github.com/GoogleCloudPlatform/artifact-registry-python-tools/pull/41/files
tox --parallel

popd

################################################################################
#### Test Case: Package Without a CLI
rm -rf test-package || true
python test_generate.py n

pushd test-package

roadie lock --all
roadie venv --python 3.9 --use-venv
source venv/bin/activate

flake8
black --diff --check .
mypy
bandit -r . -x /tests/,/venv/
validate-pyproject ./pyproject.toml 
CONFIGOME_ENV=test pytest
pip install --upgrade 'tox<4.0'  # https://github.com/GoogleCloudPlatform/artifact-registry-python-tools/pull/41/files
tox --parallel

popd
