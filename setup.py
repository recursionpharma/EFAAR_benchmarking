from setuptools import find_packages, setup

import roadie

setup(
    packages=find_packages(),
    version=roadie.infer_version("EFAAR-benchmarking"),
    include_package_data=True,
)
