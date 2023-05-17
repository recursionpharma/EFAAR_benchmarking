import re

import {{cookiecutter.python_name}}.core as sut


def test_get_version():
    # Official regex: https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
    semver_pattern = (
        r"^(?P<major>0|[1-9]\d*)"
        r"\.(?P<minor>0|[1-9]\d*)"
        r"\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+"
        r"(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.fullmatch(semver_pattern, sut.get_version()) is not None
