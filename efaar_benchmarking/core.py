from importlib.metadata import version


def get_version() -> str:
    """Returns a string representation of the version of efaar_benchmarking currently in use

    Returns
    -------
    str
        the version number installed of this package
    """
    try:
        return version("efaar_benchmarking")
    except ModuleNotFoundError:
        return "set_version_placeholder"
