def get_version() -> str:
    """Returns a string representation of the version of efaar_benchmarking currently in use

    Returns
    -------
    str
        the version number installed of this package
    """
    try:
        from importlib.metadata import version  # type: ignore

        return version("efaar_benchmarking")
    except ImportError:
        try:
            import pkg_resources

            return pkg_resources.get_distribution("efaar_benchmarking").version
        except pkg_resources.DistributionNotFound:
            return "set_version_placeholder"
    except ModuleNotFoundError:
        return "set_version_placeholder"
