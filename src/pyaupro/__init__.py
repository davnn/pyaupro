from ._implementation import PerRegionOverlap
from ._utilities import auc_compute, generate_random_data


def get_version() -> str:
    """Return the package version or "unknown" if no version can be found."""
    from importlib import metadata

    try:
        return metadata.version(__name__)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "no-version-found-in-package-metadata"


__version__ = get_version()
