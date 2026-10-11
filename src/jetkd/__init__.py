from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jetkd")
except PackageNotFoundError:
    __version__ = "unknown"
