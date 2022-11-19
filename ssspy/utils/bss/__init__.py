import warnings

__all__ = ["warning_ip2"]


def warning_ip2(spatial_algorithm: str) -> None:
    if spatial_algorithm in ["IP2"]:
        warnings.warn(
            (
                "The current implementation of IP2 is based on "
                '"Auxiliary-function-based independent component analysis '
                'for super-Gaussian sources", ',
                "but this is not what is actually known as IP2.",
                "See https://github.com/tky823/ssspy/issues/178 for more details.",
            ),
            UserWarning,
        )
