from __future__ import annotations

import logging
import sys
import warnings


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)
logging.getLogger("sam3").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    from sam31_webapp.app import main as app_main

    return app_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
