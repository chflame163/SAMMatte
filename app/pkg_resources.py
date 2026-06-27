from __future__ import annotations

from importlib import import_module
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    """
    Minimal compatibility shim for projects that only need
    pkg_resources.resource_filename().

    SAM 3 uses this to locate packaged assets such as:
    sam3/assets/bpe_simple_vocab_16e6.txt.gz
    """

    module = import_module(package_or_requirement)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ModuleNotFoundError(
            f"Cannot resolve package path for {package_or_requirement!r}"
        )
    base_dir = Path(module_file).resolve().parent
    return str(base_dir / resource_name)
