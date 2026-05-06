from typing import Any
from pathlib import Path
import yaml

def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg
