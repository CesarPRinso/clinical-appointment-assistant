import json
from pathlib import Path

_SCHEMA_CACHE = {}


def _load_json(path: str | Path) -> dict:
    p = Path(path).resolve()
    if p in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[p]
    data = json.loads(p.read_text(encoding="utf-8"))
    _SCHEMA_CACHE[p] = data
    return data


def load_output_schema() -> dict:
    # app/  ->  ../schemas/llm_output.schema.json
    return _load_json(Path(__file__).parent.parent / "schemas" / "llm_output.schema.json")
