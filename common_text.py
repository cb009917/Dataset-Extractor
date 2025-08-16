# common_text.py
import re
import json
import os
from pathlib import Path

# Set UTF-8 encoding at process start
os.environ["PYTHONIOENCODING"] = "utf-8"

STOP = {"fresh","large","small","medium","finely","coarsely","ground","crushed","dried",
        "red","green","yellow","ripe","chopped","sliced","minced","seed","seeds","leaf","leaves",
        "raw","whole","to","taste","optional","garnish"}

SL_NORMALIZE = {
    "brinjal": "eggplant", "ladies finger": "okra",
    "big onion": "onion", "red onion": "onion",
    "long beans": "green beans", "leeks": "leek",
    "green gram": "mung beans", "mung dal": "mung beans",
    "dhal": "red lentils", "dal": "red lentils", "masoor dal": "red lentils",
    "maldive fish": "dried fish",
    "sprat": "dried sprats", "sprats": "dried sprats",
    "gotukola": "pennywort", "mukunuwenna": "amaranth leaves",
    "coconut milk powder": "coconut milk", "coconut cream": "coconut milk",
    "all purpose flour": "wheat flour", "plain flour": "wheat flour", "atta flour": "wheat flour",
    "soya meat": "textured soy protein", "soya chunks": "textured soy protein",
    "soy protein": "textured soy protein",
    "black coffee": "coffee", "coffee": "coffee",
    "goat meat": "mutton", "plantain": "banana", "tamarind seedless": "tamarind",
    "plantain": "banana", "mutton": "mutton"
}

def canonicalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    for k, v in SL_NORMALIZE.items():
        s = re.sub(rf"\b{k}\b", v, s)
    toks = [t for t in s.split() if t and t not in STOP]
    return " ".join(toks)

# alias utils
ALIAS_PATH = Path("config") / "item_alias_user.json"

def load_alias():
    ALIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ALIAS_PATH.exists():
        with open(ALIAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_alias(alias: dict):
    tmp = ALIAS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(alias, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ALIAS_PATH)