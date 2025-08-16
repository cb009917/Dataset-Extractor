#!/usr/bin/env python3
r"""
FitFeast – Price Calculator (robust, strict denominator, auto-alias learn, debug-friendly)

Usage:
  python price_calculator.py <input_csv> [output_csv] [--price-data PATH] [--auto-alias]

Notes:
  - Set PRICE_DATA_PATH env or pass --price-data to point at config/extracted_prices2.csv
  - Use --auto-alias to enable a second pass that learns + applies high-confidence aliases
"""

import os, re, json, argparse, csv
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Set UTF-8 encoding at process start
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
import numpy as np
from common_text import load_alias, save_alias, canonicalize_name

# ----------------- Logging -----------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("price_calculator")

# ----------------- Data classes -----------------
@dataclass
class PriceCoverage:
    baseline: float
    strict: float
    very_strict: float

# ----------------- Canonicals & Maps -----------------
UNIT_CANON = r"(?:kg|g|l|ml|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)"

# Category fallback keywords
CATEGORY_KEYWORDS = {
    "vegetable": ["brinjal","eggplant","okra","beans","leek","cabbage","pumpkin","beetroot","carrot","tomato","capsicum","bell pepper","gourd","drumstick","banana flower","ash plantain","green beans","long beans","gotukola","mukunuwenna","kangkung","pennywort","amaranth","cassava","celery","onion","gotu kola","lotus root","entils","lentil"],
    "fruit": ["banana","lime","lemon","mango","pineapple","papaya","wood apple"],
    "staple": ["rice","rice flour","wheat flour","noodles","pasta","vermicelli","semolina","lentils"],
    "protein": ["egg","chicken","beef","pork","fish","prawns","sprats","dried fish","tofu","soya meat","chickpeas","red lentils","mung beans","cowpea","urad dal","meatball"],
    "dairy": ["milk","curd","yogurt","cheese","milk powder"],
    "fat": ["oil","coconut oil","ghee","butter","margarine","coconut"],
    "spice": ["fennel","cashew","sugar","corn starch","lemongrass","lemon grass","emongrass"]
}

UNIT_ALIAS_MAP = {
    # weights/volumes
    "gram":"g","grams":"g","kilogram":"kg","kilograms":"kg",
    "milliliter":"ml","milliliters":"ml","liter":"l","liters":"l","litre":"l","litres":"l",
    # spoons/cups
    "teaspoon":"tsp","teaspoons":"tsp","tablespoon":"tbsp","tablespoons":"tbsp","cups":"cup",
    # imperial
    "ounce":"oz","ounces":"oz","pound":"lb","pounds":"lb","lbs":"lb",
    # pieces/packs
    "piece":"pcs","pieces":"pcs","pc":"pcs","packet":"pkt","pack":"pkt","pkts":"pkt","bunches":"bunch",
    "cloves":"clove","cans":"can","tins":"tin",
}

# Liquids density (ml -> g)
DENSITIES = {
    "water": 1.00, "oil": 0.92, "vegetable oil": 0.92, "coconut oil": 0.92,
    "milk": 1.03, "coconut milk": 1.03, "yogurt": 1.03, "curd": 1.03,
    "honey": 1.42, "soy sauce": 1.20, "vinegar": 1.01, "tomato puree": 1.05,
    "treacle": 1.35, "molasses": 1.35, "ketchup": 1.10, "tomato sauce": 1.10
}

# Piece weights (pcs -> g)
PIECE_WEIGHTS = {
    "egg": 50, "onion": 150, "tomato": 120, "garlic clove": 3, "garlic": 3,
    "lemon": 70, "lime": 30, "potato": 170, "carrot": 70, "green chili": 6, "green chilli": 6,
    "green chillies": 6
}

# Ingredient aliases → DCS names (seed set; auto-learn will extend at runtime)
ITEM_ALIAS = {
    # legumes / grains / pulses
    "dhal": "red lentils", "dal": "red lentils", "masoor dal": "red lentils",
    "toor dal": "split pigeon peas", "moong dal": "mung beans", "green gram": "mung beans",
    "cowpea": "cow pea", "black eyed pea": "cow pea", "channa": "chickpeas", "chickpea": "chickpeas",
    # rice / flours / noodles
    "raw rice": "rice", "nadu rice": "rice", "samba rice": "rice", "kekulu rice": "rice",
    "all purpose flour": "wheat flour", "plain flour": "wheat flour", "atta flour": "wheat flour",
    "string hopper flour": "rice flour", "idiyappam flour": "rice flour", "roasted rice flour": "rice flour",
    "rice noodles": "noodles", "noodles": "noodles",
    # coconut & dairy
    "fresh coconut": "coconut", "grated coconut": "coconut", "desiccated coconut": "coconut",
    "coconut milk powder": "coconut milk", "coconut cream": "coconut milk", "curd": "yogurt", "yoghurt": "yogurt",
    # oils & condiments
    "vegetable oil": "oil", "cooking oil": "oil", "coconut oil": "coconut oil", "soy sauce": "soy sauce",
    "tomato paste": "tomato puree", "tomato sauce": "tomato puree",
    "thin tamarind juice": "tamarind", "thick tamarind juice": "tamarind", "tamarind juice": "tamarind",
    "vinegar": "vinegar",
    # onions & alliums
    "onion": "big onion", "small onion": "big onion", "shallots onion": "big onion", "shallots/ onion": "big onion",
    "spring onion": "leek", "spring onion flower": "spring onion",
    "garlic cloves": "garlic", "ginger garlic paste": "garlic",
    # common produce
    "brinjal": "eggplant", "ladies finger": "okra", "okra": "okra",
    "long beans": "green beans", "green bean": "green beans",
    "banana pepper": "capsicum", "capsicum": "capsicum",
    "beet root": "beetroot", "ash plantain": "plantain", "banana blossom": "banana flower", "banana flower": "banana flower",
    "snake gourd": "snake gourd", "bitter gourd": "bitter gourd", "ridge gourd": "ridge gourd", "cucumber": "cucumber",
    "leeks": "leek", "pumpkin": "pumpkin", "cabbage leaves": "cabbage",
    # leafy greens (LK)
    "gotukola": "pennywort", "mukunuwenna": "amaranth leaves", "kankun": "water spinach",
    # meats / seafood
    "chicken thighs": "chicken", "chicken breast": "chicken", "beef steak": "beef",
    "mutton": "goat meat", "eggs": "egg", "prawn/shrimp": "prawns", "shrimp": "prawns", "cuttlefish": "squid",
    # dry fish
    "sprat": "dried sprats", "sprats": "dried sprats", "maldive fish": "dried fish",
    # chiles (B-list for denom; kept for quantified pricing if ever used)
    "red chili": "dried chillies", "red chilli": "dried chillies", "chili": "dried chillies", "chilli": "dried chillies",
    "green chili": "green chillies", "green chilli": "green chillies",
    # pantry / misc
    "breadcrumbs": "breadcrumbs", "vanilla extract": "vanilla", "lime juice": "lime",
    "roasted cashew nuts": "cashew nuts", "cashew nuts": "cashew nuts",
    # additional Sri Lankan terms
    "egg": "eggs", "cardomom": "cardamom", "cabbage": "cabbage",
    "instant dry yeast": "yeast", "rapid rise yeast": "yeast", "fast acting yeast": "yeast",
    "noodles": "rice noodles", "noodle": "rice noodles", "egg noodles": "rice noodles", "ramen noodles": "rice noodles",
    "crab": "crab", "squid": "cuttlefish", "baby squid": "squid", "calamari": "squid",
    "cardamom pods": "cardamom", "habanero": "green chillies", "kochchi": "green chillies",
    "boneless goat meat": "goat meat", "goat meat boti": "goat meat", "goat meat blood": "goat meat",
    "beans": "green beans", "bonchi": "green beans", "long beans": "green beans",
    "tapioca": "cassava", "cassava": "cassava", "manioc": "cassava",
    "semolina": "wheat flour", "rava": "wheat flour", "sooji": "wheat flour", "toasted semolina": "wheat flour",
    "soya meat": "soy protein", "soy meat": "soy protein",
    "pennywort": "gotukola", "centella asiatica": "gotukola", "gotu kola": "gotukola", "vallarai": "gotukola",
    "radish": "radish", "banana flower": "banana blossom", "nutella": "chocolate spread",
    "parotta": "roti", "paratha roti": "roti", "ceylon parata": "roti",
    "emon": "lemon", "cornstarch": "corn starch", "corn starch": "corn starch",
    "sprats": "dried sprats", "sago": "sago", "banana": "banana", "wood apple": "wood apple",
    "papaya": "papaya", "mango": "mango", "unripened mango": "raw mango",
    "goraka": "garcinia", "garcinia cambogia": "garcinia", "garcinia": "tamarind", "garcinia cambodia": "garcinia",
    "salmon": "fish", "mackerel": "fish", "fennel seeds": "fennel",
    "ridge gourd": "ridge gourd", "watakolu": "ridge gourd", "vetakolu": "ridge gourd",
    "cashews": "cashew nuts", "gotu kola leaves": "gotukola", "celery": "celery", "celery stalk": "celery",
    "red lentils": "red lentils", "split lentils": "red lentils", "ketchup": "tomato sauce",
    "broccoli": "broccoli", "broccoli crown": "broccoli", "lemongrass": "lemon grass", "lemon grass": "lemon grass",
    "meatballs": "beef", "lotus root": "lotus root", "mozzarella": "cheese", "mozzarella cubes": "cheese",
    "agathi keerai": "agathi leaves", "ponnanganni keerai": "ponnanganni leaves", "alternanthera sessilis": "ponnanganni leaves",
    "spring roll wrapper": "rice paper", "leek": "leeks",
    # Direct fixes for stubborn unit parsing issues
    "tspvinegar": "vinegar", "tspvanilla extract": "vanilla", "tspfennel seeds": "fennel seed",
    "tspgarlic": "garlic", "tspginger": "ginger", "tspcorn starch": "corn flour",
    "tbspketchup": "tomato puree", "tbspcoconut": "coconut", "tbspred lentils": "red lentils",
    "tbspvinegar": "vinegar", "tbspfinely garlic": "garlic", "tbspchopped garlic": "garlic",
    "ginstant dry yeast": "wheat flour", "grapid rise yeast": "wheat flour", "gyeast": "wheat flour",
    "gcashews": "cashews", "gmeatballs": "beef", "glotus root": "lotus root",
    "/2tspsugar": "sugar", "/2tbsp": "", "/2cup": "", "/4cup": "", "/4cuponion": "big onion",
    "/2cupsplit lentils": "red lentils", "/2cupcarrot": "carrot", "/2cupleek": "leeks",
    "cupred lentils": "red lentils", "lbmeatballs": "beef", "lbfrozen": "",
    # More specific remaining fixes
    "mukunuwenna": "mukunuwenna", "cassava": "cassava", "papaya": "papaya", "mango": "mango",
    "wood apple": "wood apple", "gotu kola": "gotukola", "garcinia cambogia": "tamarind",
    "garcinia": "tamarind", "goraka": "tamarind", "celery stalk": "celery",
    "tbspred lentils": "red lentils", "ginstant dry": "wheat flour", "grapid rise": "wheat flour",
    "ggotu kola": "gotukola", "lemongrass": "lemon grass", "cabbage": "cabbage",
    "inch each": "", "and head": "", "inch pcs": "", "pcs": "", "thumb size": "",
    "ggarlic": "garlic", "gginger": "ginger", "inch": "", "baby prawns": "prawns",
    "baby squid": "prawns", "tspfennel seeds": "fennel seed", "tspcorn starch": "corn flour",
    "tbspcoconut": "coconut",
    # Slash-separated and complex patterns
    "cassava/cassava": "cassava", "cassava/cassava/cassava": "cassava",
    "mukunuwenna/mukunuwenna": "mukunuwenna", "gotukola/gotukola/gotu kola/gotukola": "gotukola",
    "prawns/squid": "prawns", "/2tspsugar": "sugar", "of garcinia cambogia": "tamarind",
    "of goraka": "tamarind", "of garcinia": "tamarind", "gcashews cut into": "cashews",
    "ggotu kola leaves": "gotukola", "tbspred lentils": "red lentils", "lemongrass a little bit": "lemon grass",
    "/2cupsplit lentils": "red lentils", "/4cuponion": "big onion", "/2cupcarrot": "carrot",
    "/2cupleek part": "leeks", "inch of lemongrass": "lemon grass", "cupred lentils": "red lentils",
    # More targeted remaining fixes for specific patterns
    "cassava": "cassava", "arcinia cambogia oraka": "tamarind", "arcinia cambogia": "tamarind",
    "oraka arcinia cambogia": "tamarind", "oraka arcinia": "tamarind", 
    "arcinia cambogia oraka": "tamarind", "agathi": "mukunuwenna", 
    "ponnanganni": "mukunuwenna", "tspfennel seeds": "fennel seed",
    "gcashews": "cashews", "ggotu kola": "gotukola", "tbspred lentils": "red lentils",
    "tspcorn starch": "corn flour", "tbspcoconut": "coconut", "lemongrass little": "lemon grass",
    "gmeatballs": "beef", "/2cupsplit lentils": "red lentils", "/4cuponion": "big onion",
    "cupcurry": "curry powder", "/2cupcarrot": "carrot", "lbmeatballs": "beef",
    "/2cupleek": "leeks", "glotus root": "lotus root", "inch lemongrass": "lemon grass",
    "cupred lentils": "red lentils", "curry": "curry powder", 
    "/2tspsugar": "sugar", "gotukola/gotukola/gotu kola/gotukola": "gotukola",
    "ponnanganni/ponnanganni": "mukunuwenna", "cassava/cassava": "cassava"
}

# Two-pass deglue
_PAT_NUM_TO_UNIT = rf'(?<![A-Za-z])(\d+(?:[.,]\d+)?(?:\s+\d/\d)?|\d/\d)\s*({UNIT_CANON})\b'
_PAT_UNIT_TO_WORD = rf'\b({UNIT_CANON})(?=[A-Za-z])'

_STOPWORDS = {"fresh","large","small","medium","finely","coarsely","ground","crushed",
              "dried","red","green","yellow","ripe","chopped","sliced","minced",
              "seed","seeds","leaf","leaves"}

def _tokenize_name(s: str) -> set:
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return set(toks)

def _singularize(word: str) -> str:
    if word.endswith("ies"): return word[:-3] + "y"
    if word.endswith("ves"): return word[:-3] + "f"
    if word.endswith("oes"): return word[:-2]
    if word.endswith("s") and not word.endswith("ss"): return word[:-1]
    return word

def _normalize_name_for_lookup(name: str) -> str:
    # Use shared canonicalization function for consistency
    return canonicalize_name(name)

def _category_from_name(name: str):
    nl = (name or "").lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in nl for k in kws): return cat
    return None

# Split-letter/typo fixes (before & after deglue) - order matters!
SPLIT_FIXES = [
    # Specific problematic patterns first
    (r"\binch pcs of g inger\b", "inch pcs ginger"),
    (r"\bpcs of g inger\b", "pcs ginger"), 
    (r"\binch g inger\b", "inch ginger"),
    (r"\bof g inger\b", "ginger"),
    (r"\bof g arlic\b", "garlic"),
    (r"\bg inger\b", "ginger"),
    (r"\bg arlic\b", "garlic"),
    # Unit degluing fixes for common patterns - specific ones first
    (r"\btspvinegar\b", "tsp vinegar"),
    (r"\btspvanilla extract\b", "tsp vanilla extract"),
    (r"\btspfennel seeds\b", "tsp fennel seeds"),
    (r"\btspgarlic\b", "tsp garlic"),
    (r"\btspginger\b", "tsp ginger"),
    (r"\btspcorn starch\b", "tsp corn starch"),
    (r"\btbspketchup\b", "tbsp ketchup"),
    (r"\btbspcoconut\b", "tbsp coconut"),
    (r"\btbspred\b", "tbsp red"),
    (r"\btbspfinely\b", "tbsp finely"),
    (r"\btbspchopped\b", "tbsp chopped"),
    (r"\btbspvinegar\b", "tbsp vinegar"),
    (r"\btsp(vinegar|vanilla|fennel|corn|garlic|ginger)", r"tsp \1"),  # tspvinegar -> tsp vinegar  
    (r"\btbsp(ketchup|coconut|red|chopped|finely|vinegar|garlic|ginger)", r"tbsp \1"), # tbspketchup -> tbsp ketchup
    (r"\bcup(red|onion|carrot|leek)", r"cup \1"),        # cupred -> cup red
    (r"\bginstant dry yeast\b", "g instant dry yeast"),
    (r"\bgrapid rise yeast\b", "g rapid rise yeast"),
    (r"\bgcashews\b", "g cashews"),
    (r"\bggotu kola\b", "g gotu kola"),
    (r"\bgmeatballs\b", "g meatballs"),
    (r"\bglotus root\b", "g lotus root"),
    (r"\bgyeast\b", "g yeast"),
    (r"\bg(instant|rapid|cashews|gotu|meatballs|lotus)", r"g \1"), # ginstant -> g instant
    (r"\blb(frozen|meatballs)", r"lb \1"),               # lbfrozen -> lb frozen
    (r"\b/2tspsugar\b", "/2 tsp sugar"),
    (r"\b/2tbsp\b", "/2 tbsp"),
    (r"\b/2cup\b", "/2 cup"),
    (r"\b/4cup\b", "/4 cup"),
    (r"\b(\d+/)(\d+)(tsp|tbsp|cup)", r"\1\2 \3"),       # /2tspsugar -> /2 tsp sugar
    (r"\b/(\d+)(tsp|tbsp|cup)", r"/\1 \2"),             # /2tsp -> /2 tsp
    # General letter-by-letter splitting fixes
    (r"\bc\s*urry\s*l\s*eaves\b", "curry leaves"),
    (r"\bc\s*urry\b",             "curry"),
    (r"\bl\s*eaves\b",            "leaves"),
    (r"\bl\s*eaf\b",              "leaf"),
    (r"\bl\s*ime\b",              "lime"),
    (r"\bl\s*eeks\b",             "leeks"),
    (r"\bl\s*ong\b",              "long"),
    (r"\bg\s*rated\b",            "grated"),
    (r"\bg\s*reen\b",             "green"),
    (r"\bg\s*arlic\b",            "garlic"),
    (r"\bg\s*inger\b",            "ginger"),
    (r"\bcardomom\b",             "cardamom"),
    # Additional specific pattern fixes
    (r"\bggotu kola leaves with stalks\b", "gotukola"),
    (r"\btbspred l entils\b",     "tbsp red lentils"),
    (r"\btspcorn starch\b",       "tsp corn starch"),
    (r"\btbspcoconut\b",          "tbsp coconut"),
    (r"\b/2cupsplit l entils\b",  "/2 cup red lentils"),
    (r"\b/4cuponion\b",           "/4 cup onion"),
    (r"\bglotus root\b",          "g lotus root"),
    (r"\binch of l emongrass thinly or\b", "lemongrass"),
    (r"\bcupred l entils\b",      "cup red lentils")
]
def _apply_split_fixes(s: str) -> str:
    t = s
    for pat, repl in SPLIT_FIXES:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()

# ----------------- Core class -----------------
class PriceCalculator:
    def __init__(self, price_data_path: str = "config/extracted_prices2.csv", auto_alias: bool = False):
        self.price_data_path = price_data_path
        self.auto_alias = auto_alias
        self.price_data: Dict[str, float] = {}  # normalized name -> LKR/kg
        self.shared_aliases = load_alias()  # Load shared aliases
        if self.shared_aliases:
            log.info(f"Loaded {len(self.shared_aliases)} user aliases from config/item_alias_user.json")
        self.unmatched_reports = []  # For comprehensive unmatched reporting
        self.load_price_data()

    # -------- Helpers: text normalization --------
    @staticmethod
    def _strip_parens(s: str) -> str:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\([^)]*\)", "", s)
        return s

    def _alias_units_everywhere(self, s: str) -> str:
        return re.sub(
            r"(?:kg|g|l|ml|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch|"
            r"grams?|kilograms?|milliliters?|litres?|liters?|teaspoons?|tablespoons?|cups|"
            r"ounces?|pounds?|lbs|piece|pieces|pc|packet|pack|pkts?|bunches|cloves|cans|tins)",
            lambda m: UNIT_ALIAS_MAP.get(m.group(0).lower(), m.group(0).lower()),
            s, flags=re.IGNORECASE
        )

    def normalize_token(self, tok: str) -> str:
        if not isinstance(tok, str):
            return ""
        s = self._strip_parens(tok.strip())
        s = self._alias_units_everywhere(s)
        s = _apply_split_fixes(s)
        s = re.sub(_PAT_UNIT_TO_WORD, r"\1 ", s, flags=re.IGNORECASE)
        s = re.sub(_PAT_NUM_TO_UNIT, r"\1 \2", s, flags=re.IGNORECASE)
        s = _apply_split_fixes(s)
        s = re.sub(r"\s+", " ", s).strip()
        # ingredient aliases after cleanup
        for k, v in ITEM_ALIAS.items():
            s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
        # apply shared aliases (override built-in ones)
        s = canonicalize_name(s)
        # Apply user aliases
        for alias, target in self.shared_aliases.items():
            if alias in s:
                s = s.replace(alias, target)
        return s

    # -------- Price CSV → per-kg index --------
    def load_price_data(self):
        price_path = self.price_data_path
        if not os.path.exists(price_path):
            raise FileNotFoundError(
                f"Price data not found at '{price_path}'. "
                f"Place 'extracted_prices2.csv' under config/ or pass --price-data."
            )

        log.info(f"Loading price data from {price_path}")
        df = pd.read_csv(price_path, encoding="utf-8", dtype=str).fillna("")
        cols = {c.strip().lower(): c for c in df.columns}

        item_col  = cols.get("name") or cols.get("item") or cols.get("commodity") or cols.get("description")
        unit_col  = cols.get("unit") or cols.get("pack") or cols.get("size") or cols.get("package")
        price_col = cols.get("average price") or cols.get("avg price") or cols.get("price") or cols.get("amount") or cols.get("rs") or cols.get("lkr")

        if not item_col or not unit_col or not price_col:
            raise ValueError(
                f"Price CSV must have item/pack/price columns. Found: {list(df.columns)}. "
                f"Expect something like: Name/Unit/Average Price."
            )

        normalized = 0
        perkg: Dict[str, float] = {}
        for _, r in df.iterrows():
            name = str(r[item_col]).strip()
            unit = str(r[unit_col]).strip()
            price_txt = str(r[price_col]).strip()
            if not name or not unit or not price_txt:
                continue
            try:
                price = float(re.sub(r"[^\d.]", "", price_txt))
            except:
                continue

            grams = self._pack_to_grams(unit, name_hint=name)
            if grams and grams > 0:
                price_per_kg = price / (grams / 1000.0)  # LKR per kg
                # Only accept per-kg prices between 50-2000 LKR/kg
                if 50.0 <= price_per_kg <= 2000.0:
                    perkg[_normalize_name_for_lookup(name)] = price_per_kg
                    normalized += 1

        if normalized == 0:
            raise ValueError("FATAL: No price rows normalized to kg. Check column names/pack formats.")

        self.price_data = perkg
        log.info(f"Normalized {normalized} price rows → LKR/kg")

        vals = list(perkg.values())
        lt50 = sum(v < 50 for v in vals)
        mid  = sum(50 <= v <= 2000 for v in vals)
        gt2k = sum(v > 2000 for v in vals)
        log.info(f"Index buckets (LKR/kg): <50={lt50}, 50–2000={mid}, >2000={gt2k}")

        # Build category medians for fallback
        from statistics import median
        self.category_median = {}
        buckets = {}
        for nm, p in self.price_data.items():
            c = _category_from_name(nm)
            if c: buckets.setdefault(c, []).append(p)
        for c, arr in buckets.items():
            if arr: self.category_median[c] = median(arr)

    def _pack_to_grams(self, pack: str, name_hint: str = "") -> Optional[float]:
        s = (pack or "").lower()
        # Remove dots, spaces, commas but preserve meaningful separators  
        s = s.replace(".", "").replace(" ", "").replace(",", "")
        
        # Enhanced pattern to match various pack formats
        m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|l|ltr|ml)", s, re.IGNORECASE)
        if not m:
            # Fallback to original pattern
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if not m:
                return None
            qty = float(m.group(1))
            unit = ""
        else:
            qty = float(m.group(1))
            unit = m.group(2).lower()
            
        # Normalize unit variations
        if unit in ["ltr"]: unit = "l"

        if "kg" in s or unit == "kg": return qty * 1000.0
        if "g" in s or unit == "g": return qty

        if "ml" in s or "l" in s or unit in ["ml", "l"]:
            ml = qty * 1000.0 if (("l" in s and "ml" not in s) or unit == "l") else qty
            dens = 1.0
            for k, v in DENSITIES.items():
                if k in name_hint.lower():
                    dens = v; break
            return ml * dens

        return None

    # -------- Denominator rules --------
    @staticmethod
    def _primary_name(name: str) -> str:
        n = name.lower()
        n = re.sub(r"\b(fresh|large|small|medium|finely|coarsely|ground|crushed|dried|red|green|yellow|ripe|chopped|sliced|minced)\b", "", n)
        return re.sub(r"\s+", " ", n).strip()

    A_DROP_WORDS = {
        "salt","pepper","black pepper","white pepper",
        "water","hot water","cold water","warm water","lukewarm water","boiled water","normal water","boiling water",
        "stock","broth"
    }
    A_DROP_PHRASES = {
        "to taste","as required","as needed","optional","for garnish","for serving","for tempering",
        "for marinade","as you need","as you want","adjust to your taste",
        "pinch","pinch of","a pinch","dash","a dash","few","a little",
        "make a paste","grind to a paste","for paste","for marination"
    }

    def _is_A_drop(self, s: str) -> bool:
        s = s.lower()
        if any(p in s for p in self.A_DROP_PHRASES):
            return True
        if any(w in s for w in self.A_DROP_WORDS):    # substring ok
            return True
        return False

    @staticmethod
    def _is_B_spice(name_l: str) -> bool:
        B_SPICES = {
            "turmeric","turmeric powder","cumin","cumin seed","cumin powder","mustard","mustard seed","mustard seeds",
            "fenugreek","coriander","coriander powder","coriander seed","coriander seeds","chili","chilli",
            "chili powder","chilli powder","chilli flakes","chili flakes","curry leaves","curry leaf",
            "pandan","rampe","cardamom","clove","cloves","cinnamon","cinnamon stick","nutmeg","mace",
            "garam masala","peppercorn","bay leaf","bay leaves","paprika","cayenne","thyme","oregano",
            "basil","rosemary","sage","mint","cilantro","parsley","dill"
        }
        return any(x in name_l for x in B_SPICES)

    # -------- Token parsing & conversion --------
    def parse_qty(self, qty_txt: str) -> float:
        q = qty_txt.replace(",", ".").strip()
        if re.match(r"^\d+\s+\d/\d$", q):
            whole, frac = q.split()
            num, den = frac.split("/")
            return float(int(whole) + float(num)/float(den))
        if re.match(r"^\d+/\d+$", q):
            num, den = q.split("/")
            return float(num)/float(den) if float(den)!=0 else 0.0
        try:
            return float(q)
        except:
            return 0.0

    def _to_grams(self, quantity: float, unit: str, name: str) -> float:
        if quantity <= 0: return 0.0
        u = (unit or "").lower()
        nl = name.lower()

        # metric
        if u == "kg": return quantity * 1000.0
        if u == "g":  return quantity
        if u == "l":
            dens = 1.0
            for k, v in DENSITIES.items():
                if k in nl: dens = v; break
            return quantity * 1000.0 * dens
        if u == "ml":
            dens = 1.0
            for k, v in DENSITIES.items():
                if k in nl: dens = v; break
            return quantity * dens

        # imperial
        if u == "lb": return quantity * 453.592
        if u == "oz": return quantity * 28.3495

        # spoons/cups (crude—filtered by <2g rule)
        if u == "tsp":  return quantity * 2.0
        if u == "tbsp": return quantity * 6.0
        if u == "cup":
            dens = 1.0
            for k, v in DENSITIES.items():
                if k in nl: dens = v; break
            return quantity * 240.0 * dens

        # pieces or missing unit → use piece weights if possible
        if u == "pcs" or u == "":
            for k, w in PIECE_WEIGHTS.items():
                if k in nl: return quantity * w
            return quantity * 10.0  # default pcs

        return quantity * 10.0

    def parse_ingredient_token(self, token: str) -> Tuple[float, str, str]:
        """Return (grams, unit, name) for one normalized token."""
        s = self.normalize_token(token)
        m = re.match(rf'^\s*(\d+(?:[.,]\d+)?(?:\s+\d/\d)?|\d/\d)\s*(?:({UNIT_CANON})\s+)?(.+?)\s*$', s, flags=re.IGNORECASE)
        if m:
            qty_txt, unit, name = m.groups()
            qty = self.parse_qty(qty_txt)
            unit = (unit or "")
            name = self._primary_name(name)
            grams = self._to_grams(qty, unit, name)
            return grams, unit, name

        # fallback: "<qty> name" with no unit
        m2 = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*$', s)
        if m2:
            qty = self.parse_qty(m2.group(1))
            name = self._primary_name(m2.group(2))
            grams = self._to_grams(qty, "", name)
            return grams, "", name

        return 0.0, "", self._primary_name(s)

    # -------- Denominator exclusion --------
    def should_exclude_from_denominator(self, ingredient_name: str, grams: float) -> bool:
        if not ingredient_name: return True
        nl = ingredient_name.lower().strip()
        if self._is_A_drop(nl): return True
        if self._is_B_spice(nl): return True
        if grams < 2.0: return True
        if nl == "water" or nl.endswith(" water"): return True
        if grams > 20.0 and self._is_B_spice(nl): return True  # spice outlier guard
        return False

    # -------- Price lookup (exact → alias → fuzzy) --------
    def find_price_match(self, ingredient_name: str) -> Tuple[str, float]:
        if not ingredient_name: return "", 0.0
        key = _normalize_name_for_lookup(ingredient_name)

        # exact
        if key in self.price_data:
            return key, self.price_data[key]

        # alias
        for k, v in ITEM_ALIAS.items():
            if re.search(rf"\b{k}\b", key):
                alias_key = _normalize_name_for_lookup(v)
                if alias_key in self.price_data:
                    return alias_key, self.price_data[alias_key]

        # fuzzy (favor recall): overlap tokens, then jaccard, then length
        key_tokens = _tokenize_name(key)
        best_name, best_overlap, best_jacc, best_len = "", 0, 0.0, 0
        for dcs_name, price in self.price_data.items():
            cand_tokens = _tokenize_name(dcs_name)
            if not cand_tokens:
                continue
            inter = len(key_tokens & cand_tokens)
            if inter == 0:
                continue
            union = len(key_tokens | cand_tokens) or 1
            jacc = inter / union
            if (inter > best_overlap) or (inter == best_overlap and jacc > best_jacc) or (inter == best_overlap and abs(jacc - best_jacc) < 1e-6 and len(dcs_name) > best_len):
                best_name, best_overlap, best_jacc, best_len = dcs_name, inter, jacc, len(dcs_name)

        if best_overlap >= 1 and best_jacc >= 0.34:
            return best_name, self.price_data.get(best_name, 0.0)

        # Category fallback
        cat = _category_from_name(ingredient_name)
        if cat and cat in getattr(self, "category_median", {}):
            return f"[category:{cat}]", self.category_median[cat]

        return "", 0.0

    # -------- Token / Recipe costing --------
    def calculate_ingredient_cost(self, token: str) -> Tuple[float, bool, str]:
        if not token: return 0.0, False, ""
        grams, unit, ingredient_name = self.parse_ingredient_token(token)
        key_for_debug = ingredient_name or token

        if self.should_exclude_from_denominator(ingredient_name, grams):
            return 0.0, False, key_for_debug

        match_key, price_per_kg = self.find_price_match(ingredient_name)
        if not match_key or price_per_kg <= 0:
            # Track unmatched for reporting
            self.unmatched_reports.append({
                'ingredient_head': canonicalize_name(ingredient_name),
                'count': 1
            })
            return 0.0, True, key_for_debug  # counts, but no price

        cost = (grams / 1000.0) * price_per_kg
        return float(cost), True, key_for_debug

    def calculate_recipe_cost(self, ingredients_str: str, calories: Optional[float] = None) -> Dict:
        if not ingredients_str or pd.isna(ingredients_str):
            return {
                "total_cost": 0.0, "baseline_pass": False, "strict_pass": False, "very_strict_pass": False,
                "match_percentage": 0.0, "denominator_count": 0, "debug_unmatched": []
            }
        tokens = [t.strip() for t in str(ingredients_str).split("|") if t.strip()]
        total_cost = 0.0
        denom = 0
        matched = 0
        debug_unmatched: List[str] = []

        for t in tokens:
            cost, count_in_den, key = self.calculate_ingredient_cost(t)
            if count_in_den:
                denom += 1
                if cost > 0:
                    matched += 1
                    total_cost += cost
                else:
                    debug_unmatched.append(key)

        match_pct = (matched / denom * 100.0) if denom > 0 else 0.0
        baseline_pass = (total_cost > 0) and (match_pct >= 80.0)
        strict_pass   = baseline_pass and (50.0 <= total_cost <= 2000.0)
        very_strict   = strict_pass
        if calories is not None and not pd.isna(calories):
            try:
                very_strict = strict_pass and (150.0 <= float(calories) <= 1500.0)
            except:
                very_strict = strict_pass

        return {
            "total_cost": float(total_cost),
            "baseline_pass": bool(baseline_pass),
            "strict_pass": bool(strict_pass),
            "very_strict_pass": bool(very_strict),
            "match_percentage": float(match_pct),
            "denominator_count": int(denom),
            "debug_unmatched": debug_unmatched
        }

    # -------- Auto-alias learning --------
    def suggest_aliases(self, unmatched_heads: List[str]) -> Dict[str, Dict]:
        """Return {unmatched -> {'suggestion': dcs_name, 'overlap': int, 'jaccard': float}}"""
        suggestions = {}
        for head in unmatched_heads:
            key = _normalize_name_for_lookup(head)
            key_tokens = _tokenize_name(key)
            if not key_tokens:
                continue
            best_name, best_overlap, best_jacc, best_len = "", 0, 0.0, 0
            for dcs_name in self.price_data.keys():
                cand_tokens = _tokenize_name(dcs_name)
                if not cand_tokens:
                    continue
                inter = len(key_tokens & cand_tokens)
                if inter == 0:
                    continue
                union = len(key_tokens | cand_tokens) or 1
                jacc = inter / union
                if (inter > best_overlap) or (inter == best_overlap and jacc > best_jacc) or (inter == best_overlap and abs(jacc - best_jacc) < 1e-6 and len(dcs_name) > best_len):
                    best_name, best_overlap, best_jacc, best_len = dcs_name, inter, jacc, len(dcs_name)
            if best_name:
                suggestions[head] = {"suggestion": best_name, "overlap": best_overlap, "jaccard": round(best_jacc, 3)}
        return suggestions

    def apply_auto_aliases(self, suggestions: Dict[str, Dict]) -> Dict[str, str]:
        """
        Keep only high-confidence suggestions:
          - overlap >= 1 and jaccard >= 0.5  OR
          - token-subset match (src ⊆ dst)
        Returns a flat alias map {unmatched -> suggested_name}
        """
        auto_map: Dict[str, str] = {}
        for src, meta in suggestions.items():
            dst = meta["suggestion"]
            ov  = meta["overlap"]
            j   = meta["jaccard"]
            src_t = _tokenize_name(_normalize_name_for_lookup(src))
            dst_t = _tokenize_name(dst)
            subset_ok = src_t.issubset(dst_t) and len(src_t) > 0
            if (ov >= 1 and j >= 0.34) or subset_ok:
                auto_map[src] = dst
        return auto_map

    def write_unmatched_pricing_report(self, all_unmatched: List[str]):
        """Write comprehensive unmatched pricing report"""
        try:
            os.makedirs('reports', exist_ok=True)
            
            # Aggregate counts
            unmatched_counts = Counter([canonicalize_name(self._primary_name(x)) for x in all_unmatched if x])
            
            if unmatched_counts:
                with open('reports/unmatched_tokens.csv', 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ingredient_head', 'count'])
                    for ingredient_head, count in unmatched_counts.most_common(200):
                        writer.writerow([ingredient_head, count])
                log.info(f"Wrote {len(unmatched_counts)} unmatched pricing entries to reports/unmatched_tokens.csv")
            else:
                # Write empty file with headers
                with open('reports/unmatched_tokens.csv', 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ingredient_head', 'count'])
        except Exception as e:
            log.warning(f"Could not write unmatched pricing report: {e}")

    # -------- CSV processing / reporting --------
    def process_csv_file(self, input_file: str, output_file: str = None):
        log.info(f"Loading recipes from {input_file}")
        df = pd.read_csv(input_file, encoding="utf-8")

        if "ingredients_per_person" not in df.columns:
            raise ValueError("Column 'ingredients_per_person' not found in input CSV")

        # ---- PASS 1 ----
        n = len(df)
        log.info(f"Loaded {n} recipes")
        results = []
        all_unmatched = []

        for _, row in df.iterrows():
            ing = row.get("ingredients_per_person", "")
            cals = row.get("calories", None)
            res = self.calculate_recipe_cost(ing, cals)
            results.append(res)
            all_unmatched.extend(res["debug_unmatched"])

        # (Optional) Auto-alias second pass
        auto_alias_map = {}
        if self.auto_alias:
            top_heads = [k for k, _ in Counter([self._primary_name(x) for x in all_unmatched if x]).most_common(200)]
            suggestions = self.suggest_aliases(top_heads)
            # Save suggestions for visibility
            if suggestions:
                pd.DataFrame(
                    [{"unmatched": k, **v} for k, v in suggestions.items()]
                ).to_csv("alias_suggestions.csv", index=False, encoding="utf-8")
                with open("auto_alias.json", "w", encoding="utf-8") as f:
                    json.dump(suggestions, f, indent=2, ensure_ascii=False)
                log.info("Wrote alias suggestion files: alias_suggestions.csv, auto_alias.json")

            auto_alias_map = self.apply_auto_aliases(suggestions)
            if auto_alias_map:
                # Apply transient aliases for this run
                log.info(f"Auto-applying {len(auto_alias_map)} high-confidence aliases for 2nd pass")
                # Extend ITEM_ALIAS on the fly
                for src, dst in auto_alias_map.items():
                    ITEM_ALIAS[src] = dst

                # ---- PASS 2 ----
                results = []
                all_unmatched = []
                for _, row in df.iterrows():
                    ing = row.get("ingredients_per_person", "")
                    cals = row.get("calories", None)
                    res = self.calculate_recipe_cost(ing, cals)
                    results.append(res)
                    all_unmatched.extend(res["debug_unmatched"])

        # Coverage
        total = len(results)
        baseline_count = sum(1 for r in results if r["baseline_pass"])
        strict_count   = sum(1 for r in results if r["strict_pass"])
        very_count     = sum(1 for r in results if r["very_strict_pass"])

        coverage = PriceCoverage(
            baseline=(baseline_count / total * 100.0) if total else 0.0,
            strict=(strict_count / total * 100.0) if total else 0.0,
            very_strict=(very_count / total * 100.0) if total else 0.0
        )

        # Fail-fast: if Baseline>0 but Strict==0, show buckets + top unmatched
        if coverage.baseline > 0 and coverage.strict == 0:
            costs = [r["total_cost"] for r in results if r["total_cost"] > 0]
            bkt = self.analyze_cost_buckets(costs)
            top = Counter([self._primary_name(x) for x in all_unmatched if x]).most_common(20)
            raise SystemExit(
                "Fail-fast: Baseline>0 but Strict==0\n"
                f"Buckets (LKR/recipe): <50={bkt['under_50']}, 50–2000={bkt['valid_range']}, >2000={bkt['over_2000']}\n"
                "Top unmatched heads:\n  " + "\n  ".join(f"{k} ×{c}" for k, c in top)
            )

        # Attach results and save CSV
        df["estimated_cost_lkr"] = [r["total_cost"] for r in results]
        df["match_percentage"]   = [r["match_percentage"] for r in results]
        df["baseline_pass"]      = [r["baseline_pass"] for r in results]
        df["strict_pass"]        = [r["strict_pass"] for r in results]
        df["very_strict_pass"]   = [r["very_strict_pass"] for r in results]

        if output_file is None:
            base = Path(input_file).stem
            output_file = f"{base}_with_costs.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        log.info(f"Saved priced CSV → {output_file}")

        # Coverage report (write in many key styles so any pipeline reader works)
        targets_met = (coverage.baseline >= 90.0 and coverage.strict >= 85.0 and coverage.very_strict >= 70.0)
        self.write_pricing_coverage_report(coverage, targets_met, total)

        # Buckets + top unmatched
        costs = [r["total_cost"] for r in results if r["total_cost"] > 0]
        bkt = self.analyze_cost_buckets(costs)
        log.info("\nCoverage summary:")
        log.info(f"  Baseline:    {coverage.baseline:.1f}%")
        log.info(f"  Strict:      {coverage.strict:.1f}%")
        log.info(f"  Very-strict: {coverage.very_strict:.1f}%")
        log.info(f"Cost buckets (LKR/recipe): <50={bkt['under_50']}, 50–2000={bkt['valid_range']}, >2000={bkt['over_2000']}")

        # Write comprehensive unmatched report
        self.write_unmatched_pricing_report(all_unmatched)

        # If we auto-applied aliases, persist them
        if auto_alias_map:
            with open("auto_alias_applied.json", "w", encoding="utf-8") as f:
                json.dump(auto_alias_map, f, indent=2, ensure_ascii=False)
            log.info("Wrote auto-applied aliases → auto_alias_applied.json")

        # Console Coverage Report block (mirrors JSON values)
        log.info("\n📄 Coverage Report")
        log.info(f"  Baseline:    {coverage.baseline:.1f}%")
        log.info(f"  Strict:      {coverage.strict:.1f}%")
        log.info(f"  VeryStrict:  {coverage.very_strict:.1f}%")

        # Print coverage for pipeline use
        print(f"PRICING_BASELINE: {coverage.baseline:.1f}")
        print(f"PRICING_STRICT: {coverage.strict:.1f}")
        print(f"PRICING_VERY_STRICT: {coverage.very_strict:.1f}")

        return df, coverage

    @staticmethod
    def analyze_cost_buckets(costs: List[float]) -> Dict[str, int]:
        return {
            "under_50":   sum(1 for c in costs if c < 50.0),
            "valid_range":sum(1 for c in costs if 50.0 <= c <= 2000.0),
            "over_2000":  sum(1 for c in costs if c > 2000.0)
        }

    @staticmethod
    def write_pricing_coverage_report(coverage: PriceCoverage, targets_met: bool, total_recipes: int):
        report = {
            # snake_case for humans / code
            "total_recipes": total_recipes,
            "coverage_percentages": {
                "baseline": round(coverage.baseline, 1),
                "strict": round(coverage.strict, 1),
                "very_strict": round(coverage.very_strict, 1)
            },
            "acceptance_targets": {"baseline": 90.0, "strict": 85.0, "very_strict": 70.0},
            "targets_met": {
                "baseline": coverage.baseline >= 90.0,
                "strict": coverage.strict >= 85.0,
                "very_strict": coverage.very_strict >= 70.0,
                "all_targets": targets_met
            },
            # UpperCamel (compat mode)
            "Coverage": {
                "Baseline": round(coverage.baseline, 1),
                "Strict": round(coverage.strict, 1),
                "VeryStrict": round(coverage.very_strict, 1)
            },
            "TargetsMet": {
                "Baseline": coverage.baseline >= 90.0,
                "Strict": coverage.strict >= 85.0,
                "VeryStrict": coverage.very_strict >= 70.0,
                "AllTargets": targets_met
            },
            # Flat top-level (legacy readers)
            "Baseline": round(coverage.baseline, 1),
            "Strict": round(coverage.strict, 1),
            "VeryStrict": round(coverage.very_strict, 1)
        }
        with open("reports/pricing_coverage_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("📊 Coverage report saved → reports/pricing_coverage_report.json")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Nutrition-enriched recipes CSV (has ingredients_per_person)")
    ap.add_argument("output_csv", nargs="?", help="Output CSV (priced). Default: *_with_costs.csv")
    ap.add_argument("--price-data", default=os.getenv("PRICE_DATA_PATH","config/extracted_prices2.csv"),
                    help="Path to DCS prices CSV (default: config/extracted_prices2.csv or env PRICE_DATA_PATH)")
    ap.add_argument("--auto-alias", action="store_true", help="Enable second pass with high-confidence alias learning")
    args = ap.parse_args()

    calc = PriceCalculator(price_data_path=args.price_data, auto_alias=args.auto_alias)
    result, coverage = calc.process_csv_file(args.input_csv, args.output_csv)
    
    return coverage

if __name__ == "__main__":
    main()
