#!/usr/bin/env python3
"""
Shared utilities for FitFeast ETL pipeline
Provides canonicalization and aliasing functions used by both nutrition and price calculators
"""

import json
import os
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Shared alias file path
SHARED_ALIAS_FILE = "config/item_alias_user.json"

def load_shared_aliases() -> Dict[str, str]:
    """Load shared aliases from config/item_alias_user.json"""
    if not os.path.exists(SHARED_ALIAS_FILE):
        # Create empty file if missing
        os.makedirs(os.path.dirname(SHARED_ALIAS_FILE), exist_ok=True)
        with open(SHARED_ALIAS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        logger.info(f"Created empty alias file: {SHARED_ALIAS_FILE}")
        return {}
    
    try:
        with open(SHARED_ALIAS_FILE, 'r', encoding='utf-8') as f:
            aliases = json.load(f)
        # Normalize to lowercase keys and string values
        return {str(k).lower().strip(): str(v).strip() for k, v in aliases.items() 
                if isinstance(k, str) and isinstance(v, str) and v.strip()}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load shared aliases from {SHARED_ALIAS_FILE}: {e}")
        return {}

def save_shared_aliases(aliases: Dict[str, str]) -> None:
    """Save aliases to the shared file"""
    try:
        os.makedirs(os.path.dirname(SHARED_ALIAS_FILE), exist_ok=True)
        with open(SHARED_ALIAS_FILE, 'w', encoding='utf-8') as f:
            json.dump(aliases, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(aliases)} aliases to {SHARED_ALIAS_FILE}")
    except IOError as e:
        logger.error(f"Could not save aliases to {SHARED_ALIAS_FILE}: {e}")

def canonicalize_name(name: str) -> str:
    """
    Canonicalize ingredient name for consistent matching
    
    Steps:
    1. Lowercase and strip punctuation
    2. Collapse whitespace  
    3. Unit aliasing (teaspoons→tsp, litres→l, etc)
    4. Sri Lankan name mapping (brinjal→eggplant, etc)
    """
    if not name or not isinstance(name, str):
        return ""
    
    # 1. Lowercase, strip punctuation, collapse whitespace
    s = name.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)  # Replace punctuation with spaces
    s = re.sub(r'\s+', ' ', s).strip()
    
    # 2. Unit aliasing
    unit_aliases = {
        'teaspoons': 'tsp', 'teaspoon': 'tsp',
        'tablespoons': 'tbsp', 'tablespoon': 'tbsp', 
        'litres': 'l', 'liters': 'l', 'liter': 'l', 'litre': 'l',
        'grams': 'g', 'gram': 'g',
        'kilograms': 'kg', 'kilogram': 'kg',
        'milliliters': 'ml', 'milliliter': 'ml',
        'pieces': 'pcs', 'piece': 'pcs', 'pc': 'pcs',
        'cloves': 'clove', 'ounces': 'oz', 'ounce': 'oz',
        'pounds': 'lb', 'pound': 'lb', 'lbs': 'lb',
        'cups': 'cup', 'packets': 'pkt', 'packet': 'pkt', 'pack': 'pkt',
        'bunches': 'bunch', 'cans': 'can', 'tins': 'tin'
    }
    
    for old_unit, new_unit in unit_aliases.items():
        s = re.sub(rf'\b{old_unit}\b', new_unit, s)
    
    # 3. Sri Lankan name mapping
    sl_mappings = {
        'brinjal': 'eggplant',
        'ladies finger': 'okra', 
        'long beans': 'green beans',
        'leeks': 'leek',
        'big onion': 'onion',
        'red onion': 'onion',
        'spring onion': 'green onion',
        'sprats': 'dried sprats',
        'maldive fish': 'dried fish',
        'gotukola': 'pennywort',
        'mukunuwenna': 'amaranth leaves',
        'dhal': 'red lentils',
        'dal': 'red lentils',
        'green gram': 'mung beans',
        'atta': 'wheat flour',
        'plain flour': 'wheat flour',
        'ap flour': 'wheat flour',
        'all purpose flour': 'wheat flour',
        'coconut cream': 'coconut milk',
        'coconut powder': 'coconut milk',
        'thick coconut milk': 'coconut milk',
        'thin coconut milk': 'coconut milk',
        'fresh coconut': 'coconut',
        'grated coconut': 'coconut',
        'desiccated coconut': 'coconut'
    }
    
    for sl_name, canonical in sl_mappings.items():
        s = re.sub(rf'\b{sl_name}\b', canonical, s)
    
    return s.strip()

def apply_shared_aliases(name: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """Apply shared aliases to a canonicalized name"""
    if aliases is None:
        aliases = load_shared_aliases()
    
    canonical = canonicalize_name(name)
    
    # Apply user aliases
    for alias, target in aliases.items():
        if alias in canonical:
            canonical = canonical.replace(alias, target)
    
    return canonical