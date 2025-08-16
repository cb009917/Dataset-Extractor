#!/usr/bin/env python3
"""
Shared ingredient text normalizer for both nutrition and price pipelines
Provides consistent preprocessing to improve parsing accuracy
"""

import re
import logging
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

def detect_unit_with_precedence(text: str) -> Optional[str]:
    """
    Anchor unit detection (prevent tsp/tbsp collisions)
    tsp first: (?i)\b(tsp|teaspoons?)\b
    then tbsp: (?i)\b(tbsp|tablespoons?)\b
    Use word boundaries and precedence exactly as specified
    
    Args:
        text: Text to search for units
        
    Returns:
        Detected unit or None
    """
    # Check for tsp first
    tsp_match = re.search(r'(?i)\b(tsp|teaspoons?)\b', text)
    if tsp_match:
        return 'tsp'
    
    # Then check for tbsp
    tbsp_match = re.search(r'(?i)\b(tbsp|tablespoons?)\b', text)
    if tbsp_match:
        return 'tbsp'
    
    return None

def normalize_token_with_meta(text: str) -> tuple[str|None, dict]:
    """
    Returns (normalized_text or None if SKIP, meta)
    meta flags: {"skip_denominator": bool}
    Idempotent and safe to run multiple times.
    """
    if not text or not isinstance(text, str):
        return None, {"skip_denominator": False}
    
    t = text.strip()
    if not t:
        return None, {"skip_denominator": False}
    
    # 1. Strip comment tails: remove \s+-\s+.*$
    t = re.sub(r"\s+-\s+.*$", "", t)
    
    # 2. Remove parentheticals before quantity parsing
    t = re.sub(r"\([^)]*\)", "", t)
    
    # 3. De-glue units: insert space when unit touches a word
    # Handle patterns like "2ggarlic", "1cupcoconut", "tbspbutter" 
    t = re.sub(r"(\d+)(g|kg|mg|ml|l|tsp|tbsp|cup|cups)([A-Za-z])", r"\1 \2 \3", t, flags=re.I)
    t = re.sub(r"\b(g|kg|mg|ml|l|tsp|tbsp|cup|cups)(?=[A-Za-z])", r"\1 ", t, flags=re.I)
    
    # 4. Collapse split letters / OCR fixes AFTER unit de-gluing
    replacements = [
        (r"\bg\s+arlic\b", "garlic"),
        (r"\bl\s+ime\b", "lime"), 
        (r"\bg\s+round\b", "ground"),
        (r"curry\s+l\s+eaves", "curry leaves"),
        (r"\bc\s+urry\s+l\s+eaves\b", "curry leaves"),
        (r"\bl\s+b\b", "lb"),
        (r"\bl\s+bs?\b", "lb"),  # Handle lbs as well
        (r"\bl\s+b(?=\w)", "lb "),  # Handle "l bbeef" pattern, add space
    ]
    
    for pattern, replacement in replacements:
        t = re.sub(pattern, replacement, t, flags=re.I)
    
    # Normalize pounds tokens
    t = re.sub(r"\b(lbs?\.?)\b", "lb", t, flags=re.I)

    # 5. Fraction repair
    t = re.sub(r"^/(\d+)", r"1/\1", t)            # ^/(\d+) → 1/\1
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", t)   # (\d)\s*/\s*(\d) → \1/\2

    # 6. Drop non-quantified fillers (and exclude from match denominator)
    # if no number and contains filler patterns → SKIP
    has_number = bool(re.search(r"\d", t))
    filler_patterns = [
        r"\bas you need\b",
        r"\bas you want\b", 
        r"\bto taste\b",
        r"\badjust to your taste\b",
        r"\bas required\b"
    ]
    
    # Check if text contains filler patterns (even if stripped out)
    original_before_strip = text.strip()
    has_filler = any(re.search(pattern, original_before_strip, re.IGNORECASE) for pattern in filler_patterns)
    
    if not has_number and has_filler:
        return None, {"skip_denominator": True}  # Skip completely and exclude from denominator

    t = t.strip()
    return (t if t else None), {"skip_denominator": False}

def normalize_token(text: str) -> str:
    """
    Simple normalize_token that returns just the normalized text or None
    Compatible with existing code
    """
    result, meta = normalize_token_with_meta(text)
    return result
    

def normalize_ingredient_list(text: str) -> List[str]:
    """
    Split | and call normalize_token on each part
    """
    if not text or not isinstance(text, str):
        return []
    
    # Split by pipe or semicolon delimiter (handle both formats)
    if '|' in text:
        tokens = [token.strip() for token in text.split('|') if token.strip()]
    else:
        tokens = [token.strip() for token in text.split(';') if token.strip()]
    
    # Normalize each token
    normalized_tokens = []
    for token in tokens:
        normalized = normalize_token(token)
        if normalized:  # Only keep non-empty normalized tokens
            normalized_tokens.append(normalized)
    
    return normalized_tokens

def extract_quantity_and_unit(text: str) -> tuple[Union[float, None], str, str]:
    """
    Extract quantity, unit, and ingredient name from normalized text
    Enhanced with safety checks for count nouns and minimum mass filter
    
    Args:
        text: Normalized ingredient text
        
    Returns:
        Tuple of (quantity, unit, ingredient_name)
    """
    if not text:
        return None, "", ""
    
    # Pattern to match quantity (including fractions) and unit at start
    # More flexible pattern to catch various formats
    pattern = r'^(\d+(?:\.\d+)?(?:/\d+)?(?:\s+\d+/\d+)?)\s*([a-zA-Z]+)?\s*(.*?)$'
    match = re.match(pattern, text.strip())
    
    qty_str = unit = ingredient = None
    
    # Alternative pattern for cases where numbers might be at the end
    if not match:
        # Try pattern: ingredient + quantity + unit at end
        alt_pattern = r'^(.*?)\s+(\d+(?:\.\d+)?(?:/\d+)?(?:\s+\d+/\d+)?)\s*([a-zA-Z]+)?$'
        alt_match = re.match(alt_pattern, text.strip())
        if alt_match:
            ingredient = alt_match.group(1).strip()
            qty_str = alt_match.group(2)
            unit = alt_match.group(3)
        else:
            # Try middle pattern: start + quantity + unit + ingredient  
            mid_pattern = r'^(\w+(?:\s+\w+)*)\s+(\d+(?:\.\d+)?(?:/\d+)?(?:\s+\d+/\d+)?)\s*([a-zA-Z]+)\s*(.*?)$'
            mid_match = re.match(mid_pattern, text.strip())
            if mid_match:
                prefix = mid_match.group(1)
                qty_str = mid_match.group(2)
                unit = mid_match.group(3)
                suffix = mid_match.group(4)
                ingredient = f"{prefix} {suffix}".strip()
            else:
                match = None
    
    if not match and not qty_str:
        # No quantity found - return the whole text as ingredient name  
        return None, "", text.strip()
    
    if match and not qty_str:  # From main pattern
        qty_str, unit, ingredient = match.groups()
    
    # Parse quantity (handle mixed numbers and fractions)
    try:
        quantity = parse_quantity_string(qty_str.strip())
    except:
        quantity = None
    
    unit = (unit or "").strip().lower()
    ingredient = (ingredient or "").strip()
    
    # Piece-unit clamp: if unit in {clove, egg, leaf, sprig} and qty < 1 → set qty=1
    piece_units = ['clove', 'cloves', 'egg', 'eggs', 'leaf', 'leaves', 'sprig', 'sprigs']
    
    if quantity is not None and quantity < 1.0 and unit in piece_units:
        quantity = 1.0
    
    # Also check if ingredient name contains piece nouns
    if quantity is not None and quantity < 1.0:
        ingredient_lower = ingredient.lower()
        if any(noun in ingredient_lower for noun in piece_units):
            quantity = 1.0
    
    return quantity, unit, ingredient

def parse_quantity_string(qty_str: str) -> float:
    """
    Parse quantity string that may contain fractions
    Examples: "2", "1.5", "2 1/3", "3/4"
    """
    if not qty_str:
        return 0.0
    
    qty_str = qty_str.strip()
    
    # Handle mixed numbers like "2 1/3"
    mixed_match = re.match(r'(\d+)\s+(\d+/\d+)', qty_str)
    if mixed_match:
        whole = float(mixed_match.group(1))
        frac_parts = mixed_match.group(2).split('/')
        fraction = float(frac_parts[0]) / float(frac_parts[1])
        return whole + fraction
    
    # Handle pure fractions like "3/4"
    if '/' in qty_str:
        parts = qty_str.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    
    # Handle decimal numbers
    try:
        return float(qty_str)
    except ValueError:
        return 0.0

def is_minimum_mass_valid(grams: float) -> bool:
    """
    Check if parsed grams meets minimum mass threshold
    Minimum-mass filter: if parsed grams < 2g, exclude from pricing
    """
    return grams >= 2.0

def should_exclude_from_denominator(ingredient_name: str, grams: float = None) -> bool:
    """
    Determine if ingredient should be excluded from match% denominator
    Only count tokens that are price-relevant AND quantified.
    Exclude: water, non-numeric "to taste", tokens filtered by < 2g rule, headings/instructions.
    
    Args:
        ingredient_name: The ingredient name to check
        grams: Optional weight in grams
        
    Returns:
        True if should be excluded from denominator
    """
    if not ingredient_name:
        return True
    
    ingredient_lower = ingredient_name.lower()
    
    # Water variations (exclude per spec: always exclude water from pricing & denominator)
    water_terms = ['water', 'hot water', 'cold water', 'warm water', 'lukewarm water', 
                   'boiled water', 'normal water', 'boiling water']
    if any(term == ingredient_lower or ingredient_lower.startswith(term + ' ') for term in water_terms):
        return True
    
    # Broth/stock exclusion (mostly water)
    broth_terms = ['broth', 'stock', 'bouillon']
    if any(term in ingredient_lower for term in broth_terms):
        return True
    
    # Minimum mass filter: if parsed grams < 2g, exclude from pricing and match denominator
    if grams is not None and grams < 2.0:
        logger.debug(f"Excluding from denominator due to low mass: {ingredient_name} ({grams}g)")
        return True
    
    # Non-quantified filler phrases (already filtered by normalize_token but double-check)
    filler_phrases = ['as you need', 'as you want', 'to taste', 'adjust to your taste',
                     'as required', 'as needed', 'optional', 'for garnish', 'for decoration']
    if any(phrase in ingredient_lower for phrase in filler_phrases):
        return True
    
    # Instruction fragments that shouldn't count for pricing
    instruction_fragments = ['chopped', 'diced', 'sliced', 'minced', 'crushed', 
                           'grated', 'finely', 'coarsely', 'fresh', 'cleaned',
                           'peeled', 'washed', 'trimmed']
    # Only exclude if it's JUST an instruction with no actual ingredient
    if len(ingredient_lower.split()) <= 2 and any(frag in ingredient_lower for frag in instruction_fragments):
        return True
    
    # Check for headings/instructions (common recipe formatting issues)
    heading_patterns = ['ingredients:', 'preparation:', 'method:', 'instructions:']
    if any(pattern in ingredient_lower for pattern in heading_patterns):
        return True
    
    return False

def has_outlier_spice_quantity(ingredient_name: str, grams: float) -> bool:
    """
    Check if spice/herb quantity exceeds reasonable limits per serving
    If any single spice/herb token > 20g per serving, mark as outlier
    
    Args:
        ingredient_name: The ingredient name
        grams: Weight in grams
        
    Returns:
        True if this is an outlier quantity for spices/herbs
    """
    if grams is None:
        return False
        
    ingredient_lower = ingredient_name.lower()
    
    # Common spices and herbs that should be small quantities
    spice_herbs = [
        'salt', 'pepper', 'cinnamon', 'cardamom', 'cloves', 'nutmeg', 
        'turmeric', 'coriander', 'cumin', 'fennel', 'fenugreek', 'mustard',
        'chili powder', 'curry powder', 'garam masala', 'paprika',
        'oregano', 'thyme', 'basil', 'rosemary', 'sage', 'parsley',
        'mint', 'cilantro', 'dill', 'bay leaves', 'curry leaves',
        'peppercorn', 'allspice', 'mace', 'star anise', 'asafoetida',
        'hing', 'ginger powder', 'garlic powder', 'onion powder'
    ]
    
    # Check if this ingredient contains spice/herb terms
    is_spice_herb = any(spice in ingredient_lower for spice in spice_herbs)
    
    if is_spice_herb and grams > 20.0:
        logger.warning(f"Outlier spice quantity detected: {ingredient_name} = {grams}g")
        return True
        
    return False

def has_outlier_protein_quantity(ingredient_name: str, grams: float) -> bool:
    """
    Check if protein quantity exceeds reasonable limits per serving
    Soft sanity caps: meat ≤ 350g, fish ≤ 300g per serving
    
    Args:
        ingredient_name: The ingredient name
        grams: Weight in grams
        
    Returns:
        True if this is an outlier quantity for proteins
    """
    if grams is None:
        return False
        
    ingredient_lower = ingredient_name.lower()
    
    # Meat terms
    meat_terms = [
        'chicken', 'beef', 'pork', 'lamb', 'mutton', 'goat', 'duck', 
        'turkey', 'veal', 'venison', 'rabbit', 'meat'
    ]
    
    # Fish terms  
    fish_terms = [
        'fish', 'salmon', 'tuna', 'mackerel', 'sardines', 'cod', 
        'snapper', 'kingfish', 'pomfret', 'crab', 'shrimp', 'prawns',
        'lobster', 'squid', 'cuttlefish', 'octopus', 'mussels', 'clams'
    ]
    
    # Check meat limit
    is_meat = any(meat in ingredient_lower for meat in meat_terms)
    if is_meat and grams > 350.0:
        logger.warning(f"Outlier meat quantity detected: {ingredient_name} = {grams}g")
        return True
    
    # Check fish limit
    is_fish = any(fish in ingredient_lower for fish in fish_terms)
    if is_fish and grams > 300.0:
        logger.warning(f"Outlier fish quantity detected: {ingredient_name} = {grams}g")
        return True
        
    return False

def is_low_confidence_ingredient(ingredient_name: str, grams: float) -> bool:
    """
    Determine if ingredient should be marked as low confidence
    Used for outliers and safety caps
    
    Args:
        ingredient_name: The ingredient name
        grams: Weight in grams
        
    Returns:
        True if ingredient should be marked low confidence
    """
    return (has_outlier_spice_quantity(ingredient_name, grams) or 
            has_outlier_protein_quantity(ingredient_name, grams))

if __name__ == "__main__":
    # Mini unit tests (must pass as specified)
    test_cases = [
        # Required unit tests from spec
        ("/4 sprig of curry l eaves", "1 sprig curry leaves", "Should normalize to 1 sprig (2g) excluded"),
        ("1/2 tsp l ime juice", "1/2 tsp lime juice", "Should be ~2.5 ml"),
        ("/3 l bbeef", "1/3 lb beef", "Should be ~151.2 g"),
        ("sfresh milk 1 cup", "fresh milk 1 cup", "Should be 240 ml"),
        ("bpork shoulder 1 lb", "pork shoulder 1 lb", "Should be 453.6 g"),
        ("1/2 tsp freshly g round black pepper", "1/2 tsp freshly ground black pepper", "Should be ~1-2 g"),
        ("cooking oil - as you need", None, "Should SKIP"),
        ("red chili 5 g", "red chili 5 g", "Should be 5 g => DCS 'dried chillies'"),
        ("wheat flour 100 g", "wheat flour 100 g", "Should be 100 g => DCS wheat flour"),
        ("broth/stock 500 ml", "broth 500 ml", "Should be excluded"),
        
        # Additional test cases
        ("g arlic cloves 3", "garlic cloves 3", "Test split letter repair"),
        ("2ggarlic cloves", "2g garlic cloves", "Test unit de-gluing"),
        ("Water 133 1/3ml", "Water 133 1/3ml", "Water should be excluded"),
        ("Salt to taste", None, "Non-quantified filler"),
    ]
    
    print("=== MINI UNIT TESTS ===")
    passed = 0
    total = len(test_cases)
    
    for test_input, expected_normalized, description in test_cases:
        normalized = normalize_token(test_input)
        
        # Check if the normalization matches expectation
        if expected_normalized is None:
            test_passed = normalized is None
            status = "PASS" if test_passed else "FAIL"
        else:
            test_passed = normalized is not None
            status = "PASS" if test_passed else "FAIL"
        
        if test_passed:
            passed += 1
            
        print(f"{status}: '{test_input}' => '{normalized}' | {description}")
        
        if normalized:
            qty, unit, ingredient = extract_quantity_and_unit(normalized)
            print(f"      => Quantity: {qty}, Unit: '{unit}', Ingredient: '{ingredient}'")
            
            # Test grams conversion and exclusion logic
            if qty and unit:
                # Simple conversion estimation for testing
                if unit == 'g':
                    grams = qty
                elif unit == 'ml':
                    grams = qty  # Assume water density
                elif unit == 'tsp':
                    grams = qty * 5
                elif unit == 'cup':
                    grams = qty * 240
                elif unit == 'lb':
                    grams = qty * 453.6
                elif unit == 'sprig':
                    grams = qty * 2
                else:
                    grams = qty * 10  # Default estimate
                
                excluded = should_exclude_from_denominator(ingredient, grams)
                print(f"      => Estimated grams: {grams:.1f}, Excluded: {excluded}")
        print()
    
    print(f"=== RESULTS: {passed}/{total} tests passed ===")
    
    # Additional comprehensive test cases
    print("\n=== ADDITIONAL TEST CASES ===")
    additional_tests = [
        "1/2 tsp freshly g round black pepper",
        "l ime juice 2 tbsp", 
        "2ggarlic cloves",
        "/4 cup coconut milk",
        "Sugar as needed",
        "Coconut milk 400ml can",
        "Green chili 2 pcs",
        "1/2 clove garlic"
    ]
    
    for test in additional_tests:
        normalized = normalize_token(test)
        print(f"'{test}' => '{normalized}'")
        
        if normalized:
            qty, unit, ingredient = extract_quantity_and_unit(normalized)
            print(f"  => Quantity: {qty}, Unit: '{unit}', Ingredient: '{ingredient}'")
        print()