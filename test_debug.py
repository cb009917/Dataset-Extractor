#!/usr/bin/env python3
import re

def test_normalize_token(text):
    """Debug version of normalize_token"""
    if not text or not isinstance(text, str):
        return None, {"skip_denominator": False}
    
    t = text.strip()
    if not t:
        return None, {"skip_denominator": False}
    
    print(f"Input: {repr(text)}")
    print(f"After strip: {repr(t)}")
    
    # Strip trailing comments after " - "
    t = re.sub(r"\s+-\s+.*$", "", t)
    print(f"After comment strip: {repr(t)}")
    
    # Remove parentheticals before quantity parse
    t = re.sub(r"\([^)]*\)", "", t)
    print(f"After parenthetical removal: {repr(t)}")
    
    # Collapse common OCR split-letters
    replacements = [
        (r"\bg\s+arlic\b", "garlic"),
        (r"\bl\s+ime\b", "lime"), 
        (r"\bg\s+round\b", "ground"),
        (r"\bc\s+urry\s+l\s+eaves\b", "curry leaves"),
        (r"\bl\s+b\b", "lb"),
        (r"\bl\s+bs?\b", "lb"),  # Handle lbs as well
    ]
    
    for pattern, replacement in replacements:
        if re.search(pattern, t, re.I):
            print(f"Pattern matched: {pattern} -> {replacement}")
            old_t = t
            t = re.sub(pattern, replacement, t, flags=re.I)
            print(f"Replacement: {repr(old_t)} -> {repr(t)}")
    
    print(f"Final result: {repr(t)}")
    return (t if t else None), {"skip_denominator": False}

# Test cases
test_cases = ["g arlic", "l ime", "2ggarlic", "/4 tsp", "cooking oil - as you need"]

for case in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing: {repr(case)}")
    result = test_normalize_token(case)
    print(f"Result: {result}")