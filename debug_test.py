#!/usr/bin/env python3

import sys
sys.path.append('.')

from serving import IngredientNormalizer

def debug_specific_case():
    """Debug specific failing cases"""
    normalizer = IngredientNormalizer()
    
    # Debug the "cooking oil - as you need" case
    test_ingredient = "cooking oil - as you need"
    print(f"Original: '{test_ingredient}'")
    
    cleaned = normalizer.clean_ingredient_text(test_ingredient)
    print(f"After clean_ingredient_text: '{cleaned}'")
    
    if cleaned:
        normalized = normalizer.normalize_ingredient(test_ingredient, 1, "Test Recipe")
        print(f"After normalize_ingredient: '{normalized}'")
    
    print("\n" + "="*50)
    
    # Debug the "/3 l bbeef" case
    test_ingredient = "/3 l bbeef"
    print(f"Original: '{test_ingredient}'")
    
    cleaned = normalizer.clean_ingredient_text(test_ingredient)
    print(f"After clean_ingredient_text: '{cleaned}'")
    
    if cleaned:
        normalized = normalizer.normalize_ingredient(test_ingredient, 1, "Test Recipe")
        print(f"After normalize_ingredient: '{normalized}'")

if __name__ == "__main__":
    debug_specific_case()