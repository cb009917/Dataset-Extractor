#!/usr/bin/env python3

import sys
sys.path.append('.')

from serving import IngredientNormalizer

def debug_edge_case():
    """Debug the failing edge case"""
    normalizer = IngredientNormalizer()
    
    # Debug the "0.05⁄4 tsp Sugar" case
    test_ingredient = "0.05⁄4 tsp Sugar"
    print(f"Original: '{test_ingredient}'")
    print(f"Unicode char code: {ord('⁄')}")  # should be 8260
    
    cleaned = normalizer.clean_ingredient_text(test_ingredient)
    print(f"After clean_ingredient_text: '{cleaned}'")
    
    if cleaned:
        normalized = normalizer.normalize_ingredient(test_ingredient, 1, "Test Recipe")
        print(f"After normalize_ingredient: '{normalized}'")
    else:
        print("Cleaned ingredient is None!")

if __name__ == "__main__":
    debug_edge_case()