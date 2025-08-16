#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Set UTF-8 encoding for Windows console
if os.name == 'nt':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from serving import IngredientNormalizer

def test_specific_cases():
    """Test the specific cases mentioned in the requirements"""
    normalizer = IngredientNormalizer()
    
    # Required test cases that must pass as specified
    required_test_cases = [
        # "1/4 cup coconut milk" (serves 5) → 0.05 cup coconut milk (no ⁄4)
        ("1/4 cup coconut milk", 5, "0.05 cup coconut milk"),
        # "½ cup coconut milk" (serves 2) → 1/4 cup coconut milk  
        ("½ cup coconut milk", 2, "1/4 cup coconut milk"),
        # "1/3 tspcooking oil" → becomes 1/3 tsp cooking oil
        ("1/3 tspcooking oil", 1, "1/3 tsp cooking oil"),
        # "/3 l bbeef" → becomes 1/3 lb beef (no l b)
        ("/3 l bbeef", 1, "1/3 lb beef"),
        # "cooking oil - as you need" → becomes "cooking oil" (no quantity to normalize)
        ("cooking oil - as you need", 1, "cooking oil"),
        # "0.05⁄4 tsp Sugar" → becomes 0.05 tsp Sugar (no ⁄4)
        ("0.05⁄4 tsp Sugar", 1, "0.05 tsp Sugar"),
    ]
    
    print("SPECIFIC REQUIRED TEST CASES:")
    print("=" * 60)
    
    all_passed = True
    for ingredient, servings, expected in required_test_cases:
        normalized = normalizer.normalize_ingredient(ingredient, servings, "Test Recipe")
        
        if expected is None:
            passed = normalized is None
            result_str = "[SKIPPED]" if normalized is None else f"FAIL: got '{normalized}'"
        else:
            passed = normalized == expected
            result_str = f"{'PASS' if passed else 'FAIL'}: got '{normalized}', expected '{expected}'"
        
        if not passed:
            all_passed = False
            
        # Handle Unicode encoding issues for Windows console
        try:
            print(f"  {ingredient:<30} (serves {servings}) -> {result_str}")
        except UnicodeEncodeError:
            safe_ingredient = ingredient.encode('ascii', 'replace').decode('ascii')
            print(f"  {safe_ingredient:<30} (serves {servings}) -> {result_str}")
    
    print(f"\n{'All required tests PASSED!' if all_passed else 'Some required tests FAILED!'}")
    return all_passed

if __name__ == "__main__":
    test_specific_cases()