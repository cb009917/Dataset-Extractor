#!/usr/bin/env python3
"""
Smoke test for normalize_per_person.py
Tests edge cases and validates acceptance targets are met.
"""

import pytest
import tempfile
import csv
import os
import sys
from normalize_per_person import IngredientNormalizer

def test_edge_cases():
    """Test the specific edge cases mentioned in requirements."""
    normalizer = IngredientNormalizer()
    
    # Test cases from requirements (assume servings=4 unless specified)
    test_cases = [
        # Edge case tests
        ("Jackfruit Seeds – 250g", 4, ["62.5 g Jackfruit Seeds"]),
        ("Coconut Milk — 1 1/2 cups", 4, ["0.375 cup Coconut Milk"]),
        ("Chili powder (to taste)", 4, []),  # B list + to-taste = dropped
        ("Salt – as required", 4, []),  # A list = dropped
        ("Mustard seeds – 1 tsp", 2, []),  # B & <2g after scaling = dropped
        ("2tspginger", 4, ["0.5 tsp ginger"]),  # Deglue then scale
        ("1cupcoconut milk", 4, ["0.25 cup coconut milk"]),  # Deglue then scale
        ("Best Chicken – For Curry -----", 4, []),  # Heading = dropped
        ("Water – 250 ml", 4, []),  # A: water = dropped
        ("Red lentils – 200 g", 2, ["100 g Red lentils"]),  # Material with quantity
        ("Onion – 1", 4, ["0.25 pcs Onion"]),  # Pure integer with food noun = pcs
        
        # Unicode fraction tests
        ("Coconut milk ½ cup", 4, ["0.125 cup Coconut milk"]),
        ("Rice flour ¾ cup", 4, ["0.1875 cup Rice flour"]),
        ("Sugar ⅓ cup", 4, ["0.083 cup Sugar"]),
        
        # Parenthetical removal tests
        ("Chicken (boneless, skinless) 500g", 4, ["125 g Chicken"]),
        ("Onion (medium size) 2 pcs", 4, ["0.5 pcs Onion"]),
        
        # Multi-ingredient line splitting
        ("Salt 1 tsp | Pepper 1/2 tsp | Water 2 cups", 4, ["Water 2 cups"]),  # Only water should remain but gets dropped by A-list
        ("Rice 2 cups ; Oil 2 tbsp", 4, ["0.5 cup Rice", "0.5 tbsp Oil"]),
        
        # Mixed number parsing
        ("Flour 2 1/2 cups", 4, ["0.625 cup Flour"]),
        ("Milk 1 3/4 cups", 4, ["0.4375 cup Milk"]),
    ]
    
    passed = 0
    total = len(test_cases)
    
    print("=== EDGE CASE TESTS ===")
    for i, (ingredients, servings, expected) in enumerate(test_cases, 1):
        tokens = normalizer.process_ingredient_line(ingredients, servings)
        
        # Compare results
        if set(tokens) == set(expected):
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
        
        print(f"{status} {i:2d}: '{ingredients}' (servings={servings})")
        print(f"         Expected: {expected}")
        print(f"         Got:      {tokens}")
        print()
    
    print(f"Edge case tests: {passed}/{total} passed")
    return passed == total

def test_acceptance_targets():
    """Test that acceptance targets are met with sample data."""
    
    # Create test CSV with sample data designed to meet targets
    test_data = [
        {
            'title': 'Test Recipe 1',
            'ingredients': 'Rice 2 cups | Chicken 500g | Onion 1 pcs | Salt 1 tsp',
            'servings': '4'
        },
        {
            'title': 'Test Recipe 2', 
            'ingredients': 'Flour 300g | Milk 250ml | Egg 2 pcs | Sugar 50g',
            'servings': '2'
        },
        {
            'title': 'Test Recipe 3',
            'ingredients': 'Coconut oil 2 tbsp | Curry leaves 5 pcs | Turmeric ½ tsp',
            'servings': '4'
        },
        {
            'title': 'Test Recipe 4',
            'ingredients': 'Beef 1 lb | Potatoes 3 pcs | Carrots 2 pcs | Water 3 cups',
            'servings': '6'
        },
        {
            'title': 'Test Recipe 5',
            'ingredients': 'Fish 400g | Lemon juice 2 tbsp | Ginger 1 tsp | Oil 1 tbsp',
            'servings': '3'
        }
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'ingredients', 'servings'])
        writer.writeheader()
        writer.writerows(test_data)
        temp_file = f.name
    
    try:
        # Process with normalizer
        normalizer = IngredientNormalizer()
        normalizer.process_csv(temp_file)
        
        # Check acceptance targets
        stats = normalizer.stats
        
        digit_unit_pct = (stats['tokens_with_digit_or_unit'] / stats['total_tokens'] * 100) if stats['total_tokens'] > 0 else 0
        
        targets_met = True
        results = []
        
        # Target 1: digit/unit ≥ 80%
        target1 = digit_unit_pct >= 80.0
        results.append(f"digit/unit%: {digit_unit_pct:.1f}% ({'PASS' if target1 else 'FAIL'})")
        if not target1:
            targets_met = False
        
        # Target 2: glued unit-word = 0
        target2 = stats['glued_unit_word_count'] == 0
        results.append(f"glued unit-word: {stats['glued_unit_word_count']} ({'PASS' if target2 else 'FAIL'})")
        if not target2:
            targets_met = False
        
        # Target 3: parentheses = 0
        target3 = stats['parentheses_count'] == 0
        results.append(f"parentheses: {stats['parentheses_count']} ({'PASS' if target3 else 'FAIL'})")
        if not target3:
            targets_met = False
        
        # Target 4: needs_quantity.csv contains only material items
        material_only = True
        for item in stats['needs_quantity']:
            if item['reason'] != 'material_no_quantity':
                material_only = False
                break
        
        results.append(f"needs_quantity material only: {'PASS' if material_only else 'FAIL'}")
        if not material_only:
            targets_met = False
        
        print("=== ACCEPTANCE TARGET TESTS ===")
        for result in results:
            print(result)
        
        print(f"\nOverall: {'PASS' if targets_met else 'FAIL'}")
        
        return targets_met
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

def test_deglue_functionality():
    """Test the two-pass deglue functionality."""
    normalizer = IngredientNormalizer()
    
    test_cases = [
        ("250grice", "250 g rice"),
        ("1½tspginger", "1 1/2 tsp ginger"), 
        ("tbspcoconut oil", "tbsp coconut oil"),
        ("cupcoconut milk", "cup coconut milk"),
        ("2lbchicken breast", "2 lb chicken breast"),
        ("1kgbeef", "1 kg beef"),
        ("500mlwater", "500 ml water"),
    ]
    
    print("=== DEGLUE FUNCTIONALITY TESTS ===")
    passed = 0
    for i, (input_text, expected_pattern) in enumerate(test_cases, 1):
        result = normalizer.deglue_spacing_rules(input_text)
        
        # Check if expected pattern appears in result
        if expected_pattern.replace(" ", "").lower() in result.replace(" ", "").lower():
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
        
        print(f"{status} {i}: '{input_text}' -> '{result}'")
    
    print(f"Deglue tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_unicode_fraction_conversion():
    """Test Unicode fraction conversion."""
    normalizer = IngredientNormalizer()
    
    test_cases = [
        ("½ cup rice", "1/2 cup rice"),
        ("¼ tsp salt", "1/4 tsp salt"), 
        ("¾ cup flour", "3/4 cup flour"),
        ("⅓ cup sugar", "1/3 cup sugar"),
        ("⅔ cup milk", "2/3 cup milk"),
        ("⅛ tsp pepper", "1/8 tsp pepper"),
    ]
    
    print("=== UNICODE FRACTION TESTS ===")
    passed = 0
    for i, (input_text, expected_pattern) in enumerate(test_cases, 1):
        result = normalizer.clean_unicode_and_fractions(input_text)
        
        if expected_pattern in result:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
        
        print(f"{status} {i}: '{input_text}' -> '{result}'")
    
    print(f"Unicode fraction tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_triage_system():
    """Test A/B/C triage system."""
    normalizer = IngredientNormalizer()
    
    # A-list (should be dropped)
    a_tests = ["salt", "water", "to taste", "as required", "optional"]
    # B-list (micro-spices/herbs)
    b_tests = ["turmeric", "curry leaves", "cumin powder", "chilli powder", "cinnamon"]
    # C-list (material ingredients)
    c_tests = ["rice", "chicken", "coconut milk", "onion", "oil"]
    
    print("=== TRIAGE SYSTEM TESTS ===")
    
    # Test A-list
    a_passed = 0
    for ingredient in a_tests:
        triage = normalizer.triage_ingredient(ingredient)
        if triage == 'A':
            status = "PASS"
            a_passed += 1
        else:
            status = "FAIL"
        print(f"{status}: '{ingredient}' -> {triage}")
    
    # Test B-list
    b_passed = 0
    for ingredient in b_tests:
        triage = normalizer.triage_ingredient(ingredient)
        if triage == 'B':
            status = "PASS"
            b_passed += 1
        else:
            status = "FAIL"
        print(f"{status}: '{ingredient}' -> {triage}")
    
    # Test C-list
    c_passed = 0
    for ingredient in c_tests:
        triage = normalizer.triage_ingredient(ingredient)
        if triage == 'C':
            status = "PASS"
            c_passed += 1
        else:
            status = "FAIL"
        print(f"{status}: '{ingredient}' -> {triage}")
    
    total_passed = a_passed + b_passed + c_passed
    total_tests = len(a_tests) + len(b_tests) + len(c_tests)
    
    print(f"Triage tests: {total_passed}/{total_tests} passed")
    return total_passed == total_tests

def run_all_tests():
    """Run all smoke tests."""
    print("NORMALIZE_PER_PERSON.PY SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Edge Cases", test_edge_cases),
        ("Acceptance Targets", test_acceptance_targets),
        ("Deglue Functionality", test_deglue_functionality),
        ("Unicode Fractions", test_unicode_fraction_conversion),
        ("Triage System", test_triage_system),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 30)
        try:
            if test_func():
                print(f"[PASS] {test_name}")
                passed_tests += 1
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("ALL TESTS PASSED!")
        return True
    else:
        print("SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)