#!/usr/bin/env python3
"""
Test script to verify the improved price calculator with coverage targets
"""

import pandas as pd
import logging
from price_calculator import MultithreadedPriceCalculator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_token_normalization():
    """Test the improved token normalization"""
    print("="*60)
    print("TESTING IMPROVED TOKEN NORMALIZATION")
    print("="*60)
    
    calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
    
    test_cases = [
        "cooking oil - as you need",
        "g arlic cloves",
        "l ime juice", 
        "c urry l eaves",
        "250grice flour",
        "tomatoes, chopped",
        "onions / sliced",
        "salt (to taste)",
        "1 tsp turmeric powder",
    ]
    
    print("Normalization results:")
    for ingredient in test_cases:
        normalized = calculator.normalize_ingredient_token(ingredient)
        print(f"  '{ingredient}' -> '{normalized}'")
    
    return True

def test_alias_matching():
    """Test the comprehensive alias mapping"""
    print("\n" + "="*60)
    print("TESTING ALIAS MATCHING")
    print("="*60)
    
    calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
    
    test_ingredients = [
        "okra",
        "eggplant", 
        "bitter gourd",
        "spring onion",
        "curry leaves",
        "cooking oil",
        "green chili",
        "rice",
        "chicken",
        "lentils",
        "coconut milk",
        "lime",
        "potato",
        "onion"
    ]
    
    matches = 0
    total = len(test_ingredients)
    
    print("Alias matching results:")
    for ingredient in test_ingredients:
        price_info, method, confidence = calculator.find_ingredient_price_advanced(ingredient)
        if price_info and confidence >= 0.8:
            matches += 1
            print(f"  MATCH {ingredient}: LKR {price_info['price_per_kg']:.2f}/kg ({method})")
        else:
            print(f"  NO MATCH {ingredient}")
    
    match_rate = (matches / total) * 100
    print(f"\nAlias match rate: {match_rate:.1f}% ({matches}/{total})")
    
    return match_rate >= 80

def test_unit_conversions():
    """Test the improved unit conversions"""
    print("\n" + "="*60)
    print("TESTING UNIT CONVERSIONS")
    print("="*60)
    
    calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
    
    test_cases = [
        ("1 kg.", "rice", 200.0),
        ("500 g", "flour", 150.0),
        ("250 g", "sugar", 100.0),
        ("750 ml", "coconut oil", 600.0),
        ("Each", "coconut", 180.0),
        ("Bunch", "gotukola", 50.0),
        ("100 leaves", "curry leaves", 300.0),
        ("100 Nuts", "arecanuts", 1000.0),
    ]
    
    print("Unit conversion results:")
    for unit, ingredient, price in test_cases:
        try:
            price_per_kg = calculator.convert_to_price_per_kg(price, unit, ingredient)
            print(f"  {ingredient} @ LKR {price}/{unit} = LKR {price_per_kg:.2f}/kg")
        except Exception as e:
            print(f"  ERROR: {ingredient} @ {unit}: {e}")
    
    return True

def test_sample_recipes():
    """Test with sample recipes to check overall performance"""
    print("\n" + "="*60)
    print("TESTING SAMPLE RECIPES")
    print("="*60)
    
    # Create sample recipes that should match well
    sample_recipes = pd.DataFrame([
        {
            'title': 'Rice and Curry',
            'ingredients_per_person': '1 cup rice, 1 onion chopped, 2 tomatoes, curry leaves, 1 tsp salt, 2 tbsp coconut oil',
            'servings': '4',
            'calories': 350
        },
        {
            'title': 'Chicken Curry',
            'ingredients_per_person': '500g chicken pieces, 1 large onion, 3 cloves garlic, 1 inch ginger, green chili, coconut milk',
            'servings': '4',
            'calories': 450
        },
        {
            'title': 'Vegetable Curry',
            'ingredients_per_person': 'eggplant, okra, potato, onion, garlic, curry leaves, coconut oil',
            'servings': '4', 
            'calories': 250
        }
    ])
    
    calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
    
    # Process the recipes
    result_df, results = calculator.process_recipes_batch_advanced(sample_recipes, 'test_improved_output.csv')
    
    print("Sample recipe results:")
    total_cost = 0
    total_match = 0
    valid_recipes = 0
    
    for result in results:
        if result.estimated_cost_lkr > 0:
            total_cost += result.estimated_cost_lkr
            total_match += result.match_percentage
            valid_recipes += 1
            print(f"  {result.recipe_title}:")
            print(f"    Cost: LKR {result.estimated_cost_lkr:.2f}")
            print(f"    Match: {result.match_percentage:.1f}%")
            print(f"    Matched: {result.matched_ingredients}/{result.total_ingredients}")
    
    if valid_recipes > 0:
        avg_cost = total_cost / valid_recipes
        avg_match = total_match / valid_recipes
        print(f"\nAverage cost: LKR {avg_cost:.2f}")
        print(f"Average match: {avg_match:.1f}%")
        
        return avg_match >= 80 and avg_cost > 0
    
    return False

def main():
    """Run all improvement tests"""
    print("TESTING IMPROVED PRICE CALCULATOR")
    print("Goal: Restore working prices and push coverage targets")
    print()
    
    tests = [
        ("Token Normalization", test_token_normalization),
        ("Alias Matching", test_alias_matching),
        ("Unit Conversions", test_unit_conversions), 
        ("Sample Recipes", test_sample_recipes)
    ]
    
    passed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"PASS {test_name}")
                passed += 1
            else:
                print(f"FAIL {test_name}")
        except Exception as e:
            print(f"ERROR {test_name}: {e}")
    
    print("\n" + "="*60)
    print(f"IMPROVEMENT TEST SUMMARY: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if passed == len(tests):
        print("All improvements working correctly!")
        print("Ready to test coverage targets with full dataset.")
    else:
        print("Some improvements need attention.")
    
    # Check for generated files
    import os
    print("\nGenerated files:")
    for filename in ['test_improved_output.csv', 'unit_warnings.csv', 'missing_prices.csv', 'unmatched_tokens.csv']:
        if os.path.exists(filename):
            print(f"  EXISTS {filename}")
        else:
            print(f"  MISSING {filename}")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()