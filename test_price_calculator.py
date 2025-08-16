#!/usr/bin/env python3
"""
Test script for the updated price calculator with real-time CSV support
"""

import pandas as pd
import logging
from price_calculator import MultithreadedPriceCalculator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_price_loading():
    """Test price loading and normalization"""
    print("="*60)
    print("TESTING PRICE LOADING AND NORMALIZATION")
    print("="*60)
    
    try:
        calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
        
        print(f"Loaded {len(calculator.price_database)} price entries")
        print(f"Sample entries:")
        
        # Show first 5 entries
        count = 0
        for key, value in list(calculator.price_database.items())[:5]:
            print(f"  {key}: LKR {value['price_per_kg']:.2f}/kg (from {value['original_unit']})")
            count += 1
            if count >= 5:
                break
        
        return True
        
    except Exception as e:
        print(f"Price loading test failed: {e}")
        return False

def test_unit_conversion():
    """Test unit conversion logic"""
    print("\n" + "="*60)
    print("TESTING UNIT CONVERSION")
    print("="*60)
    
    try:
        calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
        
        # Test various unit conversions
        test_cases = [
            ("1 kg", "rice", 100.0),
            ("500 g", "flour", 150.0),
            ("1 liter", "milk", 200.0),
            ("750 ml", "oil", 300.0),
            ("Each", "coconut", 180.0),
            ("Bunch", "curry leaves", 50.0),
        ]
        
        for unit, ingredient, price in test_cases:
            try:
                price_per_kg = calculator.convert_to_price_per_kg(price, unit, ingredient)
                print(f"  {ingredient} @ LKR {price}/{unit} = LKR {price_per_kg:.2f}/kg")
            except Exception as e:
                print(f"  Error with {ingredient}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Unit conversion test failed: {e}")
        return False

def test_ingredient_matching():
    """Test ingredient name matching and aliases"""
    print("\n" + "="*60)
    print("TESTING INGREDIENT MATCHING")
    print("="*60)
    
    try:
        calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
        
        # Test ingredient matching
        test_ingredients = [
            "brinjal",
            "lady's finger", 
            "green gram",
            "red onions",
            "tomatoes",
            "potatoes",
            "coconut",
            "rice"
        ]
        
        for ingredient in test_ingredients:
            price_info, method, confidence = calculator.find_ingredient_price_advanced(ingredient)
            if price_info:
                print(f"  {ingredient}: LKR {price_info['price_per_kg']:.2f}/kg "
                      f"(method: {method}, confidence: {confidence:.2f})")
            else:
                print(f"  {ingredient}: NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"Ingredient matching test failed: {e}")
        return False

def test_sample_recipe():
    """Test with a sample recipe"""
    print("\n" + "="*60)
    print("TESTING SAMPLE RECIPE CALCULATION")
    print("="*60)
    
    try:
        # Create a sample recipe DataFrame
        sample_recipe = pd.DataFrame([{
            'title': 'Test Rice and Curry',
            'ingredients_per_person': '1 cup rice, 1 medium onion, 2 tomatoes, 1 tsp salt, 1 tbsp oil',
            'servings': '4',
            'calories': 350
        }])
        
        calculator = MultithreadedPriceCalculator('extracted_prices2.csv', max_workers=2)
        
        # Process the recipe
        result_df, results = calculator.process_recipes_batch_advanced(sample_recipe, 'test_output.csv')
        
        if results:
            result = results[0]
            print(f"Recipe: {result.recipe_title}")
            print(f"Total cost: LKR {result.estimated_cost_lkr:.2f}")
            print(f"Matched ingredients: {result.matched_ingredients}/{result.total_ingredients}")
            print(f"Match percentage: {result.match_percentage:.1f}%")
            print(f"Price freshness: {result.price_freshness_days} days")
            
            if result.ingredient_costs:
                print("\nIngredient breakdown:")
                for ingredient in result.ingredient_costs:
                    if ingredient.total_cost > 0:
                        print(f"  {ingredient.canonical_name}: LKR {ingredient.total_cost:.2f} "
                              f"({ingredient.grams:.1f}g @ {ingredient.price_per_kg:.2f}/kg)")
        
        return True
        
    except Exception as e:
        print(f"Sample recipe test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TESTING UPDATED PRICE CALCULATOR")
    print("Testing with extracted_prices2.csv format")
    print()
    
    tests = [
        test_price_loading,
        test_unit_conversion, 
        test_ingredient_matching,
        test_sample_recipe
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("PASSED")
            else:
                print("FAILED")
        except Exception as e:
            print(f"FAILED: {e}")
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("="*60)
    
    # Check for log files
    import os
    if os.path.exists('unit_warnings.csv'):
        print("Unit warnings logged to: unit_warnings.csv")
    if os.path.exists('missing_prices.csv'):
        print("Missing prices logged to: missing_prices.csv")
    if os.path.exists('test_output.csv'):
        print("Test output saved to: test_output.csv")
    
    return passed == total

if __name__ == "__main__":
    main()