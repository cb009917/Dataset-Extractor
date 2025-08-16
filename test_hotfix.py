#!/usr/bin/env python3
"""
Test script for the pricing coverage hotfix
"""

import pandas as pd
import os
from price_calculator import MultithreadedPriceCalculator

def test_pricing_hotfix():
    """Test the pricing hotfix to achieve 90%+ coverage"""
    print("="*80)
    print("TESTING PRICING COVERAGE HOTFIX")
    print("="*80)
    
    # Check for required files
    recipe_file = 'recipes_normalized_robust_nutrition.csv'
    price_file = 'extracted_prices2.csv'
    
    if not os.path.exists(recipe_file):
        print(f"Recipe file not found: {recipe_file}")
        return False
    
    if not os.path.exists(price_file):
        print(f"Price database not found: {price_file}")
        return False
    
    # Load data
    print(f"Loading recipes from {recipe_file}")
    recipes_df = pd.read_csv(recipe_file)
    print(f"Loaded {len(recipes_df)} recipes")
    
    # Use full dataset for final test
    test_df = recipes_df
    print(f"Testing with {len(test_df)} recipes (FULL DATASET)")
    
    # Initialize calculator
    calculator = MultithreadedPriceCalculator(price_file, max_workers=4)
    
    # Process recipes
    print(f"\nStarting price calculation with hotfix improvements...")
    output_df, results = calculator.process_recipes_batch_advanced(test_df, 'test_results.csv')
    
    # Show summary
    print(f"\nHotfix test completed!")
    print(f"Results saved to test_results.csv")
    
    # Quick stats
    total = len(results)
    with_costs = sum(1 for r in results if r.estimated_cost_lkr > 0)
    high_match = sum(1 for r in results if r.match_percentage >= 80)
    low_confidence = sum(1 for r in results if r.low_confidence)
    
    print(f"\nQuick Stats:")
    print(f"  Total recipes: {total}")
    print(f"  With costs: {with_costs} ({with_costs/total*100:.1f}%)")
    print(f"  High match (>=80%): {high_match} ({high_match/total*100:.1f}%)")
    print(f"  Low confidence: {low_confidence} ({low_confidence/total*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    test_pricing_hotfix()