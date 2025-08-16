#!/usr/bin/env python3
"""
Test script to validate pricing coverage targets
"""

import pandas as pd
import numpy as np
from price_calculator import MultithreadedPriceCalculator
import json
import os

def test_pricing_coverage():
    """Test the pricing system with coverage validation"""
    
    print("PRICING COVERAGE TEST")
    print("=" * 50)
    
    # Load recipe data
    recipe_file = 'dataset/authentic_sri_lankan_recipes.csv'
    if not os.path.exists(recipe_file):
        print(f"Recipe file not found: {recipe_file}")
        return
    
    recipes = pd.read_csv(recipe_file)
    print(f"Loaded {len(recipes)} recipes")
    
    # Initialize calculator
    calc = MultithreadedPriceCalculator()
    
    # Process small sample first for testing
    sample_size = min(50, len(recipes))  
    sample = recipes.head(sample_size)
    print(f"Testing with {sample_size} recipes...")
    
    # Process recipes by creating RecipeCost objects manually
    results = []
    
    for idx, recipe in sample.iterrows():
        try:
            # Format data as expected by calculator
            recipe_data = (idx, recipe)
            result = calc.calculate_recipe_cost_advanced(recipe_data)
            results.append(result)
            
            if len(results) % 10 == 0:
                print(f"Processed {len(results)} recipes...")
                
        except Exception as e:
            print(f"Error processing recipe {idx}: {e}")
            continue
    
    print(f"Successfully processed {len(results)} recipes")
    
    if not results:
        print("No recipes processed successfully")
        return
    
    # Calculate coverage statistics
    total_recipes = len(results)
    
    # Baseline: cost > 0 and match% >= 80
    baseline_recipes = [r for r in results if r.total_cost_lkr > 0 and r.match_percentage >= 80]
    baseline_coverage = len(baseline_recipes) / total_recipes * 100
    
    # Strict: 50 <= cost <= 2000 and match% >= 80
    strict_recipes = [r for r in results if 50 <= r.total_cost_lkr <= 2000 and r.match_percentage >= 80]
    strict_coverage = len(strict_recipes) / total_recipes * 100
    
    # Very-strict: Strict + 150 <= kcal <= 1500 (if no kcal, mirror Strict)
    very_strict_recipes = []
    for r in strict_recipes:
        if r.calories is not None:
            if 150 <= r.calories <= 1500:
                very_strict_recipes.append(r)
        else:
            # No kcal data, mirror Strict criteria
            very_strict_recipes.append(r)
    
    very_strict_coverage = len(very_strict_recipes) / total_recipes * 100
    
    # Print results
    print("\nCOVERAGE RESULTS")
    print("=" * 30)
    print(f"Baseline (cost>0 & match%>=80): {baseline_coverage:.1f}% ({len(baseline_recipes)}/{total_recipes})")
    print(f"Target: >=90% - {'PASS' if baseline_coverage >= 90 else 'FAIL'}")
    print()
    print(f"Strict (50<=cost<=2000 & match%>=80): {strict_coverage:.1f}% ({len(strict_recipes)}/{total_recipes})")  
    print(f"Target: >=85% - {'PASS' if strict_coverage >= 85 else 'FAIL'}")
    print()
    print(f"Very-strict (Strict + kcal filter): {very_strict_coverage:.1f}% ({len(very_strict_recipes)}/{total_recipes})")
    print(f"Target: >=70% - {'PASS' if very_strict_coverage >= 70 else 'FAIL'}")
    
    # Show sample results
    print(f"\nSAMPLE RESULTS (first 5)")
    print("=" * 40)
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result.recipe_title[:40]}...")
        print(f"   Cost: {result.total_cost_lkr:.2f} LKR")
        print(f"   Match: {result.match_percentage:.1f}%") 
        print(f"   Confidence: {'High' if not result.low_confidence else 'Low'}")
        print()
    
    # Cost statistics
    costs = [r.total_cost_lkr for r in results if r.total_cost_lkr > 0]
    if costs:
        print(f"COST STATISTICS")
        print("=" * 20)
        print(f"Median cost: {np.median(costs):.2f} LKR")
        print(f"P90 cost: {np.percentile(costs, 90):.2f} LKR")
        print(f"Mean match%: {np.mean([r.match_percentage for r in results]):.1f}%")
    
    # Export sample results
    results_data = []
    for r in results:
        results_data.append({
            'title': r.recipe_title,
            'cost_lkr': r.total_cost_lkr,
            'match_percentage': r.match_percentage,
            'matched_ingredients': r.matched_ingredients,
            'total_ingredients': r.total_ingredients,
            'low_confidence': r.low_confidence,
            'servings': r.servings,
            'is_main_dish': r.is_main_dish
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('test_pricing_results.csv', index=False)
    print(f"\nResults exported to: test_pricing_results.csv")
    
    return {
        'baseline_coverage': baseline_coverage,
        'strict_coverage': strict_coverage, 
        'very_strict_coverage': very_strict_coverage,
        'total_processed': total_recipes
    }

if __name__ == "__main__":
    test_pricing_coverage()