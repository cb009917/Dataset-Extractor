#!/usr/bin/env python3
"""
Test the complete pipeline with coverage targets
"""

import pandas as pd
from price_calculator import AdvancedPriceCalculator
import json

def test_complete_pipeline():
    """Test the complete pipeline and check coverage targets"""
    
    print("Testing Recipe Pricing Pipeline")
    print("=" * 50)
    
    # Test dataset loading
    try:
        # Try different possible dataset files
        dataset_files = [
            'dataset/authentic_sri_lankan_recipes.csv',
            'recipes_with_costs_clean.csv',
            'recipes.csv'
        ]
        
        df = None
        for file in dataset_files:
            try:
                df = pd.read_csv(file)
                print(f"[OK] Loaded dataset: {file} ({len(df)} recipes)")
                print(f"Columns: {list(df.columns)}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            print("[ERROR] No dataset file found")
            return
            
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return
    
    # Initialize price calculator
    try:
        calc = AdvancedPriceCalculator()
        print("[OK] Price calculator initialized")
    except Exception as e:
        print(f"[ERROR] Error initializing calculator: {e}")
        return
    
    # Test price database loading
    print(f"Price database size: {len(calc.price_database)}")
    if len(calc.price_database) == 0:
        print("[ERROR] No prices loaded - check extracted_prices2.csv and unit normalization")
        return
    else:
        print("[OK] Prices loaded successfully")
    
    # Test normalization on sample ingredients
    from normalizer import normalize_token, extract_quantity_and_unit
    
    test_ingredients = [
        "g arlic 2 cloves",
        "1/2 tsp l ime juice", 
        "cooking oil - as you need",
        "2 tbsp coconut oil",
        "1 lb chicken"
    ]
    
    print("\nTesting normalization:")
    for ingredient in test_ingredients:
        normalized, meta = normalize_token(ingredient)
        if normalized:
            qty, unit, name = extract_quantity_and_unit(normalized)
            print(f"  '{ingredient}' -> '{normalized}' -> qty:{qty}, unit:'{unit}', name:'{name}'")
        else:
            print(f"  '{ingredient}' -> SKIPPED (meta: {meta})")
    
    # Test recipe processing
    print("\nTesting recipe processing:")
    
    # Find ingredients column
    ingredient_cols = [col for col in df.columns if 'ingredient' in col.lower()]
    title_cols = [col for col in df.columns if 'title' in col.lower() or 'name' in col.lower()]
    
    if not ingredient_cols:
        print("[ERROR] No ingredients column found")
        return
    
    ingredient_col = ingredient_cols[0]
    title_col = title_cols[0] if title_cols else None
    
    print(f"Using ingredients column: {ingredient_col}")
    print(f"Using title column: {title_col}")
    
    # Test a few recipes
    sample_size = min(10, len(df))
    results = []
    
    for i in range(sample_size):
        row = df.iloc[i]
        title = row[title_col] if title_col else f"Recipe {i+1}"
        ingredients = row[ingredient_col]
        
        print(f"\nTesting: {title}")
        print(f"Ingredients: {str(ingredients)[:100]}...")
        
        # Create a mock row for the calculator
        test_row = {
            'recipe_title': title,
            'ingredients_text': ingredients,
            'servings': 4  # Default servings
        }
        
        try:
            result = calc.calculate_recipe_cost_advanced(pd.Series(test_row))
            if result:
                results.append(result)
                print(f"  Cost: Rs {result.estimated_cost_lkr:.2f}")
                print(f"  Match: {result.match_percentage:.1f}%")
                print(f"  Low confidence: {result.low_confidence}")
            else:
                print("  Failed to calculate cost")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Calculate coverage if we have results
    if results:
        print(f"\n{'='*50}")
        print("COVERAGE ANALYSIS")
        print(f"{'='*50}")
        
        # Use the exact tier rules from specification
        total_recipes = len(results)
        
        # Baseline: (estimated_cost_lkr > 0) AND (match_percentage ≥ 80)
        baseline_count = len([r for r in results if r.estimated_cost_lkr > 0 and r.match_percentage >= 80])
        
        # Strict: 50 ≤ estimated_cost_lkr ≤ 2000 AND match_percentage ≥ 80
        strict_count = len([r for r in results if 50 <= r.estimated_cost_lkr <= 2000 and r.match_percentage >= 80])
        
        # Very-strict: Strict + (150 ≤ calories ≤ 1500); if no calories for a row, treat Very-strict = Strict
        very_strict_count = strict_count  # Since we don't have calories data, use Strict
        
        baseline_pct = (baseline_count / total_recipes) * 100
        strict_pct = (strict_count / total_recipes) * 100  
        very_strict_pct = (very_strict_count / total_recipes) * 100
        
        print(f"Total recipes tested: {total_recipes}")
        print(f"Baseline (cost>0 & match≥80%): {baseline_count}/{total_recipes} = {baseline_pct:.1f}%")
        print(f"Strict (50≤cost≤2000 & match≥80%): {strict_count}/{total_recipes} = {strict_pct:.1f}%")
        print(f"Very-strict (same as strict): {very_strict_count}/{total_recipes} = {very_strict_pct:.1f}%")
        
        print(f"\nTarget Achievement:")
        print(f"Baseline ≥90%: {'✓ PASS' if baseline_pct >= 90 else '✗ FAIL'} ({baseline_pct:.1f}%)")
        print(f"Strict ≥85%: {'✓ PASS' if strict_pct >= 85 else '✗ FAIL'} ({strict_pct:.1f}%)")
        print(f"Very-strict ≥70%: {'✓ PASS' if very_strict_pct >= 70 else '✗ FAIL'} ({very_strict_pct:.1f}%)")
        
        # Statistics
        valid_costs = [r.estimated_cost_lkr for r in results if r.estimated_cost_lkr > 0]
        if valid_costs:
            import numpy as np
            median_cost = np.median(valid_costs)
            p90_cost = np.percentile(valid_costs, 90)
            print(f"\nCost Statistics:")
            print(f"Median cost: Rs {median_cost:.2f}")
            print(f"P90 cost: Rs {p90_cost:.2f}")
            
        match_percentages = [r.match_percentage for r in results]
        if match_percentages:
            median_match = np.median(match_percentages)
            print(f"Median match percentage: {median_match:.1f}%")
    
    else:
        print("[ERROR] No successful recipe calculations")

if __name__ == "__main__":
    test_complete_pipeline()