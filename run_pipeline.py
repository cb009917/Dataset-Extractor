#!/usr/bin/env python3
"""
Run the complete recipe pricing pipeline
"""

import pandas as pd
from price_calculator import MultithreadedPriceCalculator
import sys

def run_pipeline():
    """Run the complete pipeline"""
    
    print("ADVANCED RECIPE PRICE CALCULATOR - 90%+ REALISTIC COVERAGE")
    print("="*65)
    print("Enhanced with normalization, aliasing, densities, and guardrails")
    print()
    
    # Use dataset file
    recipe_file = "dataset/authentic_sri_lankan_recipes.csv"
    
    try:
        # Load recipe data
        print(f"Loading recipes from: {recipe_file}")
        recipes_df = pd.read_csv(recipe_file)
        print(f"Loaded {len(recipes_df)} recipes")
        
        # Check columns
        print(f"Columns: {list(recipes_df.columns)}")
        
        # Initialize calculator with multithreading
        print("\nInitializing price calculator...")
        calculator = MultithreadedPriceCalculator(max_workers=8)
        
        # Process all recipes
        print(f"\nProcessing {len(recipes_df)} recipes...")
        results = []
        
        for idx, row in recipes_df.iterrows():
            try:
                result = calculator.calculate_recipe_cost_advanced(row)
                if result:
                    results.append(result)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(recipes_df)} recipes...")
                    
            except Exception as e:
                print(f"Error processing recipe {idx}: {e}")
                continue
        
        print(f"\nCompleted processing. {len(results)} recipes calculated successfully.")
        
        # Export results
        print("\nExporting results...")
        calculator.export_results_and_coverage(results)
        
        # Calculate and print coverage stats
        coverage_stats = calculator.calculate_and_export_coverage_stats(results)
        
        print("\n" + "="*60)
        print("COVERAGE RESULTS")
        print("="*60)
        
        print(f"Total recipes: {coverage_stats['total_recipes']}")
        print(f"Baseline coverage: {coverage_stats['baseline_coverage_percent']}% (target: >=90%)")
        print(f"Strict coverage: {coverage_stats['strict_coverage_percent']}% (target: >=85%)")
        print(f"Very-strict coverage: {coverage_stats['very_strict_coverage_percent']}% (target: >=70%)")
        
        print(f"\nCost statistics:")
        print(f"Median cost: Rs {coverage_stats['median_cost_per_person_lkr']}")
        print(f"P90 cost: Rs {coverage_stats['p90_cost_per_person_lkr']}")
        
        # Check targets
        targets = coverage_stats['targets_met']
        print(f"\nTarget achievement:")
        print(f"Baseline >=90%: {'PASS' if targets['baseline_target_90_percent'] else 'FAIL'}")
        print(f"Strict >=85%: {'PASS' if targets['strict_target_85_percent'] else 'FAIL'}")
        print(f"Very-strict >=70%: {'PASS' if targets['very_strict_target_70_percent'] else 'FAIL'}")
        
        print(f"\nFiles generated:")
        print(f"- recipes_with_costs_clean.csv")
        print(f"- pricing_coverage_report.json")
        print(f"- unmatched_tokens.csv")
        print(f"- unit_warnings.csv")
        print(f"- missing_prices.csv")
        print(f"- cost_outliers.csv")
        
        return True
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)