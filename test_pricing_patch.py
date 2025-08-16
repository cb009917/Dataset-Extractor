#!/usr/bin/env python3
"""
Test script for pricing stabilization patch
Verifies target coverage rates: Baseline ≥90%, Strict ≥85%, Very-strict ≥70%
"""

import sys
import os
import pandas as pd
from price_calculator import MultithreadedPriceCalculator

def test_pricing_patch():
    """Test the pricing stabilization patch implementation"""
    print("PRICING STABILIZATION PATCH TEST")
    print("="*60)
    print("Testing target coverage rates: Baseline >=90%, Strict >=85%, Very-strict >=70%")
    
    # Check if required files exist
    required_files = [
        'extracted_prices2.csv',
        'price_aliases.yaml', 
        'densities.yaml',
        'piece_weights.yaml',
        'unit_rules.yaml',
        'normalizer.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    print("All required configuration files found")
    
    # Use an existing recipe dataset
    recipe_files = [
        'recipes_normalized.csv',
        'dataset/authentic_sri_lankan_recipes_normalized.csv',
        'recipes.csv'
    ]
    
    recipe_file = None
    for file in recipe_files:
        if os.path.exists(file):
            recipe_file = file
            break
    
    if not recipe_file:
        print(f"No recipe dataset found. Looked for: {recipe_files}")
        return False
    
    print(f"Using recipe dataset: {recipe_file}")
    
    try:
        # Load recipe data
        recipes_df = pd.read_csv(recipe_file)
        print(f"Loaded {len(recipes_df)} recipes")
        
        # Initialize calculator
        calculator = MultithreadedPriceCalculator('extracted_prices2.csv')
        
        # Process recipes (use a smaller sample for testing)
        sample_size = min(500, len(recipes_df))  # Test with up to 500 recipes
        test_df = recipes_df.head(sample_size)
        print(f"Testing with {len(test_df)} recipes")
        
        # Run pricing calculation
        output_file = 'test_pricing_results.csv'
        print(f"Running pricing calculation...")
        
        output_df, results = calculator.process_recipes_batch_advanced(test_df, output_file)
        
        # Check coverage stats
        if hasattr(calculator, 'coverage_stats') and calculator.coverage_stats:
            stats = calculator.coverage_stats
            
            print("\nCOVERAGE RESULTS:")
            print("="*40)
            
            baseline_pct = stats['baseline_coverage_percent']
            strict_pct = stats['strict_coverage_percent']
            very_strict_pct = stats['very_strict_coverage_percent']
            
            baseline_pass = baseline_pct >= 90.0
            strict_pass = strict_pct >= 85.0
            very_strict_pass = very_strict_pct >= 70.0
            
            print(f"Baseline (>=90%):     {baseline_pct:.1f}% {'PASS' if baseline_pass else 'FAIL'}")
            print(f"Strict (>=85%):       {strict_pct:.1f}% {'PASS' if strict_pass else 'FAIL'}")
            print(f"Very-strict (>=70%):  {very_strict_pct:.1f}% {'PASS' if very_strict_pass else 'FAIL'}")
            
            print(f"\nCOST METRICS:")
            print(f"Median cost per person: Rs {stats.get('median_cost_per_person_lkr', 0):.2f}")
            print(f"P90 cost per person: Rs {stats.get('p90_cost_per_person_lkr', 0):.2f}")
            
            # Overall pass/fail
            all_targets_met = baseline_pass and strict_pass and very_strict_pass
            
            print(f"\nOVERALL RESULT: {'ALL TARGETS MET' if all_targets_met else 'TARGETS NOT MET'}")
            
            if all_targets_met:
                print("Pricing stabilization patch successfully achieves target coverage rates!")
            else:
                print("Some targets not met. Further optimization may be needed.")
            
            # Check output files
            output_files = [
                'test_pricing_results.csv',
                'pricing_coverage_report.json', 
                'unmatched_tokens.csv',
                'unit_warnings.csv',
                'missing_prices.csv',
                'cost_outliers.csv'
            ]
            
            print(f"\nOUTPUT FILES GENERATED:")
            for file in output_files:
                exists = os.path.exists(file)
                print(f"  {file}: {'YES' if exists else 'NO'}")
            
            return all_targets_met
            
        else:
            print("Coverage statistics not available")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pricing_patch()
    sys.exit(0 if success else 1)