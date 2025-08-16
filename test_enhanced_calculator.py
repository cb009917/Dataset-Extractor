#!/usr/bin/env python3
"""
Test script for the enhanced price calculator with new features
"""

import pandas as pd
import logging
from price_calculator import MultithreadedPriceCalculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_calculator():
    """Test the enhanced price calculator with sample data"""
    
    # Create a small test dataset
    test_data = {
        'title': [
            'Coconut Rice',
            'Garlic Chicken Curry', 
            'Spicy Lentil Soup',
            'Simple Water Recipe'
        ],
        'ingredients': [
            '2 cups rice|1 cup coconut milk|1 tsp salt|2 tbsp cooking oil',
            '1 kg chicken|4 garlic cloves|2 tbsp curry powder|1 cup coconut milk|salt to taste',
            '1 cup lentils|1 onion|2 green chili|1/4 tsp turmeric|water as needed',
            '2 cups water|pinch of salt'
        ],
        'servings': [4, 6, 4, 2],
        'calories': [350, 450, 180, 5]
    }
    
    df = pd.DataFrame(test_data)
    print("Test dataset:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    # Initialize the calculator
    try:
        calculator = MultithreadedPriceCalculator(max_workers=2)
        print("Calculator initialized successfully")
        
        # Process the recipes
        output_df, results = calculator.process_recipes_batch_advanced(df, 'test_results.csv')
        
        print("\nResults:")
        print(output_df[['title', 'estimated_cost_lkr', 'match_percentage', 'matched_ingredients', 'total_ingredients']])
        
        print("\nCoverage Statistics:")
        # Calculate basic coverage stats
        total_recipes = len(results)
        baseline_count = sum(1 for r in results if r.estimated_cost_lkr > 0 and r.match_percentage >= 80)
        strict_count = sum(1 for r in results if 50 <= r.estimated_cost_lkr <= 2000 and r.match_percentage >= 80)
        
        print(f"Total recipes: {total_recipes}")
        print(f"Baseline (cost>0 & match%>=80): {baseline_count}/{total_recipes} ({100*baseline_count/total_recipes:.1f}%)")
        print(f"Strict (50<=cost<=2000 & match%>=80): {strict_count}/{total_recipes} ({100*strict_count/total_recipes:.1f}%)")
        
    except Exception as e:
        print(f"Error testing calculator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_calculator()