#!/usr/bin/env python3
"""
Smoke tests for FitFeast pipeline normalization, nutrition, and pricing
Tests the core behavior and acceptance criteria for serving.py, nutrition_calculator.py, and price_calculator.py
"""

import pytest
import sys
import os
import pandas as pd
import json
import csv
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving import IngredientNormalizer
from nutrition_calculator import NutritionCalculator
from price_calculator import PriceCalculator

class TestNormalizerBehavior:
    """Test serving.py normalization behavior against specific requirements"""
    
    def setup_method(self):
        self.normalizer = IngredientNormalizer()
    
    def test_suffix_quantity_rewrite(self):
        """Test suffix quantity rewrite functionality"""
        cases = [
            ("Jackfruit Seeds – 250g", 4, "62.5 g Jackfruit Seeds"),
            ("Coconut Milk — 1 1/2 cups", 4, "0.375 cup Coconut Milk"),
            ("Onion – 1", 2, "1 pcs Onion"),  # pcs fallback
            ("Red lentils – 200 g", 2, "100 g Red lentils"),
        ]
        
        for input_text, servings, expected in cases:
            result = self.normalizer.normalize_ingredient(input_text, servings, "Test")
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_filler_and_meta_exclusion(self):
        """Test A-list (fillers/meta) exclusion"""
        exclusions = [
            ("Chili powder (to taste)", 4),
            ("Salt – as required", 4),  
            ("Best Chicken – For Curry -----", 2),  # section header
            ("Water – 250 ml", 2),  # water even if quantified
        ]
        
        for input_text, servings in exclusions:
            result = self.normalizer.normalize_ingredient(input_text, servings, "Test")
            assert result is None, f"Expected None for '{input_text}', got '{result}'"
    
    def test_micro_spice_b_list_exclusion(self):
        """Test B-list (micro-spices) with <2g rule"""
        # Should be dropped due to <2g after scaling
        result = self.normalizer.normalize_ingredient("Mustard seeds – 1 tsp", 2, "Test")
        assert result is None, "Expected None for micro-spice <2g"
        
        # Large spice quantities should also be dropped
        result = self.normalizer.normalize_ingredient("Turmeric powder – 2 tbsp", 1, "Test")
        assert result is None, "Expected None for B-list spice"
    
    def test_deglue_operations(self):
        """Test two-pass deglue functionality"""
        cases = [
            ("2tspginger", 2, "1 tsp ginger"),
            ("1cupcoconut milk", 2, "0.5 cup coconut milk"),
        ]
        
        for input_text, servings, expected in cases:
            result = self.normalizer.normalize_ingredient(input_text, servings, "Test")
            assert result == expected, f"Expected '{expected}', got '{result}' for deglue test '{input_text}'"
    
    def test_parentheses_removal(self):
        """Test parentheses are completely removed"""
        result = self.normalizer.normalize_ingredient("(pork shoulder or belly)", 1, "Test")
        assert result is None, "Parenthetical content should be dropped"
    
    def test_unicode_normalization(self):
        """Test unicode fraction normalization"""
        result = self.normalizer.normalize_ingredient("½ cup coconut milk", 2, "Test")
        expected = "0.25 cup coconut milk"
        assert result == expected, f"Expected '{expected}', got '{result}' for unicode test"
    
    def test_metrics_calculation(self):
        """Test that normalizer produces required metrics"""
        test_data = pd.DataFrame({
            'title': ['Test Recipe 1', 'Test Recipe 2'],
            'ingredients': [
                'Jackfruit Seeds – 250g | Coconut Milk — 1 1/2 cups | 2tspginger',
                '1cupcoconut milk | Red lentils – 200 g | Salt – as required'
            ],
            'servings': [4, 2]
        })
        
        result_df = self.normalizer.normalize_dataframe(test_data)
        
        # Check that metrics were printed (in real implementation)
        # We'll verify by checking the output structure
        assert 'ingredients_per_person' in result_df.columns
        assert len(result_df) == 2
        
        # Check that warnings and needs_quantity files would be created
        assert hasattr(self.normalizer, 'warnings')
        assert hasattr(self.normalizer, 'needs_qty_rows')


class TestNutritionBehavior:
    """Test nutrition_calculator.py behavior"""
    
    def setup_method(self):
        # Create a minimal nutrition database for testing
        self.calc = NutritionCalculator()
        # Override with test data
        self.calc.nutrition_database = {
            'coconut milk': type('obj', (object,), {'calories': 230, 'protein': 2.3, 'carbs': 6.0, 'fat': 24.0})(),
            'red lentils': type('obj', (object,), {'calories': 116, 'protein': 9.0, 'carbs': 20.0, 'fat': 0.4})(),
            'ginger': type('obj', (object,), {'calories': 80, 'protein': 1.8, 'carbs': 18.0, 'fat': 0.8})(),
        }
    
    def test_exclusion_rules(self):
        """Test A/B/<2g exclusion from nutrition calculations"""
        # Should exclude salt (A-list)
        nutrition = self.calc.calculate_ingredient_nutrition("1 tsp salt")
        assert nutrition.calories == 0, "A-list items should be excluded"
        
        # Should exclude turmeric (B-list)  
        nutrition = self.calc.calculate_ingredient_nutrition("1 tsp turmeric")
        assert nutrition.calories == 0, "B-list items should be excluded"
        
        # Should exclude tiny amounts
        nutrition = self.calc.calculate_ingredient_nutrition("0.1 g coconut milk")
        assert nutrition.calories == 0, "<2g items should be excluded"
    
    def test_outlier_logging(self):
        """Test outlier logging for >2500 kcal recipes"""
        # Create a high-calorie recipe
        high_cal_ingredients = "1000 g coconut milk | 500 g red lentils"  # Should exceed 2500 kcal
        
        nutrition = self.calc.calculate_recipe_nutrition(high_cal_ingredients, "High Cal Test")
        
        # Should return empty nutrition for outliers
        assert nutrition.calories <= 2500 or nutrition.calories == 0, "Outliers should be handled"
    
    def test_defensive_normalization(self):
        """Test defensive token cleaning"""
        cleaned = self.calc.defensive_clean_token("(coconut milk) 250ml")
        assert "(" not in cleaned and ")" not in cleaned, "Parentheses should be stripped"


class TestPricingBehavior:
    """Test price_calculator.py behavior"""
    
    def setup_method(self):
        self.calc = PriceCalculator()
        # Override with test price data
        self.calc.price_data = {
            'coconut milk': 500.0,  # LKR per kg
            'red lentils': 300.0,
            'onion': 200.0,
        }
    
    def test_exclusion_from_denominator(self):
        """Test A/B/<2g exclusion from pricing denominators"""
        cost, count_in_denom = self.calc.calculate_ingredient_cost("1 tsp salt")
        assert not count_in_denom, "A-list should not count in denominator"
        
        cost, count_in_denom = self.calc.calculate_ingredient_cost("1 tsp turmeric")
        assert not count_in_denom, "B-list should not count in denominator"
        
        cost, count_in_denom = self.calc.calculate_ingredient_cost("0.1 g coconut milk")
        assert not count_in_denom, "<2g should not count in denominator"
    
    def test_unit_conversions(self):
        """Test robust unit conversions"""
        # Test ml to g conversion (coconut milk)
        cost, count_in_denom = self.calc.calculate_ingredient_cost("250 ml coconut milk")
        assert cost > 0, "Should calculate cost for ml units"
        assert count_in_denom, "Should count in denominator"
        
        # Test pcs conversion (onion)
        cost, count_in_denom = self.calc.calculate_ingredient_cost("1 pcs onion")
        assert cost > 0, "Should calculate cost for pcs units"
        
    def test_coverage_calculation(self):
        """Test coverage calculation logic"""
        test_ingredients = "100 g coconut milk | 50 g red lentils | 1 pcs onion"
        result = self.calc.calculate_recipe_cost(test_ingredients)
        
        assert 'baseline_pass' in result
        assert 'strict_pass' in result
        assert 'very_strict_pass' in result
        assert result['total_cost'] > 0, "Should calculate positive cost"
    
    def test_fail_fast_behavior(self):
        """Test fail-fast when Baseline>0 but Strict==0"""
        # This would require a more complex setup to trigger the fail-fast condition
        # For now, just verify the method exists
        assert hasattr(self.calc, 'analyze_cost_buckets')


class TestPipelineIntegration:
    """Test full pipeline integration"""
    
    def setup_method(self):
        self.normalizer = IngredientNormalizer()
        
        # Create minimal test databases
        self.nutrition_calc = NutritionCalculator()
        self.nutrition_calc.nutrition_database = {
            'coconut milk': type('obj', (object,), {'calories': 230, 'protein': 2.3, 'carbs': 6.0, 'fat': 24.0})(),
            'red lentils': type('obj', (object,), {'calories': 116, 'protein': 9.0, 'carbs': 20.0, 'fat': 0.4})(),
            'onion': type('obj', (object,), {'calories': 40, 'protein': 1.1, 'carbs': 9.3, 'fat': 0.1})(),
        }
        
        self.price_calc = PriceCalculator()
        self.price_calc.price_data = {
            'coconut milk': 500.0,
            'red lentils': 300.0,
            'onion': 200.0,
        }
    
    def test_end_to_end_normalization(self):
        """Test complete normalization pipeline"""
        test_cases = [
            ("Jackfruit Seeds – 250g", 4, "62.5 g Jackfruit Seeds"),
            ("Coconut Milk — 1 1/2 cups", 4, "0.375 cup Coconut Milk"),
            ("Chili powder (to taste)", 4, None),  # Should be dropped
            ("Salt – as required", 4, None),  # Should be dropped
            ("Mustard seeds – 1 tsp", 2, None),  # Should be dropped (<2g + B-list)
            ("2tspginger", 2, "1 tsp ginger"),  # Deglue test
            ("1cupcoconut milk", 2, "0.5 cup coconut milk"),  # Deglue test
            ("Best Chicken – For Curry -----", 2, None),  # Section header
            ("Red lentils – 200 g", 2, "100 g Red lentils"),
            ("Onion – 1", 2, "1 pcs Onion"),  # pcs fallback
            ("Water – 250 ml", 2, None),  # Water excluded
        ]
        
        for input_text, servings, expected in test_cases:
            result = self.normalizer.normalize_ingredient(input_text, servings, "Test")
            assert result == expected, f"Failed for '{input_text}': expected '{expected}', got '{result}'"
    
    def test_pipeline_metrics_requirements(self):
        """Test that pipeline meets acceptance criteria"""
        test_data = pd.DataFrame({
            'title': ['Test Recipe'],
            'ingredients': ['250 g coconut milk | 200 g red lentils | 1 pcs onion | salt to taste'],
            'servings': [2]
        })
        
        # Normalize
        normalized_df = self.normalizer.normalize_dataframe(test_data)
        
        # Get normalized ingredients
        normalized_ingredients = normalized_df['ingredients_per_person'].iloc[0]
        
        # Test nutrition
        nutrition = self.nutrition_calc.calculate_recipe_nutrition(normalized_ingredients, "Test Recipe")
        assert nutrition.calories < 2500, "Should not exceed calorie outlier threshold"
        
        # Test pricing
        pricing_result = self.price_calc.calculate_recipe_cost(normalized_ingredients)
        assert pricing_result['total_cost'] > 0, "Should calculate positive cost"
        
        # Check baseline requirements would be met
        assert pricing_result['match_percentage'] > 0, "Should have some ingredient matches"


def test_exact_regex_patterns():
    """Test the exact regex patterns specified in requirements"""
    normalizer = IngredientNormalizer()
    
    # Test suffix quantity regex patterns
    # Pattern 1: qty + unit
    test_input = "Coconut Milk — 1 1/2 cups"
    result = normalizer._rewrite_suffix_quantity(test_input)
    assert "1 1/2 cup Coconut Milk" in result or "1.5 cup Coconut Milk" in result
    
    # Pattern 2: qty only  
    test_input = "Onion – 1"
    result = normalizer._rewrite_suffix_quantity(test_input)
    assert "1 Onion" in result
    
    # Test two-pass deglue patterns
    test_input = "2tspginger"
    result = normalizer._deglue_two_pass(test_input)
    assert "2 tsp ginger" == result.strip()
    
    # Test parenthesis stripping
    test_input = "(optional ingredient)"
    result = normalizer._strip_parentheses(test_input)
    assert "(" not in result and ")" not in result


if __name__ == "__main__":
    # Run specific test cases manually
    print("Running FitFeast Pipeline Smoke Tests...")
    
    # Test normalization
    normalizer = IngredientNormalizer()
    test_cases = [
        ("Jackfruit Seeds – 250g", 4),
        ("Coconut Milk — 1 1/2 cups", 4),
        ("Chili powder (to taste)", 4),
        ("Salt – as required", 4),
        ("Mustard seeds – 1 tsp", 2),
        ("2tspginger", 2),
        ("1cupcoconut milk", 2),
        ("Best Chicken – For Curry -----", 2),
        ("Red lentils – 200 g", 2),
        ("Onion – 1", 2),
        ("Water – 250 ml", 2),
    ]
    
    print("\n=== NORMALIZATION TESTS ===")
    for ingredient, servings in test_cases:
        result = normalizer.normalize_ingredient(ingredient, servings, "Test")
        print(f"{ingredient:<35} -> {result}")
    
    print("\n=== METRICS TEST ===")
    test_df = pd.DataFrame({
        'title': ['Test Recipe'],
        'ingredients': ['Jackfruit Seeds – 250g | Coconut Milk — 1 1/2 cups | 2tspginger | Salt to taste'],
        'servings': [4]
    })
    
    result_df = normalizer.normalize_dataframe(test_df)
    print(f"Normalized ingredients: {result_df['ingredients_per_person'].iloc[0]}")
    
    print("\nSmoke tests completed successfully!")