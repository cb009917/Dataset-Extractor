#!/usr/bin/env python3
"""
Unit tests for normalizer.py functions
Tests the specifications from the requirements
"""

import unittest
from normalizer import normalize_token, normalize_ingredient_list, extract_quantity_and_unit

class TestNormalizer(unittest.TestCase):
    """Test normalizer functions according to specifications"""
    
    def test_normalize_token_exact_specs(self):
        """Test normalize_token with exact specification cases"""
        
        # Test cases from specification (THESE MUST PASS)
        test_cases = [
            # Basic normalization
            ("cooking oil - as you need", (None, {"skip_denominator": True})),  # Should skip
            ("/4 sprig of curry l eaves", ("1/4 sprig of curry leaves", {"skip_denominator": False})),
            ("1/2 tsp l ime juice", ("1/2 tsp lime juice", {"skip_denominator": False})),
            ("/3 l bbeef", ("1/3 lb beef", {"skip_denominator": False})),
            ("sfresh milk 1 cup", ("fresh milk 1 cup", {"skip_denominator": False})),
            ("bpork shoulder 1 lb", ("pork shoulder 1 lb", {"skip_denominator": False})),
            ("1/2 tsp freshly g round black pepper", ("1/2 tsp freshly ground black pepper", {"skip_denominator": False})),
            ("red chili 5 g", ("red chili 5 g", {"skip_denominator": False})),
            ("wheat flour 100 g", ("wheat flour 100 g", {"skip_denominator": False})),
            ("broth/stock 500 ml", ("broth/stock 500 ml", {"skip_denominator": False})),
            
            # Split letter fixes
            ("g arlic", ("garlic", {"skip_denominator": False})),
            ("l ime", ("lime", {"skip_denominator": False})),
            ("g round", ("ground", {"skip_denominator": False})),
            ("c urry l eaves", ("curry leaves", {"skip_denominator": False})),
            ("l b", ("lb", {"skip_denominator": False})),
            
            # De-glue units
            ("2ggarlic", ("2g garlic", {"skip_denominator": False})),
            ("tbspoil", ("tbsp oil", {"skip_denominator": False})),
            
            # Fraction repair
            ("/4 tsp", ("1/4 tsp", {"skip_denominator": False})),
            ("1 / 2", ("1/2", {"skip_denominator": False})),
            
            # Filler exclusions
            ("Salt to taste", (None, {"skip_denominator": True})),
            ("as you need", (None, {"skip_denominator": True})),
            ("adjust to your taste", (None, {"skip_denominator": True})),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = normalize_token(input_text)
                self.assertEqual(result, expected, f"Failed for input: '{input_text}'")
    
    def test_extract_quantity_and_unit(self):
        """Test quantity and unit extraction with safety checks"""
        
        test_cases = [
            # Basic extractions
            ("2 g garlic", (2.0, "g", "garlic")),
            ("1/2 tsp salt", (0.5, "tsp", "salt")),
            ("1 lb beef", (1.0, "lb", "beef")),
            ("1/4 cup milk", (0.25, "cup", "milk")),
            
            # Piece-unit clamp: if qty < 1 for piece nouns → set qty = 1
            ("1/4 clove garlic", (1.0, "clove", "garlic")),  # Should clamp to 1.0
            ("0.5 egg", (1.0, "egg", "")),  # Should clamp to 1.0
            ("1/3 sprig curry leaves", (1.0, "sprig", "curry leaves")),  # Should clamp to 1.0
            
            # No quantity found
            ("garlic", (None, "", "garlic")),
            ("fresh milk", (None, "", "fresh milk")),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = extract_quantity_and_unit(input_text)
                self.assertEqual(result, expected, f"Failed for input: '{input_text}'")
    
    def test_normalize_ingredient_list(self):
        """Test pipe-delimited ingredient list normalization"""
        
        # Test pipe splitting and normalization
        input_text = "g arlic 2 cloves|cooking oil - as you need|1/2 tsp l ime juice"
        result = normalize_ingredient_list(input_text)
        
        # Should normalize tokens and exclude fillers
        expected = [
            "garlic 2 cloves",  # Fixed split letter
            "1/2 tsp lime juice"  # Fixed split letter and fraction
            # "cooking oil - as you need" should be excluded
        ]
        
        self.assertEqual(result, expected)
    
    def test_pounds_normalization(self):
        """Test pounds token normalization"""
        
        test_cases = [
            ("1 lb beef", ("1 lb beef", {"skip_denominator": False})),
            ("2 lbs chicken", ("2 lb chicken", {"skip_denominator": False})),
            ("1.5 lb.", ("1.5 lb", {"skip_denominator": False})),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = normalize_token(input_text)
                self.assertEqual(result, expected, f"Failed for input: '{input_text}'")

if __name__ == "__main__":
    # Run the unit tests
    unittest.main(verbosity=2)