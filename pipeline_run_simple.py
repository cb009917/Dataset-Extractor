#!/usr/bin/env python3
"""
FitFeast Pipeline Runner - Simple version for testing
Runs the pipeline without complex auto-iteration for now
"""

import os
import json
import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simple_pipeline():
    """Run a simple version of the pipeline for testing"""
    
    # Check if recipes_base.csv exists
    if not os.path.exists("recipes_base.csv"):
        logger.error("recipes_base.csv not found in current directory")
        return False
        
    logger.info("Starting FitFeast pipeline")
    
    try:
        # Step 1: Normalization
        logger.info("=== STEP 1: NORMALIZATION ===")
        from serving import IngredientNormalizer
        normalizer = IngredientNormalizer()
        df = normalizer.normalize_csv_file("recipes_base.csv", "recipes_base_normalized.csv")
        
        # Check metrics
        if os.path.exists("reports/normalization_metrics.json"):
            with open("reports/normalization_metrics.json", 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            logger.info(f"Normalization metrics: {metrics}")
        
        # Step 2: Nutrition
        logger.info("=== STEP 2: NUTRITION ===")
        from nutrition_calculator import NutritionCalculator
        calculator = NutritionCalculator()
        df = calculator.process_csv_file("recipes_base_normalized.csv", "recipes_base_normalized_improved_nutrition.csv")
        
        # Check nutrition coverage
        if os.path.exists("reports/nutrition_coverage_report.json"):
            with open("reports/nutrition_coverage_report.json", 'r', encoding='utf-8') as f:
                report = json.load(f)
            logger.info(f"Nutrition coverage: {report.get('coverage_pct', 0)}%")
        
        # Step 3: Pricing
        logger.info("=== STEP 3: PRICING ===")
        from price_calculator import PriceCalculator
        calculator = PriceCalculator()
        df = calculator.process_csv_file("recipes_base_normalized_improved_nutrition.csv", "recipes_base_normalized_improved_nutrition_with_costs.csv")
        
        # Check pricing coverage
        if os.path.exists("reports/pricing_coverage_report.json"):
            with open("reports/pricing_coverage_report.json", 'r', encoding='utf-8') as f:
                report = json.load(f)
            coverage = report.get('coverage_percentages', {})
            logger.info(f"Pricing coverage: Baseline={coverage.get('baseline', 0)}%, Strict={coverage.get('strict', 0)}%, VeryStrict={coverage.get('very_strict', 0)}%")
        
        logger.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_pipeline()
    sys.exit(0 if success else 1)