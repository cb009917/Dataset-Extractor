#!/usr/bin/env python3
"""
FitFeast Pipeline Runner - Auto-iterating end-to-end data processing pipeline
Runs serving.py nutrition_calculator.py price_calculator.py
Auto-heals by adding aliases and re-running until acceptance gates pass
"""

import os
import json
import pandas as pd
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.max_iterations = 8
        self.current_iteration = 0
        self.changelog_path = "reports/changelog.txt"

        # Acceptance gate thresholds
        self.normalization_gates = {
        'pct_digit_or_unit_min': 92.0,
        'glued_count_max': 0,
        'parentheses_count_max': 0
        }
        self.nutrition_gates = {
        'coverage_min': 90.0
        }
        self.pricing_gates = {
        'baseline_min': 98.0,
        'strict_min': 80.0,
        'very_strict_min': 70.0
        }

    def ensure_directories(self):
        """Ensure config/ and reports/ directories exist"""
        os.makedirs("config", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        # Create empty alias files if missing
        nutrition_alias_path = "config/nutrition_alias_user.json"
        item_alias_path = "config/item_alias_user.json"

        if not os.path.exists(nutrition_alias_path):
        with open(nutrition_alias_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=2)
        logger.info(f"Created empty {nutrition_alias_path}")

        if not os.path.exists(item_alias_path):
        with open(item_alias_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=2)
        logger.info(f"Created empty {item_alias_path}")

    def log_change(self, message: str):
        """Log changes to changelog.txt"""
        with open(self.changelog_path, 'a', encoding='utf-8') as f:
        f.write(f"Round {self.current_iteration}: {message}\n")

    def run_serving(self, input_file: str = "recipes_base.csv") -> Tuple[bool, str]:
        """Run serving.py normalization"""
        logger.info("Running serving.py normalization...")

        try:
        # Import and run serving
        from serving import IngredientNormalizer
        normalizer = IngredientNormalizer()
        df = normalizer.normalize_csv_file(input_file, "recipes_base_normalized.csv")

        # The file was successfully created
        output_file = "recipes_base_normalized.csv"

        # Check if metrics file exists and load it
        metrics_path = "reports/normalization_metrics.json"
        if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

        # Check gates
        gates_passed = (
        metrics['pct_digit_or_unit'] >= self.normalization_gates['pct_digit_or_unit_min'] and
        metrics['glued_count'] <= self.normalization_gates['glued_count_max'] and
        metrics['parentheses_count'] <= self.normalization_gates['parentheses_count_max']
        )

        logger.info(f"Normalization metrics: {metrics}")
        logger.info(f"Normalization gates passed: {gates_passed}")

        return gates_passed, output_file
        else:
        logger.warning("Normalization metrics file not found")
        return False, output_file

        except Exception as e:
        logger.error(f"Error running serving.py: {e}")
        return False, ""

    def post_fix_normalization(self, csv_file: str) -> str:
        """Apply post-fix to ingredients_per_person for glued units and parentheses"""
        logger.info("Applying post-fix to normalization...")

        try:
        df = pd.read_csv(csv_file, encoding='utf-8')

        if 'ingredients_per_person' not in df.columns:
        logger.warning("No ingredients_per_person column found for post-fix")
        return csv_file

        # Apply fixes to each token
        fixed_count = 0
        for idx, row in df.iterrows():
        ingredients = row.get('ingredients_per_person', '')
        if not ingredients:
        continue

        tokens = [t.strip() for t in str(ingredients).split('|') if t.strip()]
        fixed_tokens = []

        for token in tokens:
        # Remove parentheses
        fixed_token = token
        while '(' in fixed_token or ')' in fixed_token:
        fixed_token = fixed_token.replace('(', '').replace(')', '')

        # De-glue common patterns
        import re
        fixed_token = re.sub(r'(\d+(?:[.,]\d+)?)(tsp|tbsp|cup|g|kg|ml|l|oz|lb|pcs)([a-zA-Z])',
        r'\1 \2 \3', fixed_token, flags=re.IGNORECASE)
        fixed_token = re.sub(r'\b(tsp|tbsp|cup|g|kg|ml|l|oz|lb|pcs)(?=[A-Za-z])',
        r'\1 ', fixed_token, flags=re.IGNORECASE)
        fixed_token = re.sub(r'\s+', ' ', fixed_token).strip()

        if fixed_token != token:
        fixed_count += 1

        fixed_tokens.append(fixed_token)

        df.at[idx, 'ingredients_per_person'] = ' | '.join(fixed_tokens)

        # Save fixed version
        fixed_file = csv_file.replace('.csv', '_fixed.csv')
        df.to_csv(fixed_file, index=False, encoding='utf-8')
        logger.info(f"Applied {fixed_count} post-fixes, saved to {fixed_file}")

        return fixed_file

        except Exception as e:
        logger.error(f"Error in post-fix: {e}")
        return csv_file

    def run_nutrition(self, input_file: str) -> Tuple[bool, str]:
        """Run nutrition_calculator.py"""
        logger.info("Running nutrition_calculator.py...")

        try:
        from nutrition_calculator import NutritionCalculator
        calculator = NutritionCalculator()
        output_file = input_file.replace('.csv', '_improved_nutrition.csv')
        df = calculator.process_csv_file(input_file, output_file)

        # Check coverage report
        coverage_path = "reports/nutrition_coverage_report.json"
        if os.path.exists(coverage_path):
        with open(coverage_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

        coverage = report.get('coverage_pct', 0)
        gates_passed = coverage >= self.nutrition_gates['coverage_min']

        logger.info(f"Nutrition coverage: {coverage}%")
        logger.info(f"Nutrition gates passed: {gates_passed}")

        return gates_passed, output_file
        else:
        logger.warning("Nutrition coverage report not found")
        return False, output_file

        except Exception as e:
        logger.error(f"Error running nutrition_calculator.py: {e}")
        return False, ""

    def run_pricing(self, input_file: str) -> Tuple[bool, str]:
        """Run price_calculator.py"""
        logger.info("Running price_calculator.py...")

        try:
        from price_calculator import PriceCalculator
        calculator = PriceCalculator()
        output_file = input_file.replace('.csv', '_with_costs.csv')
        df = calculator.process_csv_file(input_file, output_file)

        # Check coverage report
        coverage_path = "reports/pricing_coverage_report.json"
        if os.path.exists(coverage_path):
        with open(coverage_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

        baseline = report.get('coverage_percentages', {}).get('baseline', 0)
        strict = report.get('coverage_percentages', {}).get('strict', 0)
        very_strict = report.get('coverage_percentages', {}).get('very_strict', 0)

        gates_passed = (
        baseline >= self.pricing_gates['baseline_min'] and
        strict >= self.pricing_gates['strict_min'] and
        very_strict >= self.pricing_gates['very_strict_min']
        )

        logger.info(f"Pricing coverage: Baseline={baseline}%, Strict={strict}%, VeryStrict={very_strict}%")
        logger.info(f"Pricing gates passed: {gates_passed}")

        return gates_passed, output_file
        else:
        logger.warning("Pricing coverage report not found")
        return False, output_file

        except Exception as e:
        logger.error(f"Error running price_calculator.py: {e}")
        return False, ""

    def add_nutrition_aliases(self, count: int = 20) -> int:
        """Add high-confidence nutrition aliases from unmatched report"""
        unmatched_path = "reports/unmatched_nutrition.csv"
        alias_path = "config/nutrition_alias_user.json"

        if not os.path.exists(unmatched_path):
        logger.warning("No unmatched nutrition file found")
        return 0

        try:
        # Load current aliases
        with open(alias_path, 'r', encoding='utf-8') as f:
        aliases = json.load(f)

        # Load unmatched items
        df = pd.read_csv(unmatched_path, encoding='utf-8')
        if df.empty:
        return 0

        # Add high-confidence aliases (this is a simplified version)
        # In a real implementation, you'd have more sophisticated matching logic
        added = 0
        for _, row in df.head(count).iterrows():
        ingredient = row.get('ingredient', '').strip().lower()
        if not ingredient or ingredient in aliases:
        continue

        # Simple alias suggestions based on common patterns
        if 'coconut milk' in ingredient:
        aliases[ingredient] = 'coconut milk'
        added += 1
        elif 'red lentil' in ingredient or 'dal' in ingredient:
        aliases[ingredient] = 'red lentils'
        added += 1
        elif 'green bean' in ingredient or 'long bean' in ingredient:
        aliases[ingredient] = 'green beans'
        added += 1
        elif 'onion' in ingredient:
        aliases[ingredient] = 'onion'
        added += 1
        elif 'chicken' in ingredient:
        aliases[ingredient] = 'chicken'
        added += 1

        # Save updated aliases
        with open(alias_path, 'w', encoding='utf-8') as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)

        logger.info(f"Added {added} nutrition aliases")
        self.log_change(f"Added {added} nutrition aliases")
        return added

        except Exception as e:
        logger.error(f"Error adding nutrition aliases: {e}")
        return 0

    def add_pricing_aliases(self, count: int = 20) -> int:
        """Add high-confidence pricing aliases from unmatched tokens"""
        unmatched_path = "reports/unmatched_tokens.csv"
        alias_path = "config/item_alias_user.json"

        if not os.path.exists(unmatched_path):
        logger.warning("No unmatched tokens file found")
        return 0

        try:
        # Load current aliases
        with open(alias_path, 'r', encoding='utf-8') as f:
        aliases = json.load(f)

        # Load unmatched items
        df = pd.read_csv(unmatched_path, encoding='utf-8')
        if df.empty:
        return 0

        # Add high-confidence aliases
        added = 0
        for _, row in df.head(count).iterrows():
        ingredient = row.get('unmatched_name', '').strip().lower()
        if not ingredient or ingredient in aliases:
        continue

        # Simple alias suggestions
        if 'coconut milk' in ingredient:
        aliases[ingredient] = 'coconut milk'
        added += 1
        elif 'red lentil' in ingredient or 'dal' in ingredient:
        aliases[ingredient] = 'red lentils'
        added += 1
        elif 'green bean' in ingredient:
        aliases[ingredient] = 'green beans'
        added += 1
        elif 'big onion' in ingredient or 'large onion' in ingredient:
        aliases[ingredient] = 'big onion'
        added += 1
        elif 'chicken' in ingredient:
        aliases[ingredient] = 'chicken'
        added += 1

        # Save updated aliases
        with open(alias_path, 'w', encoding='utf-8') as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)

        logger.info(f"Added {added} pricing aliases")
        self.log_change(f"Added {added} pricing aliases")
        return added

        except Exception as e:
        logger.error(f"Error adding pricing aliases: {e}")
        return 0

    def run_pipeline(self, input_file: str = "recipes_base.csv") -> Dict:
        """Run the complete pipeline with auto-iteration"""
        self.ensure_directories()

        # Initialize changelog
        with open(self.changelog_path, 'w', encoding='utf-8') as f:
        f.write("FitFeast Pipeline Execution Log\n")
        f.write("================================\n\n")

        logger.info(f"Starting FitFeast pipeline with {input_file}")

        current_file = input_file
        previous_metrics = None

        for iteration in range(1, self.max_iterations + 1):
        self.current_iteration = iteration
        logger.info(f"\n=== ROUND {iteration} ===")

        # Step 1: Normalization
        norm_passed, norm_file = self.run_serving(current_file)
        if not norm_passed:
        # Try post-fix
        logger.info("Normalization gates failed, applying post-fix...")
        norm_file = self.post_fix_normalization(norm_file)
        norm_passed, _ = self.run_serving(current_file) # Re-check after fix

        # Step 2: Nutrition
        nutr_passed, nutr_file = self.run_nutrition(norm_file)

        # Step 3: Pricing
        price_passed, price_file = self.run_pricing(nutr_file)

        # Collect current metrics
        current_metrics = self.collect_metrics()

        # Check if all gates pass
        all_passed = norm_passed and nutr_passed and price_passed
        if all_passed:
        logger.info(f"\n ALL ACCEPTANCE GATES PASSED in round {iteration}!")
        break

        # Check for no improvement
        if previous_metrics and not self.has_improvement(previous_metrics, current_metrics):
        logger.info("No improvement detected, stopping iterations")
        break

        # Add aliases based on what failed
        if not nutr_passed:
        self.add_nutrition_aliases(20)
        if not price_passed:
        self.add_pricing_aliases(20)

        previous_metrics = current_metrics
        current_file = input_file # Always start from base for consistency

        # Final summary
        final_metrics = self.collect_metrics()
        summary = self.generate_summary(final_metrics, iteration)

        logger.info("\n" + "="*60)
        logger.info("FINAL PIPELINE SUMMARY")
        logger.info("="*60)
        for line in summary['summary_lines']:
        logger.info(line)

        if not summary['all_passed']:
        logger.info("\n Some gates still failing. Check reports/ for details.")
        if summary['top_unmatched_nutrition']:
        logger.info("Top unmatched nutrition items:")
        for item in summary['top_unmatched_nutrition'][:10]:
        logger.info(f" {item}")
        if summary['top_unmatched_pricing']:
        logger.info("Top unmatched pricing items:")
        for item in summary['top_unmatched_pricing'][:10]:
        logger.info(f" {item}")

        return summary

    def collect_metrics(self) -> Dict:
        """Collect current metrics from all reports"""
        metrics = {}

        # Normalization metrics
        norm_path = "reports/normalization_metrics.json"
        if os.path.exists(norm_path):
        with open(norm_path, 'r', encoding='utf-8') as f:
        metrics['normalization'] = json.load(f)

        # Nutrition metrics
        nutr_path = "reports/nutrition_coverage_report.json"
        if os.path.exists(nutr_path):
        with open(nutr_path, 'r', encoding='utf-8') as f:
        metrics['nutrition'] = json.load(f)

        # Pricing metrics
        price_path = "reports/pricing_coverage_report.json"
        if os.path.exists(price_path):
        with open(price_path, 'r', encoding='utf-8') as f:
        metrics['pricing'] = json.load(f)

        return metrics

    def has_improvement(self, prev: Dict, curr: Dict) -> bool:
        """Check if current metrics show improvement over previous"""
        try:
        # Check nutrition improvement
        prev_nutr = prev.get('nutrition', {}).get('coverage_pct', 0)
        curr_nutr = curr.get('nutrition', {}).get('coverage_pct', 0)

        # Check pricing improvement
        prev_price = prev.get('pricing', {}).get('coverage_percentages', {}).get('baseline', 0)
        curr_price = curr.get('pricing', {}).get('coverage_percentages', {}).get('baseline', 0)

        # Consider improvement if either metric improved by at least 1%
        return (curr_nutr > prev_nutr + 1.0) or (curr_price > prev_price + 1.0)

        except Exception:
        return True # Assume improvement if we can't compare

    def generate_summary(self, metrics: Dict, iterations: int) -> Dict:
        """Generate final summary"""
        summary = {
        'iterations_used': iterations,
        'all_passed': False,
        'normalization_passed': False,
        'nutrition_passed': False,
        'pricing_passed': False,
        'summary_lines': [],
        'top_unmatched_nutrition': [],
        'top_unmatched_pricing': []
        }

        # Check gates
        norm_metrics = metrics.get('normalization', {})
        if norm_metrics:
        norm_passed = (
        norm_metrics.get('pct_digit_or_unit', 0) >= self.normalization_gates['pct_digit_or_unit_min'] and
        norm_metrics.get('glued_count', 999) <= self.normalization_gates['glued_count_max'] and
        norm_metrics.get('parentheses_count', 999) <= self.normalization_gates['parentheses_count_max']
        )
        summary['normalization_passed'] = norm_passed

        nutr_metrics = metrics.get('nutrition', {})
        if nutr_metrics:
        nutr_passed = nutr_metrics.get('coverage_pct', 0) >= self.nutrition_gates['coverage_min']
        summary['nutrition_passed'] = nutr_passed

        price_metrics = metrics.get('pricing', {})
        if price_metrics:
        coverage = price_metrics.get('coverage_percentages', {})
        price_passed = (
        coverage.get('baseline', 0) >= self.pricing_gates['baseline_min'] and
        coverage.get('strict', 0) >= self.pricing_gates['strict_min'] and
        coverage.get('very_strict', 0) >= self.pricing_gates['very_strict_min']
        )
        summary['pricing_passed'] = price_passed

        summary['all_passed'] = summary['normalization_passed'] and summary['nutrition_passed'] and summary['pricing_passed']

        # Build summary lines
        lines = []

        # Normalization
        if norm_metrics:
        status = "PASS" if summary['normalization_passed'] else "FAIL"
        lines.append(f"A) Normalization: {status}")
        lines.append(f" - Digit/unit coverage: {norm_metrics.get('pct_digit_or_unit', 0)}% (need 92%)")
        lines.append(f" - Glued units: {norm_metrics.get('glued_count', 0)} (need 0)")
        lines.append(f" - Parentheses: {norm_metrics.get('parentheses_count', 0)} (need 0)")
        else:
        lines.append("A) Normalization: NO METRICS")

        # Nutrition
        if nutr_metrics:
        status = "PASS" if summary['nutrition_passed'] else "FAIL"
        coverage = nutr_metrics.get('coverage_pct', 0)
        lines.append(f"B) Nutrition: {status}")
        lines.append(f" - Coverage: {coverage}% (need 90%)")
        else:
        lines.append("B) Nutrition: NO METRICS")

        # Pricing
        if price_metrics:
        status = "PASS" if summary['pricing_passed'] else "FAIL"
        coverage = price_metrics.get('coverage_percentages', {})
        lines.append(f"C) Pricing: {status}")
        lines.append(f" - Baseline: {coverage.get('baseline', 0)}% (need 98%)")
        lines.append(f" - Strict: {coverage.get('strict', 0)}% (need 80%)")
        lines.append(f" - Very-strict: {coverage.get('very_strict', 0)}% (need 70%)")
        else:
        lines.append("C) Pricing: NO METRICS")

        summary['summary_lines'] = lines

        # Get top unmatched items
        try:
        unmatched_nutr_path = "reports/unmatched_nutrition.csv"
        if os.path.exists(unmatched_nutr_path):
        df = pd.read_csv(unmatched_nutr_path, encoding='utf-8')
        summary['top_unmatched_nutrition'] = df['ingredient'].head(50).tolist()

        unmatched_price_path = "reports/unmatched_tokens.csv"
        if os.path.exists(unmatched_price_path):
        df = pd.read_csv(unmatched_price_path, encoding='utf-8')
        summary['top_unmatched_pricing'] = df['unmatched_name'].head(50).tolist()
        except Exception:
        pass

        return summary


    def main():
        """Main entry point"""
        runner = PipelineRunner()

        # Check if recipes_base.csv exists
        if not os.path.exists("recipes_base.csv"):
        logger.error("recipes_base.csv not found in current directory")
        return 1

        try:
        summary = runner.run_pipeline()

        # Exit with appropriate code
        return 0 if summary['all_passed'] else 1

        except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


        if __name__ == "__main__":
        sys.exit(main())
