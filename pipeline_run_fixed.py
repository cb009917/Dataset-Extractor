#!/usr/bin/env python3
"""
FitFeast Auto-Iteration ETL Pipeline
Runs serving → nutrition → pricing with gate checking and auto-learning until convergence or max rounds
"""

import json
import logging
import os
import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

# Set UTF-8 encoding at process start
os.environ["PYTHONIOENCODING"] = "utf-8"

from common_text import load_alias, save_alias, canonicalize_name

# Constants
REPORTS_ROOT = Path("reports")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gate thresholds
GATES = {
    "norm_pct_digit_unit": 92.0,
    "norm_glued": 0,
    "norm_parens": 0,
    "nutrition_coverage": 100.0,   # Must reach 100%
    "price_baseline": 100.0,       # Must reach 100%
    "price_strict": 80.0,
    "price_very_strict": 70.0
}

MAX_ROUNDS = 8

class PipelineRunner:
    def __init__(self):
        self.current_round = 0
        self.previous_metrics = {}
        self.improvement_threshold = 1.0  # 1% absolute improvement required
        
        # Ensure required directories exist
        os.makedirs('reports', exist_ok=True)
        
    def run_command(self, cmd: List[str], desc: str) -> Tuple[bool, str]:
        """Run a command and capture output"""
        try:
            logger.info(f"Running: {desc}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {desc}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False, e.stderr
    
    def run_serving_stage(self) -> Tuple[bool, Dict]:
        """Run serving.py and extract normalization metrics"""
        cmd = [sys.executable, "serving.py", "recipes_base.csv"]
        success, output = self.run_command(cmd, "Serving normalization")
        
        if not success:
            return False, {}
        
        # Load metrics from reports/normalization_metrics.json
        try:
            with open('reports/normalization_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            norm_metrics = {
                'total_tokens': metrics.get('total_tokens', 0),
                'pct_digit_or_unit': metrics.get('pct_digit_or_unit', 0.0),
                'glued_count': metrics.get('glued_count', 0),
                'parentheses_count': metrics.get('parentheses_count', 0)
            }
            
            logger.info(f"Normalization metrics: {norm_metrics}")
            return True, norm_metrics
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load normalization metrics: {e}")
            return False, {}
    
    def run_nutrition_stage(self) -> Tuple[bool, float]:
        """Run nutrition_calculator.py and extract coverage"""
        cmd = [sys.executable, "nutrition_calculator.py", "recipes_base_normalized.csv"]
        success, output = self.run_command(cmd, "Nutrition calculation")
        
        if not success:
            return False, 0.0
        
        # Extract coverage from output
        coverage = 0.0
        for line in output.split('\n'):
            if 'NUTRITION_COVERAGE:' in line:
                try:
                    coverage = float(line.split(':')[1].strip())
                    break
                except (ValueError, IndexError):
                    pass
        
        # Also try to load from reports file
        if coverage == 0.0:
            try:
                with open('reports/nutrition_coverage.json', 'r') as f:
                    report = json.load(f)
                coverage = report.get('coverage', 0.0) * 100.0  # Convert fraction to percentage
            except:
                pass
        
        logger.info(f"Nutrition coverage: {coverage}%")
        return True, coverage
    
    def run_pricing_stage(self) -> Tuple[bool, Dict]:
        """Run price_calculator.py and extract coverage metrics"""
        cmd = [sys.executable, "price_calculator.py", "recipes_base_normalized_improved_nutrition.csv"]
        success, output = self.run_command(cmd, "Price calculation")
        
        if not success:
            return False, {}
        
        # Extract coverage from output
        baseline = strict = very_strict = 0.0
        
        for line in output.split('\n'):
            if 'PRICING_BASELINE:' in line:
                try:
                    baseline = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'PRICING_STRICT:' in line:
                try:
                    strict = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'PRICING_VERY_STRICT:' in line:
                try:
                    very_strict = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
        
        # Also try to load from reports file
        if baseline == 0.0:
            try:
                with open('reports/pricing_coverage_report.json', 'r') as f:
                    report = json.load(f)
                    coverage = report.get('coverage_percentages', {})
                    baseline = coverage.get('baseline', 0.0)
                    strict = coverage.get('strict', 0.0)
                    very_strict = coverage.get('very_strict', 0.0)
            except:
                pass
        
        pricing_metrics = {
            'baseline': baseline,
            'strict': strict,
            'very_strict': very_strict
        }
        
        logger.info(f"Pricing coverage: {pricing_metrics}")
        return True, pricing_metrics
    
    def check_gates(self, norm_metrics: Dict, nutrition_coverage: float, pricing_metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if all gates pass - focus on nutrition 100% and pricing baseline 100%"""
        failures = []
        
        # Normalization gates (keep as-is)
        if norm_metrics.get('pct_digit_or_unit', 0) < GATES['norm_pct_digit_unit']:
            failures.append(f"Digit/unit coverage: {norm_metrics.get('pct_digit_or_unit', 0):.1f}% < {GATES['norm_pct_digit_unit']}%")
        
        if norm_metrics.get('glued_count', 0) > GATES['norm_glued']:
            failures.append(f"Glued units: {norm_metrics.get('glued_count', 0)} > {GATES['norm_glued']}")
        
        if norm_metrics.get('parentheses_count', 0) > GATES['norm_parens']:
            failures.append(f"Parentheses: {norm_metrics.get('parentheses_count', 0)} > {GATES['norm_parens']}")
        
        # CRITICAL: Nutrition coverage must be 100%
        if nutrition_coverage < GATES['nutrition_coverage']:
            failures.append(f"Nutrition coverage: {nutrition_coverage:.1f}% < {GATES['nutrition_coverage']}% ⚠️ REQUIRED")
        
        # CRITICAL: Price baseline must be 100%
        if pricing_metrics.get('baseline', 0) < GATES['price_baseline']:
            failures.append(f"Price baseline: {pricing_metrics.get('baseline', 0):.1f}% < {GATES['price_baseline']}% ⚠️ REQUIRED")
        
        # Track but don't block on strict/very-strict metrics
        if pricing_metrics.get('strict', 0) < GATES['price_strict']:
            logger.info(f"Note: Price strict: {pricing_metrics.get('strict', 0):.1f}% < {GATES['price_strict']}% (tracking only)")
        
        if pricing_metrics.get('very_strict', 0) < GATES['price_very_strict']:
            logger.info(f"Note: Price very-strict: {pricing_metrics.get('very_strict', 0):.1f}% < {GATES['price_very_strict']}% (tracking only)")
        
        return len(failures) == 0, failures
    
    def clean_malformed_head(self, head: str) -> str:
        """Clean malformed heads that still contain quantities/units"""
        if not head:
            return ""
        
        s = head.lower().strip()
        
        # Remove quantities and units that got left in (more targeted than global canonicalize)
        # Pattern: number + optional space + unit at start
        s = re.sub(r'^\d+(\.\d+)?\s*(g|kg|ml|l|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)\s+', '', s)
        
        # Remove trailing quantities and units
        s = re.sub(r'\s+\d+(\.\d+)?\s*(g|kg|ml|l|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)$', '', s)
        
        # Clean up space-separated letters that should be together FIRST (like 'l oose' -> 'loose')
        s = re.sub(r'\bl\s+o([a-z])', r'lo\1', s)  # Fix 'l oose' -> 'loose', 'l otus' -> 'lotus'
        
        # Remove standalone units at start (like 'g otu kola')
        s = re.sub(r'^(g|kg|ml|l|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)\s+', '', s)
        
        # Remove malformed fragments
        s = re.sub(r'\s+g\s+', ' ', s)  # Remove isolated 'g'
        s = re.sub(r'\s+l\s+', ' ', s)  # Remove isolated 'l'
        
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def load_unmatched_heads(self) -> List[str]:
        """Load unmatched heads from both nutrition and pricing reports"""
        heads = []
        
        # Load from nutrition unmatched
        nutrition_file = REPORTS_ROOT / "unmatched_nutrition.csv"
        if nutrition_file.exists():
            try:
                df = pd.read_csv(nutrition_file)
                # Try candidate columns in order of preference
                cand_cols = ["ingredient_head", "canonical_head", "canonical_suggested", "unmatched_name", "raw_token", "raw"]
                head_col = next((c for c in cand_cols if c in df.columns), None)
                if head_col:
                    raw_heads = [str(x).strip() for x in df[head_col] if str(x).strip()]
                    cleaned_heads = [self.clean_malformed_head(h) for h in raw_heads]
                    nutrition_heads = [h.lower() for h in cleaned_heads if h]
                    heads.extend(nutrition_heads)
                    logger.info(f"Loaded {len(nutrition_heads)} unmatched heads from nutrition")
            except Exception as e:
                logger.warning(f"Could not load nutrition unmatched: {e}")
        
        # Load from pricing unmatched  
        pricing_file = REPORTS_ROOT / "unmatched_tokens.csv"
        if pricing_file.exists():
            try:
                df = pd.read_csv(pricing_file)
                head_col = "ingredient_head" if "ingredient_head" in df.columns else (
                    "unmatched_name" if "unmatched_name" in df.columns else None)
                if head_col:
                    raw_heads = [str(x).strip() for x in df[head_col] if str(x).strip()]
                    cleaned_heads = [self.clean_malformed_head(h) for h in raw_heads]
                    pricing_heads = [h.lower() for h in cleaned_heads if h]
                    heads.extend(pricing_heads)
                    logger.info(f"Loaded {len(pricing_heads)} unmatched heads from pricing")
            except Exception as e:
                logger.warning(f"Could not load pricing unmatched: {e}")
        
        # Return unique heads, sorted
        unique_heads = sorted(set(heads))
        logger.info(f"Total unique unmatched heads to process: {len(unique_heads)}")
        return unique_heads

    def learn_from_unmatched(self) -> int:
        """Learn aliases from unmatched files using enhanced fuzzy matching with sanity guards"""
        unmatched_heads = self.load_unmatched_heads()
        if not unmatched_heads:
            logger.info("No unmatched heads found")
            return 0
        
        current_aliases = load_alias()
        original_alias_count = len(current_aliases)
        
        # Load candidate databases
        nutrition_candidates = self.load_nutrition_candidates()
        pricing_candidates = self.load_pricing_candidates()
        
        logger.info(f"Loaded {len(nutrition_candidates)} nutrition candidates, {len(pricing_candidates)} pricing candidates")
        
        new_aliases = {}
        nutrition_learned = 0
        pricing_learned = 0
        
        # Process each unmatched head
        for head in unmatched_heads:
            canonical_head = canonicalize_name(head)
            
            # Skip if already has alias
            if canonical_head in current_aliases:
                continue
                
            # Apply sanity guards
            if not self.passes_sanity_checks(canonical_head):
                continue
            
            # Try fuzzy matching against both databases
            best_match = None
            best_score = 0.0
            match_source = None
            
            # Try nutrition database
            if nutrition_candidates:
                nutrition_match, nutrition_score = self.fuzzy_match_with_jaccard(canonical_head, nutrition_candidates)
                if nutrition_score > best_score:
                    best_match = nutrition_match
                    best_score = nutrition_score
                    match_source = "nutrition"
            
            # Try pricing database  
            if pricing_candidates:
                pricing_match, pricing_score = self.fuzzy_match_with_jaccard(canonical_head, pricing_candidates)
                if pricing_score > best_score:
                    best_match = pricing_match
                    best_score = pricing_score
                    match_source = "pricing"
            
            # Accept if score meets threshold (start with lower threshold for better learning)
            threshold = 0.30 if self.current_round <= 2 else 0.34
            if best_match and best_score >= threshold:
                # Additional validation for the match
                if self.validate_match(canonical_head, best_match, match_source):
                    new_aliases[canonical_head] = best_match
                    if match_source == "nutrition":
                        nutrition_learned += 1
                    else:
                        pricing_learned += 1
                    logger.info(f"Learned alias: '{canonical_head}' → '{best_match}' (score: {best_score:.3f}, source: {match_source})")
        
        # Save new aliases (merge, don't overwrite)
        if new_aliases:
            merged_aliases = {**current_aliases, **new_aliases}
            save_alias(merged_aliases)
            
            # Save diff for this round
            round_diff_file = REPORTS_ROOT / f"alias_diff_round{self.current_round}.json"
            with open(round_diff_file, 'w', encoding='utf-8') as f:
                json.dump(new_aliases, f, indent=2, ensure_ascii=False)
            
            total_learned = len(new_aliases)
            logger.info(f"Learned {total_learned} new aliases (nutrition: {nutrition_learned}, pricing: {pricing_learned})")
            logger.info(f"Alias diff saved to {round_diff_file}")
            return total_learned
        else:
            logger.info("No new aliases learned")
            return 0
    
    def load_nutrition_candidates(self) -> List[str]:
        """Load nutrition database names as candidates for fuzzy matching"""
        try:
            nutrition_db_path = Path('config/ingredient-dataset_nutrition.xlsx')
            if nutrition_db_path.exists():
                df = pd.read_excel(nutrition_db_path)
                # Try common column names
                name_col = None
                for col in ['Name', 'name', 'Ingredient', 'ingredient']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col:
                    names = [str(name).strip() for name in df[name_col] if pd.notna(name)]
                    return [name for name in names if name and name.lower() != 'nan']
            return []
        except Exception as e:
            logger.warning(f"Could not load nutrition candidates: {e}")
            return []
    
    def load_pricing_candidates(self) -> List[str]:
        """Load pricing database names as candidates for fuzzy matching"""
        try:
            price_data_path = Path('config/extracted_prices2.csv')
            if price_data_path.exists():
                df = pd.read_csv(price_data_path, encoding='utf-8')
                # Try common column names for item names
                name_col = None
                for col in ['Name', 'name', 'Item', 'item', 'Commodity', 'commodity', 'Description', 'description']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col:
                    names = [str(name).strip() for name in df[name_col] if pd.notna(name)]
                    return [name for name in names if name and name.lower() != 'nan']
            return []
        except Exception as e:
            logger.warning(f"Could not load pricing candidates: {e}")
            return []
    
    def fuzzy_match_with_jaccard(self, query: str, candidates: List[str], threshold: float = 0.30) -> Tuple[Optional[str], float]:
        """Fuzzy match using Jaccard similarity with stopword removal"""
        if not query or not candidates:
            return None, 0.0
        
        # Remove stopwords
        stopwords = {'fresh', 'large', 'small', 'medium', 'ground', 'sliced', 'chopped', 'diced', 'minced', 'whole', 'raw', 'cooked', 'dried'}
        query_tokens = set(canonicalize_name(query).split()) - stopwords
        if not query_tokens:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            candidate_tokens = set(canonicalize_name(candidate).split()) - stopwords
            if not candidate_tokens:
                continue
            
            # Jaccard similarity
            intersection = len(query_tokens & candidate_tokens)
            union = len(query_tokens | candidate_tokens)
            
            if union > 0:
                score = intersection / union
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = candidate
        
        return best_match, best_score
    
    def passes_sanity_checks(self, head: str) -> bool:
        """Apply sanity guards to prevent learning garbage"""
        if not head or len(head) < 2:
            return False
        
        # Don't map if head is numeric
        if head.isdigit():
            return False
        
        # Don't map salt/pepper/water unless it has material unit & qty context
        a_list_items = {'salt', 'pepper', 'water', 'black pepper', 'white pepper'}
        if head.lower() in a_list_items:
            return False
        
        return True
    
    def validate_match(self, head: str, match: str, source: str) -> bool:
        """Additional validation for proposed matches"""
        # Don't map to A-list items unless head has material context
        a_list_items = {'salt', 'pepper', 'water'}
        if match.lower() in a_list_items:
            return False
        
        if source == "nutrition":
            # For nutrition, could check if matched item has valid macro data
            # For now, just basic validation
            return True
        
        elif source == "pricing":
            # For pricing, could check if matched item has valid price range (50-2000 LKR/kg)
            # For now, just basic validation
            return True
        
        return True
    
    def print_debugging_info(self, round_num: int):
        """Print debugging information for convergence issues"""
        logger.info(f"\n🔍 DEBUGGING INFO - ROUND {round_num}")
        logger.info("-" * 50)
        
        # Show current alias count
        current_aliases = load_alias()
        logger.info(f"Current alias count: {len(current_aliases)}")
        
        # Show top unmatched heads with counts
        unmatched_heads = self.load_unmatched_heads()
        if unmatched_heads:
            # Get counts from unmatched files
            head_counts = Counter(unmatched_heads)
            logger.info(f"\nTop 50 unmatched heads (with estimated counts):")
            for head, count in head_counts.most_common(50):
                logger.info(f"  {head} ×{count}")
        
        # Show recent alias additions
        round_diff_file = REPORTS_ROOT / f"alias_diff_round{round_num}.json"
        if round_diff_file.exists():
            try:
                with open(round_diff_file, 'r', encoding='utf-8') as f:
                    recent_aliases = json.load(f)
                logger.info(f"\nAliases learned this round: {len(recent_aliases)}")
                for k, v in list(recent_aliases.items())[:10]:  # Show first 10
                    logger.info(f"  '{k}' → '{v}'")
            except Exception:
                pass
        
        logger.info("-" * 50)
    
    def save_final_summary(self):
        """Save final summary when pipeline completes or stops"""
        try:
            # Gather final metrics
            final_summary = {
                'pipeline_completed': False,
                'total_rounds': self.current_round,
                'final_aliases_count': len(load_alias()),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add latest metrics if available
            latest_summary_file = REPORTS_ROOT / f"round_{self.current_round}_summary.json"
            if latest_summary_file.exists():
                with open(latest_summary_file, 'r') as f:
                    latest = json.load(f)
                final_summary.update({
                    'final_normalization': latest.get('normalization', {}),
                    'final_nutrition_coverage': latest.get('nutrition_coverage', 0),
                    'final_pricing': latest.get('pricing', {}),
                    'gates_passed': latest.get('gates_passed', False)
                })
                final_summary['pipeline_completed'] = latest.get('gates_passed', False)
            
            # List final output files
            final_files = [
                "recipes_base_normalized.csv",
                "recipes_base_normalized_improved_nutrition.csv", 
                "recipes_base_normalized_improved_nutrition_with_costs.csv"
            ]
            
            final_summary['output_files'] = {}
            for file in final_files:
                if os.path.exists(file):
                    final_summary['output_files'][file] = {
                        'exists': True,
                        'size_kb': round(os.path.getsize(file) / 1024, 1)
                    }
                else:
                    final_summary['output_files'][file] = {'exists': False}
            
            # Save final summary
            final_summary_file = REPORTS_ROOT / "final_summary.json"
            with open(final_summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Final summary saved to {final_summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not save final summary: {e}")
    
    def has_significant_improvement(self, current_metrics: Dict) -> bool:
        """Check if current metrics show significant improvement over previous round"""
        if not self.previous_metrics:
            return True  # First round always counts as improvement
        
        # Key metrics to track improvement
        key_metrics = [
            ('nutrition_coverage', 'nutrition_coverage'),
            ('price_baseline', 'pricing.baseline'),
            ('price_strict', 'pricing.strict'),
            ('price_very_strict', 'pricing.very_strict')
        ]
        
        for metric_name, metric_path in key_metrics:
            current_val = self.get_nested_value(current_metrics, metric_path)
            previous_val = self.get_nested_value(self.previous_metrics, metric_path)
            
            if current_val is not None and previous_val is not None:
                improvement = current_val - previous_val
                if improvement >= self.improvement_threshold:
                    logger.info(f"Significant improvement in {metric_name}: {improvement:.1f}%")
                    return True
        
        return False
    
    def get_nested_value(self, data: Dict, path: str):
        """Get nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def save_round_summary(self, round_num: int, norm_metrics: Dict, nutrition_coverage: float, 
                          pricing_metrics: Dict, gates_passed: bool, failures: List[str]):
        """Save summary for this round"""
        summary = {
            'round': round_num,
            'gates_passed': gates_passed,
            'gate_failures': failures,
            'normalization': norm_metrics,
            'nutrition_coverage': nutrition_coverage,
            'pricing': pricing_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_file = f'reports/round_{round_num}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved round {round_num} summary to {summary_file}")
        return summary
    
    def run_pipeline(self) -> bool:
        """Run the complete auto-iteration pipeline"""
        logger.info("=" * 60)
        logger.info("FITFEAST AUTO-ITERATION ETL PIPELINE")
        logger.info("=" * 60)
        
        for round_num in range(1, MAX_ROUNDS + 1):
            self.current_round = round_num
            logger.info(f"\n🔄 ROUND {round_num}")
            logger.info("-" * 40)
            
            # Stage 1: Serving/Normalization
            success, norm_metrics = self.run_serving_stage()
            if not success:
                logger.error(f"Serving stage failed in round {round_num}")
                return False
            
            # Stage 2: Nutrition
            success, nutrition_coverage = self.run_nutrition_stage()
            if not success:
                logger.error(f"Nutrition stage failed in round {round_num}")
                return False
            
            # Stage 3: Pricing
            success, pricing_metrics = self.run_pricing_stage()
            if not success:
                logger.error(f"Pricing stage failed in round {round_num}")
                return False
            
            # Check gates
            gates_passed, failures = self.check_gates(norm_metrics, nutrition_coverage, pricing_metrics)
            
            # Save round summary
            current_metrics = {
                'normalization': norm_metrics,
                'nutrition_coverage': nutrition_coverage,
                'pricing': pricing_metrics
            }
            
            self.save_round_summary(round_num, norm_metrics, nutrition_coverage, 
                                  pricing_metrics, gates_passed, failures)
            
            # Report results
            logger.info("\n📊 ROUND RESULTS:")
            logger.info(f"  Normalization: {norm_metrics.get('pct_digit_or_unit', 0):.1f}% digit/unit, "
                       f"{norm_metrics.get('glued_count', 0)} glued, {norm_metrics.get('parentheses_count', 0)} parens")
            logger.info(f"  Nutrition: {nutrition_coverage:.1f}% coverage")
            logger.info(f"  Pricing: {pricing_metrics.get('baseline', 0):.1f}% baseline, "
                       f"{pricing_metrics.get('strict', 0):.1f}% strict, {pricing_metrics.get('very_strict', 0):.1f}% very-strict")
            
            if gates_passed:
                logger.info("\n✅ ALL GATES PASSED!")
                logger.info(f"Pipeline converged after {round_num} rounds")
                self.save_final_summary()
                self.print_final_results()
                return True
            else:
                logger.info(f"\n❌ Gates failed: {len(failures)} issues")
                for failure in failures:
                    logger.info(f"    • {failure}")
            
            # Check for improvement plateau
            if round_num > 1 and not self.has_significant_improvement(current_metrics):
                logger.info(f"\n⚠️  No significant improvement since last round")
                if round_num >= 3:  # Give at least 3 rounds
                    logger.info("Stopping due to improvement plateau")
                    self.save_final_summary()
                    self.print_final_results()
                    return False
            
            # Learn from unmatched data
            if round_num < MAX_ROUNDS:
                logger.info("\n🧠 LEARNING PHASE:")
                aliases_added = self.learn_from_unmatched()
                
                if aliases_added == 0:
                    logger.info("No new aliases learned - may have reached convergence")
                    # Print debugging info when learning stalls
                    self.print_debugging_info(round_num)
                    if round_num >= 3:  # Give at least 3 rounds
                        logger.info("Stopping due to no learning progress")
                        self.save_final_summary()
                        self.print_final_results()
                        return False
                else:
                    logger.info(f"Learned {aliases_added} new aliases, continuing to next round")
            
            # Store metrics for next round comparison
            self.previous_metrics = current_metrics
        
        logger.info(f"\n⏰ Maximum rounds ({MAX_ROUNDS}) reached without full convergence")
        self.save_final_summary()
        self.print_final_results()
        return False
    
    def print_final_results(self):
        """Print final pipeline results"""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL PIPELINE RESULTS")
        logger.info("=" * 60)
        
        # List final output files
        final_files = [
            "recipes_base_normalized.csv",
            "recipes_base_normalized_improved_nutrition.csv", 
            "recipes_base_normalized_improved_nutrition_with_costs.csv"
        ]
        
        logger.info("\n📁 Final output files:")
        for file in final_files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # KB
                logger.info(f"  ✅ {file} ({size:.1f} KB)")
            else:
                logger.info(f"  ❌ {file} (missing)")
        
        # Show final metrics
        try:
            with open(f'reports/round_{self.current_round}_summary.json', 'r', encoding='utf-8') as f:
                final_summary = json.load(f)
            
            logger.info("\n📊 Final metrics:")
            norm = final_summary.get('normalization', {})
            logger.info(f"  Normalization: {norm.get('pct_digit_or_unit', 0):.1f}% digit/unit")
            logger.info(f"  Nutrition: {final_summary.get('nutrition_coverage', 0):.1f}% coverage")
            pricing = final_summary.get('pricing', {})
            logger.info(f"  Pricing: {pricing.get('baseline', 0):.1f}% baseline, "
                       f"{pricing.get('strict', 0):.1f}% strict, {pricing.get('very_strict', 0):.1f}% very-strict")
            
            # Show gate status
            if final_summary.get('gates_passed', False):
                logger.info("\n🎉 All acceptance gates PASSED")
            else:
                logger.info(f"\n⚠️  {len(final_summary.get('gate_failures', []))} gates still failing")
                
        except Exception as e:
            logger.warning(f"Could not load final summary: {e}")
        
        logger.info("\n" + "=" * 60)

def main():
    """Main entry point"""
    # Ensure we have the base recipes file
    if not os.path.exists('recipes_base.csv'):
        logger.error("recipes_base.csv not found! Please ensure the base dataset exists.")
        return 1
    
    # Run the pipeline
    runner = PipelineRunner()
    success = runner.run_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())