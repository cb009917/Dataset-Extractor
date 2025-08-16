#!/usr/bin/env python3
"""
Per-person ingredient normalizer for downstream nutrition and pricing stages.
Implements comprehensive text normalization, A/B/C triage, and per-person scaling.
"""

import re
import csv
import sys
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngredientNormalizer:
    def __init__(self):
        # Canonical units with their aliases
        self.unit_aliases = {
            # Mass
            'grams': 'g', 'gram': 'g', 'gm': 'g', 
            'kilogram': 'kg', 'kilograms': 'kg', 'kilo': 'kg', 'kgs': 'kg',
            
            # Volume
            'milliliter': 'ml', 'milliliters': 'ml', 'millilitre': 'ml', 'millilitres': 'ml', 'mls': 'ml',
            'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l', 'lts': 'l',
            
            # Cooking measures
            'teaspoon': 'tsp', 'teaspoons': 'tsp', 'tea spoon': 'tsp', 'tea spoons': 'tsp',
            'tablespoon': 'tbsp', 'tablespoons': 'tbsp', 'table spoon': 'tbsp', 'table spoons': 'tbsp', 'tbl': 'tbsp', 'tbls': 'tbsp',
            'cups': 'cup',
            
            # Imperial
            'ounce': 'oz', 'ounces': 'oz', 'ozs': 'oz',
            'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb',
            
            # Pieces
            'piece': 'pcs', 'pieces': 'pcs', 'pc': 'pcs', 'pce': 'pcs', 'pces': 'pcs',
            'cloves': 'clove',
            'cans': 'can', 'tins': 'tin',
            'packet': 'pkt', 'packets': 'pkt', 'pack': 'pkt', 'packs': 'pkt', 'pkts': 'pkt',
            'bunches': 'bunch'
        }
        
        # Canonical units list
        self.canonical_units = ['kg', 'g', 'l', 'ml', 'tsp', 'tbsp', 'cup', 'oz', 'lb', 'pcs', 'clove', 'can', 'tin', 'pkt', 'bunch']
        
        # Build safe unit alternation pattern
        all_units = self.canonical_units + list(self.unit_aliases.keys())
        # Sort by length descending to match longer units first
        all_units_sorted = sorted(all_units, key=len, reverse=True)
        self.unit_pattern = '(?:' + '|'.join(re.escape(unit) for unit in all_units_sorted) + ')'
        
        # Unicode fraction mapping
        self.unicode_fractions = {
            '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
            '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
            '⅑': '1/9', '⅒': '1/10', '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6', '⅚': '5/6'
        }
        
        # A-list: fillers & meta (completely exclude)
        self.a_list = [
            'salt', 'pepper', 'water', 'hot water', 'cold water', 'warm water', 'lukewarm water',
            'boiled water', 'normal water', 'boiling water', 'to taste', 'as required', 
            'optional', 'for garnish', 'for serving', 'for tempering', 'for marinade',
            'as you need', 'as you want', 'adjust to your taste', 'as needed'
        ]
        
        # B-list: micro-spices/herbs (log nominal but exclude from nutrition/pricing)
        self.b_list = [
            'turmeric', 'cumin seed', 'cumin powder', 'cumin', 'mustard seed', 'mustard seeds',
            'fenugreek', 'coriander powder', 'coriander seed', 'coriander seeds', 'chilli powder',
            'chili powder', 'chilli flakes', 'chili flakes', 'curry leaves', 'curry leaf',
            'pandan', 'rampe', 'cardamom', 'clove', 'cloves', 'cinnamon', 'cinnamon stick',
            'nutmeg', 'mace', 'black pepper', 'white pepper', 'red chili powder', 'green chili powder',
            'paprika', 'cayenne', 'bay leaves', 'bay leaf', 'thyme', 'oregano', 'basil',
            'rosemary', 'sage', 'mint', 'cilantro', 'parsley', 'dill', 'chilli', 'chili'
        ]
        
        # Conversion factors to grams (approximate)
        self.to_grams = {
            'g': 1.0, 'kg': 1000.0,
            'ml': 1.0, 'l': 1000.0,  # Assume water density
            'tsp': 5.0, 'tbsp': 15.0, 'cup': 240.0,
            'oz': 28.35, 'lb': 453.6,
            'pcs': 10.0, 'clove': 3.0, 'can': 400.0, 'tin': 400.0, 'pkt': 50.0, 'bunch': 30.0
        }
        
        # Statistics
        self.stats = {
            'total_tokens': 0,
            'tokens_with_digit_or_unit': 0,
            'glued_unit_word_count': 0,
            'parentheses_count': 0,
            'warnings': [],
            'needs_quantity': []
        }
    
    def clean_unicode_and_fractions(self, text: str) -> str:
        """Clean Unicode characters and normalize fractions."""
        if not text:
            return text
            
        # Replace unicode fraction slash ⁄ → /
        text = text.replace('⁄', '/')
        
        # Map common unicode fractions
        for unicode_frac, ascii_frac in self.unicode_fractions.items():
            text = text.replace(unicode_frac, ascii_frac)
        
        # Normalize multiple spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def strip_parentheticals(self, text: str) -> str:
        """Remove all (...) segments anywhere in text (non-greedy, repeat until none)."""
        if not text:
            return text
            
        # Count parentheses for statistics before removal
        self.stats['parentheses_count'] += text.count('(') + text.count(')')
        
        # Remove parentheticals repeatedly until none remain
        while '(' in text and ')' in text:
            text = re.sub(r'\([^)]*\)', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_heading_or_banner(self, line: str) -> bool:
        """Detect headings/banners that should be dropped."""
        line = line.strip()
        if not line:
            return True
            
        # Lines containing only dashes/equals
        if re.match(r'^[-=\s]+$', line):
            return True
            
        # Lines starting with "For " and ending with ":"
        if re.match(r'^For\s.*:$', line, re.IGNORECASE):
            return True
            
        # Other common heading patterns
        heading_patterns = [
            r'^\s*ingredients?\s*:?\s*$',
            r'^\s*method\s*:?\s*$',
            r'^\s*preparation\s*:?\s*$',
            r'^\s*instructions?\s*:?\s*$',
            r'^\s*for\s+.*:$'
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
                
        return False
    
    def split_multi_ingredient_lines(self, text: str, title: str = "") -> List[str]:
        """Split on |, •, ·, ; and drop headings/banners."""
        if not text:
            return []
            
        # Split on various delimiters
        delimiters = r'[|•·;]'
        parts = re.split(delimiters, text)
        
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if it's a heading/banner
            if self.is_heading_or_banner(part):
                if title:
                    self.stats['warnings'].append({
                        'title': title,
                        'reason': 'Non-ingredient text',
                        'original': part,
                        'cleaned': 'DROPPED'
                    })
                continue
                
            cleaned_parts.append(part)
            
        return cleaned_parts
    
    def detect_suffix_quantity(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Detect patterns like:
        Ingredient – 250g
        Ingredient — ½ tsp  
        Ingredient - 2 cups
        Ingredient: 3 tbsp
        
        Returns (ingredient, quantity, unit) or (None, None, None)
        """
        # Patterns for different dash types and colon - support mixed numbers
        patterns = [
            r'^(.+?)\s*[–—-]\s*(\d+(?:\s+\d+/\d+)?(?:\.\d+)?|\d+/\d+)\s*(' + self.unit_pattern + r')\s*$',
            r'^(.+?)\s*:\s*(\d+(?:\s+\d+/\d+)?(?:\.\d+)?|\d+/\d+)\s*(' + self.unit_pattern + r')\s*$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                ingredient = match.group(1).strip()
                quantity = match.group(2).strip()
                unit = match.group(3).strip().lower()
                
                # Normalize unit
                unit = self.unit_aliases.get(unit, unit)
                
                return ingredient, quantity, unit
                
        return None, None, None
    
    def parse_quantity_string(self, qty_str: str) -> float:
        """Parse quantity string that may contain fractions or mixed numbers."""
        if not qty_str:
            return 0.0
            
        qty_str = qty_str.strip()
        
        # Handle mixed numbers like "2 1/3"
        mixed_match = re.match(r'(\d+)\s+(\d+/\d+)', qty_str)
        if mixed_match:
            whole = float(mixed_match.group(1))
            frac_parts = mixed_match.group(2).split('/')
            if len(frac_parts) == 2 and frac_parts[1] != '0':
                fraction = float(frac_parts[0]) / float(frac_parts[1])
                return whole + fraction
        
        # Handle pure fractions like "3/4"
        if '/' in qty_str:
            parts = qty_str.split('/')
            if len(parts) == 2:
                try:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    if denominator != 0:
                        return numerator / denominator
                except (ValueError, ZeroDivisionError):
                    pass
        
        # Handle decimal numbers
        try:
            return float(qty_str)
        except ValueError:
            return 0.0
    
    def parse_leading_quantity(self, text: str) -> Tuple[Optional[float], Optional[str], str]:
        """Parse leading quantity, unit, and ingredient name."""
        if not text:
            return None, None, text
        
        text = text.strip()
        
        # Try patterns for leading quantity first - support mixed numbers and fractions
        patterns = [
            # Pattern 1: Mixed number + unit + ingredient (e.g., "2 1/2 cups rice")
            r'^(\d+\s+\d+/\d+)\s+(' + self.unit_pattern + r')\s+(.*?)$',
            # Pattern 2: Fraction + unit + ingredient (e.g., "1/2 cup rice")
            r'^(\d+/\d+)\s+(' + self.unit_pattern + r')\s+(.*?)$',
            # Pattern 3: Decimal + unit + ingredient (e.g., "2.5 cups rice")
            r'^(\d+(?:\.\d+)?)\s+(' + self.unit_pattern + r')\s+(.*?)$',
            # Pattern 4: Number + unit (no space) + ingredient (e.g., "2cups rice", "250g chicken") 
            r'^(\d+(?:\s+\d+/\d+)?(?:\.\d+)?|\d+/\d+)(' + self.unit_pattern + r')\s+(.*?)$',
            # Pattern 5: Just number + ingredient (e.g., "2 onions")
            r'^(\d+(?:\s+\d+/\d+)?(?:\.\d+)?|\d+/\d+)\s+(.*?)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                quantity_str = match.group(1).strip()
                if len(match.groups()) >= 3:
                    unit = (match.group(2) or "").strip().lower()
                    ingredient = match.group(3).strip()
                else:
                    unit = ""
                    ingredient = match.group(2).strip()
                
                quantity = self.parse_quantity_string(quantity_str)
                
                # Normalize unit
                if unit:
                    unit = self.unit_aliases.get(unit, unit)
                
                return quantity, unit, ingredient
        
        return None, None, text
    
    def deglue_spacing_rules(self, text: str) -> str:
        """Two-pass deglue & spacing rules implementation."""
        if not text:
            return text
        
        # First pass: Insert missing space between number and unit
        # Pattern: number directly followed by unit
        for unit in self.canonical_units:
            pattern = r'(?<![A-Za-z])(\d+(?:[.,]\d+)?(?:\s+\d+/\d+|\d+/\d+)?)\s*(' + re.escape(unit) + r')\b'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        
        # Second pass: Insert missing space between unit and word
        for unit in self.canonical_units:
            pattern = r'\b(' + re.escape(unit) + r')(?=[A-Za-z])'
            text = re.sub(pattern, r'\1 ', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Count remaining glued instances AFTER attempting to fix them
        for unit in self.canonical_units:
            pattern = r'\b(' + re.escape(unit) + r')(?=[A-Za-z])'
            remaining_glued = re.findall(pattern, text, re.IGNORECASE)
            self.stats['glued_unit_word_count'] += len(remaining_glued)
        
        return text
    
    def triage_ingredient(self, ingredient_name: str, quantity: Optional[float] = None, unit: str = "") -> str:
        """
        Triage system: A (drop), B (micro-spices), C (material)
        Returns: 'A', 'B', or 'C'
        """
        if not ingredient_name:
            return 'A'
            
        ingredient_lower = ingredient_name.lower().strip()
        
        # A-list: fillers & meta
        for a_item in self.a_list:
            if a_item in ingredient_lower:
                return 'A'
        
        # Check for filler patterns
        if any(pattern in ingredient_lower for pattern in ['to taste', 'as required', 'optional', 'for garnish']):
            return 'A'
        
        # B-list: micro-spices/herbs  
        for b_item in self.b_list:
            if b_item in ingredient_lower:
                return 'B'
        
        # C-list: material ingredients (everything else)
        return 'C'
    
    def scale_per_person(self, quantity: float, unit: str, servings: int) -> Tuple[float, str]:
        """Scale quantity per person with reasonable precision."""
        if servings <= 0:
            servings = 1
            
        per_person_qty = quantity / servings
        
        # Keep reasonable precision based on unit type
        if unit in ['g', 'ml']:
            # Grams/ml → 0.1 precision
            per_person_qty = round(per_person_qty, 1)
        elif unit in ['tsp', 'tbsp', 'cup']:
            # Spoons/cups → 0.25 precision
            per_person_qty = round(per_person_qty * 4) / 4
        elif unit in ['pcs', 'clove', 'can', 'tin', 'pkt', 'bunch']:
            # Pieces → 1 precision
            per_person_qty = round(per_person_qty)
        else:
            # Default → 0.1 precision
            per_person_qty = round(per_person_qty, 1)
            
        return per_person_qty, unit
    
    def should_apply_tiny_spice_rule(self, ingredient_name: str, grams: float) -> bool:
        """Check if tiny-spice rule applies (<2g and B-list)."""
        return grams < 2.0 and self.triage_ingredient(ingredient_name) == 'B'
    
    def estimate_grams(self, quantity: float, unit: str) -> float:
        """Estimate weight in grams for given quantity and unit."""
        conversion_factor = self.to_grams.get(unit, 10.0)  # Default 10g if unknown
        return quantity * conversion_factor
    
    def has_digit_or_unit(self, text: str) -> bool:
        """Check if text contains digit or recognized unit."""
        has_digit = bool(re.search(r'\d', text))
        has_unit = bool(re.search(r'\b(' + '|'.join(self.canonical_units) + r')\b', text, re.IGNORECASE))
        return has_digit or has_unit
    
    def process_ingredient_line(self, ingredient_line: str, servings: int, title: str = "") -> List[str]:
        """Process a single ingredient line and return normalized tokens."""
        if not ingredient_line or not isinstance(ingredient_line, str):
            return []
            
        # Step 1: Unicode & fraction cleanup
        text = self.clean_unicode_and_fractions(ingredient_line)
        
        # Step 2: Strip parentheticals globally
        text = self.strip_parentheticals(text)
        
        # Step 3: Split multi-ingredient lines
        parts = self.split_multi_ingredient_lines(text, title)
        
        normalized_tokens = []
        
        for part in parts:
            if not part.strip():
                continue
                
            self.stats['total_tokens'] += 1
            
            # Step 4: Check for suffix quantities
            ingredient, qty_str, unit = self.detect_suffix_quantity(part)
            
            if ingredient and qty_str and unit:
                # Rewrite to leading form
                quantity = self.parse_quantity_string(qty_str)
                part = f"{quantity} {unit} {ingredient}"
            
            # Step 5: Apply deglue & spacing rules
            part = self.deglue_spacing_rules(part)
            
            # Step 6: Parse leading quantity
            quantity, unit, ingredient_name = self.parse_leading_quantity(part)
            
            # Step 7: A/B/C triage
            triage = self.triage_ingredient(ingredient_name, quantity, unit)
            
            if triage == 'A':
                # Drop completely
                continue
                
            if triage == 'C' and quantity is None:
                # Material ingredient without quantity
                if title:
                    self.stats['needs_quantity'].append({
                        'title': title,
                        'original': ingredient_line.strip(),
                        'reason': 'material_no_quantity'
                    })
                continue
            
            if triage == 'B':
                # Log nominal quantity for QA but will exclude later
                if quantity is None:
                    quantity = 0.5  # Nominal 0.5g
                    unit = 'g'
                elif not unit:
                    unit = 'g'  # Default to grams for B-list items
            
            if quantity is None:
                # No quantity found, skip
                continue
            
            # Handle missing unit for pure integers with food nouns
            if not unit and quantity and triage == 'C':
                # Safe default to "pcs" for countable items
                unit = 'pcs'
            elif not unit and quantity:
                # Default unit for any numeric quantity
                unit = 'pcs'
            
            # Step 8: Per-person scaling
            scaled_qty, scaled_unit = self.scale_per_person(quantity, unit, servings)
            
            # Step 9: Tiny-spice rule
            estimated_grams = self.estimate_grams(scaled_qty, scaled_unit)
            if self.should_apply_tiny_spice_rule(ingredient_name, estimated_grams):
                continue  # Drop tiny spices
            
            # Step 10: Build normalized token
            if scaled_qty > 0:
                # Format quantity appropriately
                if scaled_qty == int(scaled_qty):
                    qty_formatted = str(int(scaled_qty))
                else:
                    qty_formatted = str(scaled_qty)
                    
                normalized_token = f"{qty_formatted} {scaled_unit} {ingredient_name}"
                normalized_tokens.append(normalized_token)
                
                # Update statistics - count as having digit/unit
                self.stats['tokens_with_digit_or_unit'] += 1
        
        return normalized_tokens
    
    def process_csv(self, input_file: str, output_file: str = None):
        """Process CSV file with ingredient normalization."""
        if not output_file:
            base_name = Path(input_file).stem
            output_file = f"{base_name}_normalized.csv"
            
        logger.info(f"Processing {input_file} -> {output_file}")
        
        # Reset statistics
        self.stats = {
            'total_tokens': 0,
            'tokens_with_digit_or_unit': 0,
            'glued_unit_word_count': 0,
            'parentheses_count': 0,
            'warnings': [],
            'needs_quantity': []
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                
                reader = csv.DictReader(infile)
                
                # Ensure required columns exist
                if 'ingredients' not in reader.fieldnames:
                    logger.error("Input CSV must have 'ingredients' column")
                    return
                
                if 'servings' not in reader.fieldnames:
                    logger.error("Input CSV must have 'servings' column") 
                    return
                
                # Add ingredients_per_person column
                fieldnames = list(reader.fieldnames)
                if 'ingredients_per_person' not in fieldnames:
                    fieldnames.append('ingredients_per_person')
                
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    ingredients = row.get('ingredients', '')
                    servings_str = row.get('servings', '1')
                    title = row.get('title', row.get('name', ''))
                    
                    try:
                        servings = int(float(servings_str))
                        if servings <= 0:
                            servings = 1
                    except (ValueError, TypeError):
                        servings = 1
                        
                    # Process ingredients
                    normalized_tokens = self.process_ingredient_line(ingredients, servings, title)
                    
                    # Join with | delimiter
                    row['ingredients_per_person'] = ' | '.join(normalized_tokens)
                    
                    writer.writerow(row)
                    
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return
            
        # Write warnings and needs_quantity files
        self.write_warnings()
        self.write_needs_quantity()
        
        # Print metrics
        self.print_metrics()
    
    def write_warnings(self):
        """Write normalization warnings to CSV."""
        if not self.stats['warnings']:
            return
            
        with open('normalization_warnings.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'reason', 'original', 'cleaned'])
            writer.writeheader()
            writer.writerows(self.stats['warnings'])
    
    def write_needs_quantity(self):
        """Write ingredients needing quantities to CSV."""
        if not self.stats['needs_quantity']:
            return
            
        with open('needs_quantity.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'original', 'reason'])
            writer.writeheader()
            writer.writerows(self.stats['needs_quantity'])
    
    def print_metrics(self):
        """Print processing metrics."""
        total = self.stats['total_tokens']
        digit_unit_pct = (self.stats['tokens_with_digit_or_unit'] / total * 100) if total > 0 else 0
        
        print("\n=== NORMALIZATION METRICS ===")
        print(f"Total tokens: {total}")
        print(f"Tokens with digit or unit: {self.stats['tokens_with_digit_or_unit']} ({digit_unit_pct:.1f}%)")
        print(f"Glued unit-word still present: {self.stats['glued_unit_word_count']}")
        print(f"Tokens with parentheses: {self.stats['parentheses_count']}")
        print(f"Warnings logged: {len(self.stats['warnings'])}")
        print(f"Items needing quantity: {len(self.stats['needs_quantity'])}")
        
        # Check acceptance targets
        targets_met = True
        if digit_unit_pct < 80.0:
            print(f"FAIL: digit/unit% ({digit_unit_pct:.1f}%) < 80%")
            targets_met = False
        else:
            print(f"PASS: digit/unit% ({digit_unit_pct:.1f}%) >= 80%")
            
        if self.stats['glued_unit_word_count'] > 0:
            print(f"FAIL: glued unit-word ({self.stats['glued_unit_word_count']}) > 0")
            targets_met = False
        else:
            print("PASS: glued unit-word = 0")
            
        if self.stats['parentheses_count'] > 0:
            print(f"FAIL: parentheses ({self.stats['parentheses_count']}) > 0")
            targets_met = False
        else:
            print("PASS: parentheses = 0")
        
        if targets_met:
            print("\nALL ACCEPTANCE TARGETS MET!")
        else:
            print("\nSOME TARGETS FAILED - see output above")

def main():
    if len(sys.argv) != 2:
        print("Usage: python normalize_per_person.py <input_csv>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    normalizer = IngredientNormalizer()
    normalizer.process_csv(input_file)

if __name__ == "__main__":
    main()