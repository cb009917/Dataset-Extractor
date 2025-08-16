"""
One-pass ingredient normalizer

Transforms messy tokens like:
* `Beef oxtail - 1/6kg`
* `1/8cupwater`
* `tbspcoconut oil`
* `turmeric powder 1/2 teaspoon`
* `salt – as you need`

into canonical form:
* `1/6 kg beef oxtail`
* `1/8 cup water`
* `1 tbsp coconut oil`
* `1/2 tsp turmeric powder`
* *(skip or set 0 g for "as needed" items)*
"""

import re
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class IngredientNormalizer:
    def __init__(self):
        # Unicode fraction mappings including fraction slash
        self.unicode_fractions = {
            '½': '1/2', '¼': '1/4', '¾': '3/4', '⅓': '1/3', '⅔': '2/3',
            '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
            '⅚': '5/6', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
            '⁄': '/'  # Unicode fraction slash to regular slash
        }
        
        # Unit canonicalization map
        self.unit_map = {
            'tablespoon': 'tbsp', 'tablespoons': 'tbsp',
            'teaspoon': 'tsp', 'teaspoons': 'tsp',
            'grams': 'g', 'gram': 'g',
            'kilograms': 'kg', 'kilogram': 'kg',
            'litre': 'l', 'litres': 'l', 'liter': 'l', 'liters': 'l',
            'cups': 'cup',
            'pieces': 'piece', 'cloves': 'clove', 'sprigs': 'sprig',
            'slices': 'slice', 'cans': 'can', 'packets': 'packet',
            'bunches': 'bunch'
        }
        
        # Quantity words to numbers
        self.quantity_words = {
            'a': '1', 'an': '1', 'one': '1', 'two': '2', 'three': '3', 
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 
            'nine': '9', 'ten': '10', 'half': '1/2', 'quarter': '1/4', 
            'third': '1/3', 'few': '3', 'couple': '2', 'some': '1'
        }
        
        # Liquid densities (g/ml) for volume conversions
        self.liquid_densities = {
            'water': 1.00, 'coconut water': 1.00, 'broth': 1.00, 'vinegar': 1.00,
            'milk': 1.03, 'coconut milk': 1.03, 'yogurt': 1.03,
            'oil': 0.92, 'coconut oil': 0.92, 'vegetable oil': 0.92, 'olive oil': 0.92,
            'soy sauce': 1.20, 'fish sauce': 1.20,
            'honey': 1.42, 'treacle': 1.42, 'syrup': 1.33
        }
        
        # Cooking state keywords to preserve
        self.cooking_states = {'cooked', 'boiled', 'steamed', 'fried', 'raw'}
        
        # "As needed" patterns
        self.as_needed_patterns = [
            r'\bto taste\b', r'\bas needed\b', r'\bas you need\b', 
            r'\boptional\b', r'\baccording to taste\b'
        ]

    def normalize_token(self, raw_token: str) -> tuple[str, bool]:
        """
        Normalize a single ingredient token.
        Returns: (normalized_string, should_skip)
        """
        if not raw_token or not raw_token.strip():
            return "", True
            
        # Step 1: Trim & lowercase
        token = raw_token.strip().lower()
        
        # Step 2: Replace ⁄ with / and Unicode fractions → ASCII
        for unicode_frac, ascii_frac in self.unicode_fractions.items():
            token = token.replace(unicode_frac, ascii_frac)
        
        # Step 2.5: Collapse triple-fractions a/b/c → b/c
        token = self._collapse_triple_fractions(token)
        
        # Step 3: Fix glued number–unit
        token = re.sub(r'(?<=\d)(?=(g|kg|ml|l|cup|tbsp|tsp)\b)', ' ', token)
        # Also handle fractions glued to units like "1/8cupwater"
        token = re.sub(r'(\d+/\d+)(cup|tbsp|tsp|g|kg|ml|l)', r'\1 \2', token)
        
        # Step 3.5: Insert space after unit when followed by a letter (more comprehensive)
        token = re.sub(r'\b(\d+)(g|kg|ml|l|tsp|tbsp|cup|cups)(?=[a-zA-Z])', r'\1 \2 ', token)
        token = re.sub(r'\b(g|kg|ml|l|tsp|tbsp|cup|cups)(\d+)', r'\1 \2', token)
        
        # Step 4: First canonicalize units & plurals (before fixing glued patterns)
        for old_unit, new_unit in self.unit_map.items():
            token = re.sub(r'\b' + re.escape(old_unit) + r'\b', new_unit, token)
        
        # Step 5: Fix glued unit–ingredient (after plurals are fixed)
        token = re.sub(r'\b(cup|tbsp|tsp|g|kg|ml|l)(?=[a-z])', r'\1 ', token)
        # Also fix ingredient followed by number+unit
        token = re.sub(r'([a-z])(\d+)(g|kg|ml|l|tsp|tbsp|cup|cups)', r'\1 \2 \3', token)
        
        # Step 6: Replace quantity words with numbers (before regex matching)
        for word, number in self.quantity_words.items():
            token = re.sub(r'\b' + re.escape(word) + r'\b', number, token)
        
        # Step 7: Handle "ingredient – qty unit" (hyphen format)
        pattern_a = r'^(?P<name>.+?)\s*[-–]\s*(?P<qty>\d+(?:\.\d+)?|\d+\s*\d+/\d+|\d+/\d+)\s*(?P<unit>[a-z]+)\b'
        match_a = re.match(pattern_a, token)
        if match_a:
            name, qty, unit = match_a.groups()
            token = f"{qty} {unit} {name.strip()}"
            logger.info(f"Normalized hyphen format: '{raw_token}' -> '{token}'")
        
        # Step 8: Handle "ingredient qty unit" (unit at end) - more flexible patterns
        elif re.search(r'\b(cup|tbsp|tsp|g|kg|ml|l|piece|clove|sprig|slice|can|packet|pinch|dash|bunch)\b$', token):
            # Pattern for "ingredient qty unit" with complex fractions
            pattern_b = r'^(.+?)\s+(\d+(?:\s+\d+/\d+|\s*\.\d+)?|\d+/\d+)\s*(cup|tbsp|tsp|g|kg|ml|l|piece|clove|sprig|slice|can|packet|pinch|dash|bunch)\b'
            match_b = re.match(pattern_b, token)
            if match_b:
                name, qty, unit = match_b.groups()
                token = f"{qty.strip()} {unit} {name.strip()}"
                logger.info(f"Normalized end-unit format: '{raw_token}' -> '{token}'")
        
        # Step 9: Handle "qty unit ingredient" (already correct format - just verify)
        pattern_c = r'^(\d+(?:\s+\d+/\d+|\s*\.\d+)?|\d+/\d+)\s*(cup|tbsp|tsp|g|kg|ml|l|piece|clove|sprig|slice|can|packet|pinch|dash|bunch)\b\s+(.+)'
        match_c = re.match(pattern_c, token)
        if match_c:
            # Already in correct format, just clean up
            qty, unit, name = match_c.groups()
            token = f"{qty.strip()} {unit} {name.strip()}"
        
        # Step 10: Remove "of" after unit
        token = re.sub(r'^(\S+)\s+(\S+)\s+of\s+', r'\1 \2 ', token)
        
        # Step 11: Handle missing quantity (add default of 1)
        # If we have a unit but no quantity at the start, add 1
        if re.match(r'^(tbsp|tsp|cup|g|kg|ml|l|piece|clove|sprig|slice|can|packet|pinch|dash|bunch)\s+', token):
            token = f"1 {token}"
            logger.info(f"Added default quantity: '{raw_token}' -> '{token}'")
        
        # Step 12: Handle "to taste / as needed / optional"
        has_as_needed = any(re.search(pattern, token) for pattern in self.as_needed_patterns)
        if has_as_needed:
            # Check if there's a clear quantity
            has_quantity = re.search(r'\b\d+(?:\.\d+)?(?:\s*\d+)?/?(?:\d+)?\s*(cup|tbsp|tsp|g|kg|ml|l|piece|clove|sprig|slice|can|packet|pinch|dash|bunch)\b', token)
            if not has_quantity:
                logger.info(f"Skipping 'as needed' item with no quantity: '{raw_token}'")
                return token, True  # Skip this item
            else:
                # Remove the "as needed" part but keep the quantity
                for pattern in self.as_needed_patterns:
                    token = re.sub(pattern, '', token)
                logger.info(f"Removed 'as needed' but kept quantity: '{raw_token}' -> '{token}'")
        
        # Step 13: Preserve cooking state keywords
        # Keep tokens like 'cooked|boiled|fried|steamed|raw' in the name
        # This is handled naturally by keeping them in the ingredient name
        
        # Step 14: Collapse spaces; strip
        token = re.sub(r'\s+', ' ', token).strip()
        
        return token, False
    
    def _collapse_triple_fractions(self, token: str) -> str:
        """Collapse triple-fractions a/b/c → b/c"""
        # Pattern to match number/number/number or more
        pattern = r'(\d+)/(\d+)/(\d+)(?:/(\d+))*'
        
        def collapse_fraction(match):
            groups = match.groups()
            # Find the last two non-None groups
            non_none = [g for g in groups if g is not None]
            if len(non_none) >= 2:
                return f"{non_none[-2]}/{non_none[-1]}"
            return match.group(0)  # Return original if can't process
        
        return re.sub(pattern, collapse_fraction, token)

    def get_liquid_density(self, ingredient_name: str) -> float:
        """Get density for liquid ingredients (g/ml)"""
        ingredient_lower = ingredient_name.lower()
        
        for liquid, density in self.liquid_densities.items():
            if liquid in ingredient_lower:
                return density
        
        # Default to water density with warning
        logger.warning(f"Unknown liquid density for '{ingredient_name}', using 1.0 g/ml")
        return 1.0

    def normalize_ingredient_list(self, ingredients_str: str) -> list[str]:
        """
        Normalize a full ingredient string (pipe-separated tokens)
        Returns list of normalized tokens, filtering out skipped items
        """
        if not ingredients_str or not ingredients_str.strip():
            return []
        
        tokens = [token.strip() for token in ingredients_str.split('|') if token.strip()]
        normalized_tokens = []
        
        for token in tokens:
            normalized, should_skip = self.normalize_token(token)
            if not should_skip and normalized:
                normalized_tokens.append(normalized)
            elif should_skip:
                logger.info(f"Skipped ingredient: '{token}'")
        
        return normalized_tokens
    
    def parse_servings(self, servings_str: str, overrides: Optional[dict] = None) -> int:
        """Parse servings string to derive numeric value, skip rows without clean integer"""
        if not servings_str or pd.isna(servings_str):
            return 0
            
        servings_str = str(servings_str).strip().lower()
        
        # Check overrides first if provided
        if overrides and servings_str in overrides:
            return overrides[servings_str]
        
        # Extract numeric values
        numbers = re.findall(r'\d+', servings_str)
        if not numbers:
            return 0  # Skip row
            
        # Handle single number (most common)
        if len(numbers) == 1:
            return int(numbers[0])
            
        # Handle ranges like "4-6" - take average
        if len(numbers) == 2 and ('-' in servings_str or 'to' in servings_str):
            return int((int(numbers[0]) + int(numbers[1])) / 2)
            
        # If multiple numbers without clear pattern, skip
        return 0
    
    def has_suspicious_values(self, calories: float, ingredients_text: str) -> bool:
        """Check for suspicious recipe values (guardrails)"""
        if calories <= 1500:
            return False
            
        # Check for suspicious tokens when calories > 1500
        suspicious_patterns = [
            (r'(\d+)\s*g(?:rams?)?\b(?![a-z])', 1200),  # grams > 1200
            (r'coconut\s+milk.*(\d+)\s*ml', 300),       # coconut milk > 300ml
        ]
        
        for pattern, limit in suspicious_patterns:
            matches = re.findall(pattern, ingredients_text.lower())
            for match in matches:
                try:
                    value = int(match)
                    if value > limit:
                        logger.warning(f"Suspicious value detected: {value} (limit: {limit})")
                        return True
                except ValueError:
                    continue
                    
        return False

# Test function for the normalizer
def test_normalizer():
    """Test the normalizer with the examples from the spec"""
    normalizer = IngredientNormalizer()
    
    test_cases = [
        'Beef oxtail - 1/6kg',
        '1/8cupwater',
        'tbspcoconut oil',
        'turmeric powder 1/2 teaspoon',
        'salt – as you need',
        '1 can coconut milk',
        '½ cup rice flour',
        'a pinch of salt',
        '2 3/4 cups water',
        'few curry leaves',
        'some fresh ginger to taste',
        '1⁄4 cup rice flour',  # Unicode fraction slash test
        '1/4/2/3 cup water',   # Triple fraction test
        'Water 133 1/3ml',     # Reordering test
        '25gflour',            # Space after unit test
        'coconut milk500ml'    # Space after unit test
    ]
    
    print("\nTESTING INGREDIENT NORMALIZER")
    print("="*60)
    
    for test_input in test_cases:
        try:
            normalized, should_skip = normalizer.normalize_token(test_input)
            status = "SKIP" if should_skip else "KEEP"
            print(f"Input:  '{test_input}'")
            print(f"Output: '{normalized}' [{status}]")
            print()
        except UnicodeEncodeError:
            # Handle Unicode display issues
            print(f"Input:  [Unicode test case]")
            normalized, should_skip = normalizer.normalize_token(test_input)
            status = "SKIP" if should_skip else "KEEP"
            print(f"Output: '{normalized}' [{status}]")
            print()
    
    # Test servings parsing
    print("\nTESTING SERVINGS PARSING")
    print("="*40)
    servings_tests = ["4", "4-6", "serves 4", "2 people", "invalid", ""]
    for s in servings_tests:
        result = normalizer.parse_servings(s)
        print(f"'{s}' -> {result}")
    
    # Test guardrails
    print("\nTESTING GUARDRAILS")
    print("="*40)
    test_cases = [
        (1200, "100g flour | 200ml milk"),
        (1600, "1500g flour | 400ml coconut milk"),
        (1800, "500g flour | 200ml milk")
    ]
    for calories, ingredients in test_cases:
        suspicious = normalizer.has_suspicious_values(calories, ingredients)
        print(f"{calories} kcal, '{ingredients}' -> Suspicious: {suspicious}")

if __name__ == "__main__":
    test_normalizer()