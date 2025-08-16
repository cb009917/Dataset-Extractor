#!/usr/bin/env python3
"""Debug glued unit count."""

import re
from normalize_per_person import IngredientNormalizer

def main():
    normalizer = IngredientNormalizer()
    
    test = "250gchicken"
    print(f"Original: '{test}'")
    
    result = normalizer.deglue_spacing_rules(test)
    print(f"After deglue: '{result}'")
    
    # Check what's being counted
    canonical_units = ['kg', 'g', 'l', 'ml', 'tsp', 'tbsp', 'cup', 'oz', 'lb', 'pcs', 'clove', 'can', 'tin', 'pkt', 'bunch']
    
    for unit in canonical_units:
        pattern = r'\b(' + re.escape(unit) + r')(?=[A-Za-z])'
        matches = re.findall(pattern, result, re.IGNORECASE)
        if matches:
            print(f"Unit '{unit}' found glued: {matches}")

if __name__ == "__main__":
    main()