#!/usr/bin/env python3
"""Quick debug script to test normalize_per_person functionality."""

from normalize_per_person import IngredientNormalizer

def main():
    normalizer = IngredientNormalizer()
    
    test_cases = [
        "Rice 2 cups",
        "250g chicken",
        "1 tsp salt", 
        "2tspginger",
        "Onion 1 piece",
    ]
    
    for test in test_cases:
        print(f"\nTesting: '{test}'")
        
        # Step by step processing
        print(f"1. Clean unicode: '{normalizer.clean_unicode_and_fractions(test)}'")
        
        step2 = normalizer.strip_parentheticals(test)
        print(f"2. Strip parentheses: '{step2}'")
        
        step3_parts = normalizer.split_multi_ingredient_lines(step2)
        print(f"3. Split parts: {step3_parts}")
        
        if step3_parts:
            part = step3_parts[0]
            
            # Check for suffix quantities
            ingredient, qty_str, unit = normalizer.detect_suffix_quantity(part)
            print(f"4. Suffix quantity: ingredient='{ingredient}', qty='{qty_str}', unit='{unit}'")
            
            # Apply deglue
            step5 = normalizer.deglue_spacing_rules(part)
            print(f"5. Deglue: '{step5}'")
            
            # Parse leading quantity
            quantity, unit, ingredient_name = normalizer.parse_leading_quantity(step5)
            print(f"6. Leading quantity: qty={quantity}, unit='{unit}', ingredient='{ingredient_name}'")
            
            # A/B/C triage
            triage = normalizer.triage_ingredient(ingredient_name)
            print(f"7. Triage: {triage}")
            
            # Final result
            result = normalizer.process_ingredient_line(test, 4)
            print(f"8. Final result: {result}")

if __name__ == "__main__":
    main()