#!/usr/bin/env python3
"""
Missing Ingredients Analyzer
Identifies all ingredients from recipes that are not found in the price CSV
"""

import pandas as pd
import re
from typing import Dict, List, Set, Optional
from collections import Counter

# Ingredient aliases for better matching
INGREDIENT_ALIASES = {
    'onion': ['onions', 'big onion', 'red onion', 'white onion', 'big onions'],
    'tomato': ['tomatoes'],
    'garlic': ['garlic cloves', 'garlic paste'],
    'ginger': ['ginger paste', 'fresh ginger'],
    'chili': ['chillies', 'green chili', 'red chili', 'chilli', 'chilies', 'green chillies'],
    'coconut': ['fresh coconut', 'grated coconut', 'coconut grated'],
    'rice': ['basmati rice', 'jasmine rice', 'white rice', 'raw rice', 'idli rice'],
    'chicken': ['chicken pieces', 'chicken breast', 'chicken thigh'],
    'curry leaves': ['fresh curry leaves'],
    'cinnamon': ['cinnamon stick'],
    'cardamom': ['green cardamom'],
    'cloves': ['whole cloves'],
    'cumin': ['cumin seeds'],
    'coriander': ['coriander seeds', 'fresh coriander'],
    'turmeric': ['turmeric powder'],
    'chili powder': ['red chili powder', 'chilli powder'],
    'black pepper': ['pepper powder', 'ground black pepper']
}

def evaluate_fraction(fraction_str: str) -> float:
    """Convert fraction string to decimal"""
    if '/' in fraction_str:
        parts = fraction_str.split('/')
        return float(parts[0]) / float(parts[1])
    return float(fraction_str)

def parse_ingredient_line(ingredient_str: str) -> Optional[Dict]:
    """Parse ingredient line to extract quantity, unit, and ingredient name"""
    cleaned = ingredient_str.strip().lower()
    
    # Skip instruction lines
    if re.match(r'^\d+\s+(add|mix|cook|heat|boil|fry|wash|drain|loosen|pour|close|enjoy)', cleaned):
        return None
    
    # Remove price information like "$8", "$23", etc.
    cleaned = re.sub(r'\$\d+(?:\.\d+)?', '', cleaned)
    
    # Pattern 1: "2 tbsp ginger paste" or "1/2 cup rice flour"
    pattern1 = re.match(r'^(\d+(?:/\d+)?(?:\.\d+)?)\s+(\w+)\s+(.+)', cleaned)
    if pattern1:
        quantity, unit, ingredient = pattern1.groups()
        return {
            'quantity': evaluate_fraction(quantity),
            'unit': unit.lower(),
            'ingredient': clean_ingredient_name(ingredient)
        }
    
    # Pattern 2: "2 onions" (just quantity and ingredient)
    pattern2 = re.match(r'^(\d+(?:/\d+)?(?:\.\d+)?)\s+(.+)', cleaned)
    if pattern2:
        quantity, ingredient = pattern2.groups()
        return {
            'quantity': evaluate_fraction(quantity),
            'unit': 'piece',
            'ingredient': clean_ingredient_name(ingredient)
        }
    
    # Default: assume 1 piece
    return {
        'quantity': 1,
        'unit': 'piece',
        'ingredient': clean_ingredient_name(cleaned)
    }

def clean_ingredient_name(ingredient: str) -> str:
    """Clean ingredient name to focus on the actual food item"""
    cleaned = ingredient.lower().strip()
    
    # Remove preparation instructions
    removal_patterns = [
        r'\b(roughly|finely|thinly|thickly)\s+\w+',
        r'\b(chopped|sliced|diced|minced|grated|ground|peeled|cleaned)\b',
        r'\b(fresh|dried|frozen|cooked|raw|roasted|boiled)\b',
        r'\b(to make.*)',
        r'\b(for.*)',
        r'\b(soaked.*)',
        r'\b(bought.*)',
        r'\b(each.*)',
    ]
    
    for pattern in removal_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Remove extra punctuation and spaces
    cleaned = re.sub(r'[,\(\)\[\]"]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def find_best_match(ingredient_name: str, price_df: pd.DataFrame) -> Optional[pd.Series]:
    """Find best matching price item for ingredient"""
    normalized = ingredient_name.lower().strip()
    
    # Direct match
    direct_match = price_df[price_df['Name_Lower'] == normalized]
    if not direct_match.empty:
        return direct_match.iloc[0]
    
    # Check aliases
    for standard, aliases in INGREDIENT_ALIASES.items():
        if any(alias in normalized for alias in aliases) or standard in normalized:
            alias_match = price_df[price_df['Name_Lower'].str.contains(standard, na=False)]
            if not alias_match.empty:
                return alias_match.iloc[0]
    
    # Partial word matching
    words = [word for word in normalized.split() if len(word) > 2]
    for word in words:
        partial_match = price_df[
            price_df['Name_Lower'].str.contains(word, na=False) | 
            price_df['Name_Lower'].apply(lambda x: word in x.split() if pd.notna(x) else False)
        ]
        if not partial_match.empty:
            return partial_match.iloc[0]
    
    return None

def extract_all_ingredients(recipe_df: pd.DataFrame) -> List[Dict]:
    """Extract all ingredients from all recipes"""
    all_ingredients = []
    
    print("Extracting ingredients from recipes...")
    
    for idx, recipe in recipe_df.iterrows():
        if pd.isna(recipe['ingredients']) or not recipe['ingredients']:
            continue
            
        recipe_title = recipe['title']
        ingredient_lines = [line.strip() for line in recipe['ingredients'].split('|') if line.strip()]
        
        for ingredient_line in ingredient_lines:
            parsed = parse_ingredient_line(ingredient_line)
            
            if parsed:
                all_ingredients.append({
                    'recipe_title': recipe_title,
                    'original_line': ingredient_line,
                    'ingredient_name': parsed['ingredient'],
                    'quantity': parsed['quantity'],
                    'unit': parsed['unit']
                })
        
        # Show progress
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(recipe_df)} recipes...")
    
    return all_ingredients

def analyze_missing_ingredients(recipe_df: pd.DataFrame, price_df: pd.DataFrame):
    """Comprehensive analysis of missing ingredients"""
    
    print("=== MISSING INGREDIENTS ANALYSIS ===\n")
    
    # Extract all ingredients
    all_ingredients = extract_all_ingredients(recipe_df)
    print(f"Total ingredient entries found: {len(all_ingredients)}")
    
    # Categorize ingredients
    found_ingredients = []
    missing_ingredients = []
    
    print("\nAnalyzing ingredient matches...")
    
    for ingredient_data in all_ingredients:
        ingredient_name = ingredient_data['ingredient_name']
        match = find_best_match(ingredient_name, price_df)
        
        if match is not None:
            found_ingredients.append({
                **ingredient_data,
                'matched_item': match['Name'],
                'matched_price': match['Average Price']
            })
        else:
            missing_ingredients.append(ingredient_data)
    
    # Statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total ingredient entries: {len(all_ingredients)}")
    print(f"Found in price database: {len(found_ingredients)} ({len(found_ingredients)/len(all_ingredients)*100:.1f}%)")
    print(f"Missing from database: {len(missing_ingredients)} ({len(missing_ingredients)/len(all_ingredients)*100:.1f}%)")
    
    # Count unique missing ingredients
    unique_missing = set(item['ingredient_name'] for item in missing_ingredients)
    unique_found = set(item['ingredient_name'] for item in found_ingredients)
    
    print(f"\nUnique ingredients:")
    print(f"Found: {len(unique_found)} unique ingredients")
    print(f"Missing: {len(unique_missing)} unique ingredients")
    
    # Most common missing ingredients
    missing_counter = Counter(item['ingredient_name'] for item in missing_ingredients)
    
    print(f"\n=== TOP 20 MISSING INGREDIENTS ===")
    print("(Ingredient name - Number of times used in recipes)")
    for ingredient, count in missing_counter.most_common(20):
        print(f"{count:3d}x - {ingredient}")
    
    # Show all unique missing ingredients
    print(f"\n=== ALL UNIQUE MISSING INGREDIENTS ({len(unique_missing)} total) ===")
    sorted_missing = sorted(unique_missing)
    
    for i, ingredient in enumerate(sorted_missing, 1):
        usage_count = missing_counter[ingredient]
        print(f"{i:3d}. {ingredient} (used {usage_count}x)")
    
    # Recipe impact analysis
    recipe_analysis = {}
    for item in missing_ingredients:
        recipe_title = item['recipe_title']
        if recipe_title not in recipe_analysis:
            recipe_analysis[recipe_title] = {
                'missing_count': 0,
                'missing_ingredients': set(),
                'total_ingredients': 0
            }
        recipe_analysis[recipe_title]['missing_count'] += 1
        recipe_analysis[recipe_title]['missing_ingredients'].add(item['ingredient_name'])
    
    # Add total ingredient counts
    for item in all_ingredients:
        recipe_title = item['recipe_title']
        if recipe_title in recipe_analysis:
            recipe_analysis[recipe_title]['total_ingredients'] += 1
    
    # Show recipes most affected by missing ingredients
    print(f"\n=== RECIPES MOST AFFECTED BY MISSING INGREDIENTS ===")
    affected_recipes = [(title, data) for title, data in recipe_analysis.items()]
    affected_recipes.sort(key=lambda x: x[1]['missing_count'], reverse=True)
    
    for i, (recipe_title, data) in enumerate(affected_recipes[:10], 1):
        missing_pct = (data['missing_count'] / data['total_ingredients']) * 100 if data['total_ingredients'] > 0 else 0
        print(f"{i:2d}. {recipe_title}")
        print(f"     Missing: {data['missing_count']}/{data['total_ingredients']} ingredients ({missing_pct:.1f}%)")
        print(f"     Missing items: {', '.join(list(data['missing_ingredients'])[:3])}{'...' if len(data['missing_ingredients']) > 3 else ''}")
        print()
    
    # Generate CSV reports
    print("=== GENERATING REPORTS ===")
    
    # 1. Missing ingredients report
    missing_df = pd.DataFrame([
        {
            'ingredient_name': ingredient,
            'usage_count': count,
            'percentage_of_missing': (count / len(missing_ingredients)) * 100
        }
        for ingredient, count in missing_counter.most_common()
    ])
    
    missing_df.to_csv('missing_ingredients_report.csv', index=False)
    print("✓ Saved: missing_ingredients_report.csv")
    
    # 2. Detailed missing ingredients with context
    detailed_missing_df = pd.DataFrame(missing_ingredients)
    detailed_missing_df.to_csv('detailed_missing_ingredients.csv', index=False)
    print("✓ Saved: detailed_missing_ingredients.csv")
    
    # 3. Recipe impact analysis
    recipe_impact_data = []
    for recipe_title, data in recipe_analysis.items():
        missing_pct = (data['missing_count'] / data['total_ingredients']) * 100 if data['total_ingredients'] > 0 else 0
        recipe_impact_data.append({
            'recipe_title': recipe_title,
            'total_ingredients': data['total_ingredients'],
            'missing_ingredients_count': data['missing_count'],
            'missing_percentage': round(missing_pct, 1),
            'missing_ingredients_list': ', '.join(sorted(data['missing_ingredients']))
        })
    
    recipe_impact_df = pd.DataFrame(recipe_impact_data)
    recipe_impact_df = recipe_impact_df.sort_values('missing_ingredients_count', ascending=False)
    recipe_impact_df.to_csv('recipe_impact_analysis.csv', index=False)
    print("✓ Saved: recipe_impact_analysis.csv")
    
    # 4. Suggested price list template
    suggested_prices = []
    for ingredient, count in missing_counter.most_common():
        suggested_prices.append({
            'Name': ingredient.title(),
            'Unit': '1 kg',  # Default unit
            'Average Price': '',  # Empty for manual entry
            'Min Price': '',     # Empty for manual entry
            'Usage Count': count,
            'Notes': 'Add realistic Sri Lankan market price'
        })
    
    suggested_df = pd.DataFrame(suggested_prices)
    suggested_df.to_csv('suggested_price_additions.csv', index=False)
    print("✓ Saved: suggested_price_additions.csv")
    
    print(f"\n=== COMPLETION ===")
    print("Analysis complete! Check the generated CSV files for detailed reports.")
    print("\nNext steps:")
    print("1. Review 'missing_ingredients_report.csv' for most important missing items")
    print("2. Use 'suggested_price_additions.csv' as a template to add prices")
    print("3. Add the new prices to your original extracted_prices2.csv")
    print("4. Re-run the recipe cost calculator for improved accuracy")

def load_data():
    """Load recipe and price data from CSV files"""
    print("Loading data...")
    
    recipe_df = pd.read_csv('recipes_normalized_robust_nutrition.csv')
    price_df = pd.read_csv('extracted_prices2.csv')
    
    # Clean price data
    price_df = price_df.dropna(subset=['Name', 'Average Price'])
    price_df['Name_Lower'] = price_df['Name'].str.lower().str.strip()
    price_df = price_df[price_df['Average Price'] > 0]
    
    print(f"Loaded {len(recipe_df)} recipes and {len(price_df)} price items")
    return recipe_df, price_df

def main():
    """Main function"""
    print("=== Missing Ingredients Analyzer ===")
    print("This tool will identify all ingredients missing from your price database.\n")
    
    # Load data
    recipe_df, price_df = load_data()
    
    # Run analysis
    analyze_missing_ingredients(recipe_df, price_df)
    
    return True

if __name__ == "__main__":
    main()