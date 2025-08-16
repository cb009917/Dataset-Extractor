import pandas as pd
import numpy as np
import re
import logging
import requests
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from fractions import Fraction
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParsedIngredient(NamedTuple):
    """Structured representation of a parsed ingredient"""
    quantity: float
    unit: str
    ingredient_name: str
    grams: float
    original_text: str

@dataclass
class NutritionInfo:
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0

class RobustIngredientParser:
    """
    Robust ingredient parser that handles all edge cases and converts everything to grams
    """
    
    def __init__(self):
        # Comprehensive unit conversion table (all to grams)
        self.unit_conversions = {
            # WEIGHT UNITS (direct conversion)
            'g': 1.0,
            'gram': 1.0, 
            'grams': 1.0,
            'gr': 1.0,
            'gm': 1.0,
            'kg': 1000.0,
            'kilogram': 1000.0,
            'kilograms': 1000.0,
            'oz': 28.35,
            'ounce': 28.35,
            'ounces': 28.35,
            'lb': 453.6,
            'lbs': 453.6,
            'pound': 453.6,
            'pounds': 453.6,
            'mg': 0.001,
            'milligram': 0.001,
            'milligrams': 0.001,
            
            # VOLUME UNITS (ingredient-specific densities) - CORRECTED FOR COOKED PORTIONS
            'cup': {
                'flour': 125, 'rice flour': 125, 'wheat flour': 125, 'all purpose flour': 125,
                'rice': 150, 'basmati rice': 150, 'white rice': 150, 'brown rice': 150,  # REDUCED from 185 (cooked rice is lighter)
                'sugar': 200, 'white sugar': 200, 'brown sugar': 220,
                'milk': 240, 'coconut milk': 240, 'whole milk': 240,
                'water': 240, 'coconut water': 240,
                'oil': 220, 'coconut oil': 220, 'vegetable oil': 220, 'olive oil': 220,
                'butter': 227, 'melted butter': 227,
                'coconut': 80, 'shredded coconut': 80, 'desiccated coconut': 93,
                'lentils': 180, 'dal': 180, 'dhal': 180, 'split peas': 180,  # REDUCED (cooked)
                'breadcrumbs': 60, 'panko': 50,
                'default': 140  # REDUCED from 150
            },
            'tbsp': {
                'oil': 14, 'coconut oil': 14, 'vegetable oil': 14, 'olive oil': 14,
                'butter': 14, 'melted butter': 14,
                'flour': 8, 'rice flour': 8,
                'sugar': 12, 'honey': 21, 'syrup': 20,
                'milk': 15, 'water': 15, 'coconut milk': 15,
                'soy sauce': 16, 'vinegar': 15,
                'default': 15
            },
            'tablespoon': 'tbsp',  # Redirect to tbsp
            'tablespoons': 'tbsp',
            'tsp': {
                'salt': 6, 'sea salt': 6, 'table salt': 6,
                'sugar': 4, 'brown sugar': 4,
                'oil': 5, 'coconut oil': 5,
                'spices': 2, 'curry powder': 2, 'turmeric': 2, 'cumin': 2,
                'seeds': 3, 'mustard seeds': 3, 'cumin seeds': 3,
                'vanilla': 4, 'vanilla extract': 4,
                'baking powder': 3, 'baking soda': 4,
                'default': 5
            },
            'teaspoon': 'tsp',  # Redirect to tsp
            'teaspoons': 'tsp',
            'ml': 1.0,  # Assume water density for liquids
            'milliliter': 1.0,
            'milliliters': 1.0,
            'l': 1000.0,
            'liter': 1000.0,
            'liters': 1000.0,
            'litre': 1000.0,
            'litres': 1000.0,
            
            # COUNT UNITS (average weights) - SOME CORRECTIONS
            'piece': {
                'onion': 150, 'medium onion': 150, 'large onion': 200, 'small onion': 100,
                'potato': 150, 'medium potato': 150, 'large potato': 200, 'small potato': 100,
                'tomato': 123, 'medium tomato': 123, 'large tomato': 180, 'small tomato': 80,
                'egg': 50, 'large egg': 60, 'medium egg': 50, 'small egg': 40,
                'chicken breast': 150, 'chicken thigh': 120,  # REDUCED - more realistic portion sizes
                'fish fillet': 120, 'fish': 120,  # REDUCED
                'chili': 5, 'chilli': 5, 'green chili': 5, 'red chili': 5,
                'garlic': 3, 'garlic clove': 3,
                'ginger': 10, 'piece ginger': 10,
                'bread': 25, 'slice bread': 25,
                'default': 50
            },
            'pieces': 'piece',
            'clove': {
                'garlic': 3, 'default': 3
            },
            'cloves': 'clove',
            'sprig': {
                'curry leaves': 2, 'mint': 2, 'cilantro': 2, 'parsley': 2,
                'thyme': 1, 'rosemary': 1, 'default': 2
            },
            'sprigs': 'sprig',
            'leaf': {
                'bay': 0.1, 'bay leaf': 0.1,
                'curry': 0.5, 'curry leaf': 0.5,
                'pandan': 1, 'pandan leaf': 1,
                'default': 0.5
            },
            'leaves': 'leaf',
            'stick': {
                'cinnamon': 2, 'cinnamon stick': 2,
                'celery': 40, 'celery stick': 40,
                'default': 5
            },
            'sticks': 'stick',
            
            # DESCRIPTIVE UNITS (rough estimates)
            'pinch': 0.5,  # About 1/8 teaspoon
            'pinches': 0.5,
            'dash': 0.6,   # Slightly more than a pinch
            'dashes': 0.6,
            'handful': 30,  # Rough estimate
            'handfuls': 30,
            'bunch': 100,   # For herbs/greens
            'bunches': 100,
            'slice': {
                'bread': 25, 'cheese': 20, 'tomato': 15, 'onion': 10,
                'ginger': 2, 'lemon': 5, 'lime': 3, 'default': 15
            },
            'slices': 'slice',
            'can': {
                'coconut milk': 400, 'tomatoes': 400, 'beans': 400,
                'tuna': 150, 'sardines': 120, 'default': 400
            },
            'cans': 'can',
            'packet': {
                'instant noodles': 85, 'spices': 10, 'seasoning': 10,
                'default': 100
            },
            'packets': 'packet'
        }
        
        # Quantity words to numbers
        self.quantity_words = {
            'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'half': 0.5, 'quarter': 0.25, 'third': 0.33,
            'some': 1, 'few': 3, 'several': 4, 'couple': 2
        }
        
        # Common ingredient name cleanups
        self.ingredient_cleanups = {
            'salted butter': 'butter',
            'unsalted butter': 'butter', 
            'melted butter': 'butter',
            'extra virgin olive oil': 'olive oil',
            'virgin coconut oil': 'coconut oil',
            'fresh ginger': 'ginger',
            'fresh garlic': 'garlic',
            'green chilli': 'green chili',
            'red chilli': 'red chili',
            'medium onion': 'onion',
            'large onion': 'onion',
            'small onion': 'onion'
        }
    
    def parse_fraction(self, fraction_str: str) -> float:
        """Convert fraction string to decimal"""
        try:
            if '/' in fraction_str:
                return float(Fraction(fraction_str))
            return float(fraction_str)
        except:
            return 1.0
    
    def parse_quantity(self, text: str) -> Tuple[float, str]:
        """
        Extract quantity from text, return (quantity, remaining_text)
        """
        text = text.strip().lower()
        
        # Pattern 1: Mixed numbers (1 1/2, 2 3/4)
        mixed_match = re.match(r'^(\d+)\s+(\d+/\d+)\s+(.*)$', text)
        if mixed_match:
            whole, frac, remaining = mixed_match.groups()
            quantity = int(whole) + self.parse_fraction(frac)
            return quantity, remaining.strip()
        
        # Pattern 2: Simple fractions (1/2, 3/4)
        frac_match = re.match(r'^(\d+/\d+)\s+(.*)$', text)
        if frac_match:
            frac, remaining = frac_match.groups()
            quantity = self.parse_fraction(frac)
            return quantity, remaining.strip()
        
        # Pattern 3: Decimal numbers (2.5, 1.25)
        decimal_match = re.match(r'^(\d+\.\d+)\s+(.*)$', text)
        if decimal_match:
            num, remaining = decimal_match.groups()
            quantity = float(num)
            return quantity, remaining.strip()
        
        # Pattern 4: Whole numbers (2, 10)
        whole_match = re.match(r'^(\d+)\s+(.*)$', text)
        if whole_match:
            num, remaining = whole_match.groups()
            quantity = int(num)
            return quantity, remaining.strip()
        
        # Pattern 5: Quantity words (a, half, few)
        for word, value in self.quantity_words.items():
            if text.startswith(word + ' '):
                remaining = text[len(word):].strip()
                return value, remaining
        
        # No quantity found
        return 1.0, text
    
    def parse_unit(self, text: str) -> Tuple[str, str]:
        """
        Extract unit from text, return (unit, remaining_text)
        """
        text = text.strip().lower()
        
        # Try to match known units at the beginning
        for unit in self.unit_conversions.keys():
            if isinstance(self.unit_conversions[unit], dict):
                continue  # Skip compound units for now
            
            # Check if text starts with this unit
            patterns = [
                f'^{unit}\\b\\s*(.*)',  # Exact word boundary
                f'^{unit}s\\b\\s*(.*)', # Plural form
                f'^{unit}\\.\\s*(.*)'   # With period
            ]
            
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    remaining = match.group(1).strip()
                    return unit, remaining
        
        # Check compound units (cup, tbsp, tsp, etc.)
        compound_units = ['cup', 'cups', 'tbsp', 'tablespoon', 'tablespoons', 
                         'tsp', 'teaspoon', 'teaspoons', 'piece', 'pieces',
                         'clove', 'cloves', 'sprig', 'sprigs', 'leaf', 'leaves',
                         'stick', 'sticks', 'slice', 'slices', 'can', 'cans',
                         'packet', 'packets', 'pinch', 'pinches', 'dash', 'dashes']
        
        for unit in compound_units:
            patterns = [
                f'^{unit}\\b\\s*(.*)',
                f'^{unit}\\.\\s*(.*)'
            ]
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    remaining = match.group(1).strip()
                    return unit.rstrip('s'), remaining  # Remove plural
        
        # Special cases for "of" (pinch of salt, cup of rice)
        of_match = re.match(r'^(.*?)\s+of\s+(.*)$', text)
        if of_match:
            potential_unit, remaining = of_match.groups()
            potential_unit = potential_unit.strip()
            if potential_unit in compound_units:
                return potential_unit.rstrip('s'), remaining.strip()
        
        # No unit found, assume "piece"
        return 'piece', text
    
    def convert_to_grams(self, quantity: float, unit: str, ingredient_name: str) -> float:
        """
        Convert any quantity/unit to grams using comprehensive conversion table
        """
        try:
            unit = unit.lower().strip()
            ingredient_lower = ingredient_name.lower().strip()
            
            # Handle redirects (tablespoon -> tbsp)
            if unit in self.unit_conversions and isinstance(self.unit_conversions[unit], str):
                unit = self.unit_conversions[unit]
            
            # Direct conversion units
            if unit in self.unit_conversions and isinstance(self.unit_conversions[unit], (int, float)):
                return quantity * self.unit_conversions[unit]
            
            # Ingredient-specific conversions
            if unit in self.unit_conversions and isinstance(self.unit_conversions[unit], dict):
                conversion_table = self.unit_conversions[unit]
                
                # Try to find specific ingredient match
                for ingredient_key, weight in conversion_table.items():
                    if ingredient_key == 'default':
                        continue
                    if ingredient_key in ingredient_lower:
                        return quantity * weight
                
                # Use default if no specific match
                return quantity * conversion_table.get('default', 50)
            
            # Fallback for unknown units
            print(f"   WARNING: Unknown unit '{unit}', using 10g estimate")
            return quantity * 10
            
        except Exception as e:
            print(f"   Conversion error: {e}")
            return 10  # Safe fallback
    
    def clean_ingredient_name(self, name: str) -> str:
        """Clean and standardize ingredient name"""
        name = name.lower().strip()
        
        # Remove common adjectives and descriptors
        name = re.sub(r'\b(fresh|dried|raw|cooked|boiled|fried|steamed|chopped|sliced|diced|minced|grated|shredded)\b', '', name)
        
        # Remove parenthetical information
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Apply specific cleanups
        for old_name, new_name in self.ingredient_cleanups.items():
            if old_name in name:
                name = name.replace(old_name, new_name)
        
        return name
    
    def parse_ingredient(self, ingredient_text: str) -> ParsedIngredient:
        """
        Main parsing function: converts any ingredient string to standardized format
        """
        try:
            original_text = ingredient_text.strip()
            
            # Step 1: Extract quantity
            quantity, remaining_text = self.parse_quantity(original_text)
            
            # Step 2: Extract unit
            unit, ingredient_name = self.parse_unit(remaining_text)
            
            # Step 3: Clean ingredient name
            clean_name = self.clean_ingredient_name(ingredient_name)
            
            # Step 4: Convert to grams
            grams = self.convert_to_grams(quantity, unit, clean_name)
            
            return ParsedIngredient(
                quantity=quantity,
                unit=unit,
                ingredient_name=clean_name,
                grams=grams,
                original_text=original_text
            )
            
        except Exception as e:
            print(f"   Parsing error for '{ingredient_text}': {e}")
            # Return safe fallback
            return ParsedIngredient(
                quantity=1.0,
                unit='piece',
                ingredient_name=ingredient_text.lower().strip(),
                grams=50.0,
                original_text=ingredient_text
            )

class HybridNutritionCalculator:
    """
    Nutrition calculator using robust parsing + hybrid database approach
    """
    
    def __init__(self, nutrition_db_file: str = 'ingredient-dataset_nutrition.xlsx', 
                 usda_api_key: str = "fz7YpdJ3tR0PfazCyXr4mJYgyDGir5swSwIANOJz"):
        
        self.parser = RobustIngredientParser()
        self.nutrition_db_file = nutrition_db_file
        self.usda_api_key = usda_api_key
        self.usda_base_url = "https://api.nal.usda.gov/fdc/v1"
        
        # Caches
        self.usda_food_cache = {}
        self.usda_nutrition_cache = {}
        self.nutrition_database = {}
        
        self.load_nutrition_database()
    
    def load_nutrition_database(self):
        """Load Sri Lankan nutrition database"""
        try:
            logger.info(f"Loading Sri Lankan nutrition database from {self.nutrition_db_file}")
            df = pd.read_excel(self.nutrition_db_file)
            logger.info(f"Loaded {len(df)} ingredients from database")
            
            for _, row in df.iterrows():
                name = str(row.get('Name', '')).strip()
                if not name or name == 'nan':
                    continue
                
                try:
                    # Clean nutrition values
                    energy_str = str(row.get('Energy', '0')).strip()
                    energy_kj = float(re.sub(r'[^\d.]', '', energy_str)) if energy_str != 'nan' else 0
                    
                    carbs_str = str(row.get('Carbohydrates', '0')).strip()
                    carbs = float(re.sub(r'[^\d.]', '', carbs_str)) if carbs_str != 'nan' else 0
                    
                    fat_str = str(row.get('Fat', '0')).strip()
                    fat = float(re.sub(r'[^\d.]', '', fat_str)) if fat_str != 'nan' else 0
                    
                    # NEW: Try to get protein directly if available
                    protein_str = str(row.get('Protein', '0')).strip()
                    protein = float(re.sub(r'[^\d.]', '', protein_str)) if protein_str != 'nan' else 0
                    
                except (ValueError, TypeError):
                    energy_kj, carbs, fat, protein = 0, 0, 0, 0
                
                # Convert kJ to kcal
                calories = round(energy_kj * 0.239, 1)
                
                # IMPROVED: Only estimate protein if not provided directly
                if protein == 0:
                    carb_calories = carbs * 4
                    fat_calories = fat * 9
                    remaining_calories = max(0, calories - carb_calories - fat_calories)
                    protein = max(0, round(remaining_calories / 4, 1))
                    # Limit unrealistic protein values
                    if protein > calories / 4 * 0.8:  # Max 80% of calories from protein
                        protein = round(calories / 4 * 0.2, 1)  # Use 20% as reasonable estimate
                
                nutrition = NutritionInfo(calories=calories, protein=protein, carbs=carbs, fat=fat)
                
                # Store with multiple keys
                clean_name = name.lower().strip()
                self.nutrition_database[clean_name] = nutrition
                
                # Also store cleaned version
                parsed = self.parser.clean_ingredient_name(name)
                self.nutrition_database[parsed] = nutrition
            
            logger.info(f"Processed {len(self.nutrition_database)} nutrition entries")
            
        except Exception as e:
            logger.error(f"Error loading nutrition database: {e}")
            raise
    
    def find_nutrition_data(self, ingredient_name: str) -> Optional[NutritionInfo]:
        """Find nutrition data using hybrid approach"""
        
        # Try Sri Lankan database first
        if ingredient_name in self.nutrition_database:
            print(f"   Found in Sri Lankan DB: '{ingredient_name}'")
            return self.nutrition_database[ingredient_name]
        
        # Try fuzzy matching in Sri Lankan DB
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.0
        
        for db_name in self.nutrition_database.keys():
            score = SequenceMatcher(None, ingredient_name, db_name).ratio()
            if score > best_score and score > 0.8:  # INCREASED threshold from 0.7 to 0.8
                best_score = score
                best_match = db_name
        
        if best_match:
            print(f"   Found in Sri Lankan DB (fuzzy): '{ingredient_name}' -> '{best_match}' ({best_score:.2f})")
            return self.nutrition_database[best_match]
        
        # Fallback to USDA API
        print(f"   Trying USDA API for: '{ingredient_name}'")
        fdc_id = self.search_usda_food(ingredient_name)
        if fdc_id:
            nutrition = self.get_usda_nutrition(fdc_id)
            if nutrition:
                return nutrition
        
        print(f"   No nutrition data found for: '{ingredient_name}'")
        return None
    
    def search_usda_food(self, food_name: str) -> Optional[str]:
        """Search USDA database"""
        try:
            if food_name in self.usda_food_cache:
                return self.usda_food_cache[food_name]
            
            search_term = re.sub(r'\b(fresh|dried|raw|cooked)\b', '', food_name).strip()
            
            url = f"{self.usda_base_url}/foods/search"
            params = {
                'query': search_term,
                'dataType': ['Foundation', 'SR Legacy'],
                'pageSize': 3,
                'api_key': self.usda_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('foods'):
                fdc_id = str(data['foods'][0]['fdcId'])
                self.usda_food_cache[food_name] = fdc_id
                print(f"   Found in USDA: ID {fdc_id}")
                return fdc_id
            
            return None
            
        except Exception as e:
            print(f"   USDA search error: {e}")
            return None
    
    def get_usda_nutrition(self, fdc_id: str) -> Optional[NutritionInfo]:
        """Get USDA nutrition data"""
        try:
            if fdc_id in self.usda_nutrition_cache:
                return self.usda_nutrition_cache[fdc_id]
            
            url = f"{self.usda_base_url}/food/{fdc_id}"
            params = {'api_key': self.usda_api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            nutrition = NutritionInfo()
            
            if 'foodNutrients' in data:
                for nutrient in data['foodNutrients']:
                    nutrient_id = nutrient.get('nutrient', {}).get('id')
                    amount = nutrient.get('amount', 0)
                    
                    if nutrient_id == 1008:  # Energy (kcal)
                        nutrition.calories = amount
                    elif nutrient_id == 1003:  # Protein
                        nutrition.protein = amount
                    elif nutrient_id == 1005:  # Carbohydrates
                        nutrition.carbs = amount
                    elif nutrient_id == 1004:  # Fat
                        nutrition.fat = amount
            
            self.usda_nutrition_cache[fdc_id] = nutrition
            print(f"   USDA nutrition: {nutrition.calories:.1f} kcal")
            return nutrition
            
        except Exception as e:
            print(f"   USDA nutrition error: {e}")
            return None
    
    def calculate_ingredient_nutrition(self, ingredient_text: str) -> NutritionInfo:
        """Calculate nutrition with robust parsing"""
        try:
            print(f"\nPARSING: '{ingredient_text}'")
            
            # Parse ingredient using robust parser
            parsed = self.parser.parse_ingredient(ingredient_text)
            
            print(f"   Quantity: {parsed.quantity}")
            print(f"   Unit: {parsed.unit}")
            print(f"   Ingredient: '{parsed.ingredient_name}'")
            print(f"   Grams: {parsed.grams:.1f}g")
            
            # Find nutrition data
            nutrition_per_100g = self.find_nutrition_data(parsed.ingredient_name)
            
            if not nutrition_per_100g:
                print(f"   No nutrition data found")
                return NutritionInfo()
            
            print(f"   Nutrition per 100g: {nutrition_per_100g.calories:.1f} kcal")
            
            # Scale to actual grams
            scale_factor = parsed.grams / 100.0
            print(f"   Scale factor: {scale_factor:.3f}")
            
            scaled_nutrition = NutritionInfo(
                calories=nutrition_per_100g.calories * scale_factor,
                protein=nutrition_per_100g.protein * scale_factor,
                carbs=nutrition_per_100g.carbs * scale_factor,
                fat=nutrition_per_100g.fat * scale_factor
            )
            
            print(f"   Final: {scaled_nutrition.calories:.1f} kcal")
            
            # Add small delay for API rate limiting
            time.sleep(0.1)
            
            return scaled_nutrition
            
        except Exception as e:
            print(f"   Error: {e}")
            return NutritionInfo()
    
    def calculate_recipe_nutrition(self, ingredients_str: str) -> NutritionInfo:
        """Calculate recipe nutrition"""
        try:
            if not ingredients_str or pd.isna(ingredients_str):
                return NutritionInfo()
            
            ingredients = [ing.strip() for ing in ingredients_str.split('|') if ing.strip()]
            
            print(f"\nRECIPE: Processing {len(ingredients)} ingredients")
            print("=" * 60)
            
            total_nutrition = NutritionInfo()
            
            for i, ingredient in enumerate(ingredients, 1):
                # Skip obvious instructions
                if len(ingredient) > 100 or any(word in ingredient.lower() for word in 
                                               ['enjoy', 'serve', 'garnish', 'taste', 'optional', 'hot with']):
                    print(f"SKIPPING {i}: '{ingredient}' (instruction)")
                    continue
                
                print(f"\n--- INGREDIENT {i}/{len(ingredients)} ---")
                ingredient_nutrition = self.calculate_ingredient_nutrition(ingredient)
                
                total_nutrition.calories += ingredient_nutrition.calories
                total_nutrition.protein += ingredient_nutrition.protein
                total_nutrition.carbs += ingredient_nutrition.carbs
                total_nutrition.fat += ingredient_nutrition.fat
                
                print(f"RUNNING TOTAL: {total_nutrition.calories:.1f} kcal")
            
            print(f"\nFINAL TOTAL: {total_nutrition.calories:.1f} kcal")
            print("=" * 60)
            
            return total_nutrition
            
        except Exception as e:
            logger.error(f"Error calculating recipe nutrition: {e}")
            return NutritionInfo()
    
    def process_csv_file(self, input_file: str, output_file: str = None,
                        ingredients_col: str = 'ingredients_per_person'):
        """Process CSV file"""
        try:
            logger.info(f"Loading recipes from {input_file}")
            df = pd.read_csv(input_file, encoding='utf-8')
            
            if ingredients_col not in df.columns:
                logger.error(f"Column '{ingredients_col}' not found!")
                logger.info(f"Available columns: {list(df.columns)}")
                return
            
            logger.info(f"Loaded {len(df)} recipes")
            
            # Add nutrition columns
            df['calories'] = 0.0
            df['protein_g'] = 0.0
            df['carbs_g'] = 0.0
            df['fat_g'] = 0.0
            
            # Process each recipe
            for idx, row in df.iterrows():
                try:
                    if (idx + 1) % 5 == 0:
                        logger.info(f"Progress: {idx+1}/{len(df)} recipes processed")
                    
                    ingredients = row.get(ingredients_col, '')
                    recipe_title = row.get('title', f'Recipe {idx+1}')
                    
                    nutrition = self.calculate_recipe_nutrition(ingredients)
                    
                    df.at[idx, 'calories'] = round(nutrition.calories, 1)
                    df.at[idx, 'protein_g'] = round(nutrition.protein, 1)
                    df.at[idx, 'carbs_g'] = round(nutrition.carbs, 1)
                    df.at[idx, 'fat_g'] = round(nutrition.fat, 1)
                    
                    logger.info(f"{idx+1}. '{recipe_title[:30]}...' -> {nutrition.calories:.0f} kcal")
                    
                except Exception as e:
                    logger.error(f"Error processing recipe {idx+1}: {e}")
                    continue
            
            # Save results
            if output_file is None:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}_corrected_nutrition.csv"
            
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_file}")
            
            # Show summary
            self.show_summary(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def show_summary(self, df: pd.DataFrame):
        """Display summary"""
        print("\n" + "="*70)
        print("CORRECTED INGREDIENT PARSING + NUTRITION CALCULATION")
        print("="*70)
        
        avg_calories = df['calories'].mean()
        avg_protein = df['protein_g'].mean()
        avg_carbs = df['carbs_g'].mean()
        avg_fat = df['fat_g'].mean()
        
        print(f"\nAVERAGE NUTRITION PER RECIPE (single serving):")
        print(f"  Calories: {avg_calories:.1f} kcal")
        print(f"  Protein:  {avg_protein:.1f} g")
        print(f"  Carbs:    {avg_carbs:.1f} g")
        print(f"  Fat:      {avg_fat:.1f} g")
        
        print(f"\nSAMPLE RECIPES:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            title = row.get('title', f'Recipe {i+1}')
            calories = row.get('calories', 0)
            protein = row.get('protein_g', 0)
            carbs = row.get('carbs_g', 0)
            fat = row.get('fat_g', 0)
            
            print(f"{i+1}. {title[:45]}...")
            print(f"   {calories:.0f} kcal | {protein:.1f}g protein | {carbs:.1f}g carbs | {fat:.1f}g fat")
        
        print("="*70)

# Main execution
if __name__ == "__main__":
    print("CORRECTED INGREDIENT PARSER + NUTRITION CALCULATOR")
    print("=" * 60)
    print("Fixed: Cooked rice portions, protein estimation, fuzzy matching")
    print("Fixed: More realistic portion sizes for meat and fish")
    print("Fixed: Higher fuzzy matching threshold")
    print()
    
    # Check database file
    db_file = 'ingredient-dataset_nutrition.xlsx'
    if not os.path.exists(db_file):
        print(f"ERROR: Nutrition database file not found: {db_file}")
        exit(1)
    
    try:
        # Initialize calculator
        calculator = HybridNutritionCalculator(db_file)
        
        # Get input file
        input_file = input("Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(input_file):
            print(f"ERROR: File not found: {input_file}")
            exit(1)
        
        # Get column name
        ingredients_col = input("Ingredients column name (default: 'ingredients_per_person'): ").strip() or 'ingredients_per_person'
        
        # Output file
        output_file = input("Output file path (press Enter for auto-generated): ").strip()
        if not output_file:
            output_file = None
        
        print(f"\nStarting corrected nutrition calculation...")
        print("Fixed parsing + Sri Lankan DB + USDA API")
        print("This may take a few minutes...")
        
        # Process CSV
        calculator.process_csv_file(
            input_file=input_file,
            output_file=output_file,
            ingredients_col=ingredients_col
        )
        
        print("\nCorrected nutrition calculation completed!")
        print("More accurate parsing and nutrition data!")
        
    except KeyboardInterrupt:
        print("\nCalculation interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure:")
        print("   - Internet connection for USDA API")
        print("   - Nutrition database Excel file is present")
        print("   - CSV has correct ingredients column")