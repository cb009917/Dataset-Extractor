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
from ingredient_normalizer import IngredientNormalizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing

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
        # Initialize the ingredient normalizer
        self.normalizer = IngredientNormalizer()
        
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
            
            # VOLUME UNITS (ingredient-specific densities)
            'cup': {
                'flour': 125, 'rice flour': 125, 'wheat flour': 125, 'all purpose flour': 125,
                'rice': 185, 'basmati rice': 185, 'white rice': 185, 'brown rice': 185,
                'sugar': 200, 'white sugar': 200, 'brown sugar': 220,
                'milk': 240, 'coconut milk': 240, 'whole milk': 240,
                'water': 240, 'coconut water': 240,
                'oil': 220, 'coconut oil': 220, 'vegetable oil': 220, 'olive oil': 220,
                'butter': 227, 'melted butter': 227,
                'coconut': 80, 'shredded coconut': 80, 'desiccated coconut': 93,
                'lentils': 200, 'dal': 200, 'dhal': 200, 'split peas': 200,
                'breadcrumbs': 60, 'panko': 50,
                'default': 150  # Conservative default
            },
            'tbsp': {
                # Liquids (using density * 15ml)
                'water': 15.0, 'coconut water': 15.0, 'broth': 15.0, 'vinegar': 15.0,
                'milk': 15.45, 'coconut milk': 15.45, 'yogurt': 15.45,
                'oil': 13.8, 'coconut oil': 13.8, 'vegetable oil': 13.8, 'olive oil': 13.8,
                'soy sauce': 18.0, 'fish sauce': 18.0,
                'honey': 21.3, 'treacle': 21.3, 'syrup': 20.0,
                # Solids
                'butter': 14, 'melted butter': 14,
                'flour': 8, 'rice flour': 8,
                'sugar': 12,
                'default': 15
            },
            'tablespoon': 'tbsp',  # Redirect to tbsp
            'tablespoons': 'tbsp',
            'tsp': {
                # Liquids (using density * 5ml)
                'water': 5.0, 'coconut water': 5.0, 'broth': 5.0, 'vinegar': 5.0,
                'milk': 5.15, 'coconut milk': 5.15, 'yogurt': 5.15,
                'oil': 4.6, 'coconut oil': 4.6, 'vegetable oil': 4.6, 'olive oil': 4.6,
                'soy sauce': 6.0, 'fish sauce': 6.0,
                'honey': 7.1, 'treacle': 7.1, 'syrup': 6.65,
                'vanilla': 5.0, 'vanilla extract': 5.0,
                # Solids
                'salt': 6, 'sea salt': 6, 'table salt': 6,
                'sugar': 4, 'brown sugar': 4,
                'spices': 2, 'curry powder': 2, 'turmeric': 2, 'cumin': 2,
                'seeds': 3, 'mustard seeds': 3, 'cumin seeds': 3,
                'baking powder': 3, 'baking soda': 4,
                'default': 5
            },
            'teaspoon': 'tsp',  # Redirect to tsp
            'teaspoons': 'tsp',
            # VOLUME UNITS WITH DENSITY SUPPORT (ml/l with ingredient-specific densities)
            'ml': {
                'water': 1.0, 'coconut water': 1.0, 'broth': 1.0, 'vinegar': 1.0,
                'milk': 1.03, 'coconut milk': 1.03, 'yogurt': 1.03,
                'oil': 0.92, 'coconut oil': 0.92, 'vegetable oil': 0.92, 'olive oil': 0.92,
                'soy sauce': 1.20, 'fish sauce': 1.20,
                'honey': 1.42, 'treacle': 1.42, 'syrup': 1.33,
                'default': 1.0
            },
            'milliliter': 'ml',
            'milliliters': 'ml',
            'l': {
                'water': 1000.0, 'coconut water': 1000.0, 'broth': 1000.0, 'vinegar': 1000.0,
                'milk': 1030.0, 'coconut milk': 1030.0, 'yogurt': 1030.0,
                'oil': 920.0, 'coconut oil': 920.0, 'vegetable oil': 920.0, 'olive oil': 920.0,
                'soy sauce': 1200.0, 'fish sauce': 1200.0,
                'honey': 1420.0, 'treacle': 1420.0, 'syrup': 1330.0,
                'default': 1000.0
            },
            'liter': 'l',
            'liters': 'l',
            'litre': 'l',
            'litres': 'l',
            
            # COUNT UNITS (average weights)
            'piece': {
                'onion': 150, 'medium onion': 150, 'large onion': 200, 'small onion': 100,
                'potato': 150, 'medium potato': 150, 'large potato': 200, 'small potato': 100,
                'tomato': 123, 'medium tomato': 123, 'large tomato': 180, 'small tomato': 80,
                'egg': 50, 'large egg': 60, 'medium egg': 50, 'small egg': 40,
                'chicken breast': 200, 'chicken thigh': 150,
                'fish fillet': 150, 'fish': 150,
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
    
    def _has_grams_prefix(self, ingredient_name: str) -> bool:
        """Check if ingredient name starts with grams-looking prefix"""
        grams_patterns = [
            r'^\d+\s*g\b',     # "500g"
            r'^\d+\s*gram',   # "500gram"
            r'^\d+\s*ml\b',   # "200ml"
            r'^gram',         # "grams something"
            r'^ml\b'          # "ml something"
        ]
        
        for pattern in grams_patterns:
            if re.match(pattern, ingredient_name.lower()):
                return True
        return False
    
    def _convert_grams_prefix_to_unit(self, text: str) -> str:
        """Convert grams-looking prefix in ingredient name to proper unit format"""
        # Pattern to extract number and unit from ingredient name
        pattern = r'^(.+?)\s+(\d+(?:\.\d+)?)\s*(g|gram|grams|ml|milliliter)\s+(.+)'
        match = re.match(pattern, text.lower())
        
        if match:
            prefix, number, unit, rest = match.groups()
            # Standardize unit
            if unit in ['gram', 'grams']:
                unit = 'g'
            elif unit in ['milliliter']:
                unit = 'ml'
            return f"{number} {unit} {rest.strip()}"
        
        # Also handle cases where the number is at the start of ingredient name
        pattern2 = r'^(\d+(?:\.\d+)?)\s*(g|gram|grams|ml|milliliter)\s+(.+)'
        match2 = re.match(pattern2, text.lower())
        if match2:
            number, unit, rest = match2.groups()
            if unit in ['gram', 'grams']:
                unit = 'g'
            elif unit in ['milliliter']:
                unit = 'ml'
            return f"{number} {unit} {rest.strip()}"
            
        return text
    
    def parse_ingredient(self, ingredient_text: str, verbose: bool = True) -> ParsedIngredient:
        """
        Main parsing function: converts any ingredient string to standardized format
        Now includes pre-normalization step
        """
        try:
            original_text = ingredient_text.strip()
            
            # STEP 0: Normalize the ingredient text using the normalizer
            normalized_text, should_skip = self.normalizer.normalize_token(original_text)
            
            if should_skip:
                print(f"   SKIPPED (as needed/to taste): '{original_text}'")
                return ParsedIngredient(
                    quantity=0.0,
                    unit='skip',
                    ingredient_name=original_text.lower().strip(),
                    grams=0.0,
                    original_text=original_text
                )
            
            print(f"   Normalized: '{original_text}' -> '{normalized_text}'")
            
            # Step 1: Extract quantity from normalized text
            quantity, remaining_text = self.parse_quantity(normalized_text)
            
            # Step 2: Extract unit
            unit, ingredient_name = self.parse_unit(remaining_text)
            
            # Step 3: Clean ingredient name
            clean_name = self.clean_ingredient_name(ingredient_name)
            
            # Step 4: Convert to grams
            grams = self.convert_to_grams(quantity, unit, clean_name)
            
            # Step 4.5: Re-parse fallback for grams-looking prefixes
            if unit == 'piece' and self._has_grams_prefix(clean_name):
                if verbose:
                    print(f"   Re-parsing grams-looking ingredient: '{clean_name}'")
                # Try to re-normalize treating as grams/ml
                reparsed_text = self._convert_grams_prefix_to_unit(normalized_text)
                if reparsed_text != normalized_text:
                    if verbose:
                        print(f"   Re-normalized: '{normalized_text}' -> '{reparsed_text}'")
                    # Re-run the parsing with the new text
                    quantity, remaining_text = self.parse_quantity(reparsed_text)
                    unit, ingredient_name = self.parse_unit(remaining_text)
                    clean_name = self.clean_ingredient_name(ingredient_name)
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
        
        # Thread safety
        self.cache_lock = Lock()
        
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
                    
                except (ValueError, TypeError):
                    energy_kj, carbs, fat = 0, 0, 0
                
                # Convert kJ to kcal
                calories = round(energy_kj * 0.239, 1)
                
                # Estimate protein
                carb_calories = carbs * 4
                fat_calories = fat * 9
                remaining_calories = max(0, calories - carb_calories - fat_calories)
                protein = max(0, round(remaining_calories / 4, 1))
                
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
    
    def find_nutrition_data(self, ingredient_name: str, verbose: bool = True) -> Optional[NutritionInfo]:
        """Find nutrition data using hybrid approach"""
        
        # Try Sri Lankan database first
        if ingredient_name in self.nutrition_database:
            if verbose:
                print(f"   Found in Sri Lankan DB: '{ingredient_name}'")
            return self.nutrition_database[ingredient_name]
        
        # Try fuzzy matching in Sri Lankan DB
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.0
        
        for db_name in self.nutrition_database.keys():
            score = SequenceMatcher(None, ingredient_name, db_name).ratio()
            if score > best_score and score > 0.7:  # High threshold
                best_score = score
                best_match = db_name
        
        if best_match:
            if verbose:
                print(f"   Found in Sri Lankan DB (fuzzy): '{ingredient_name}' -> '{best_match}' ({best_score:.2f})")
            return self.nutrition_database[best_match]
        
        # Fallback to USDA API
        if verbose:
            print(f"   Trying USDA API for: '{ingredient_name}'")
        fdc_id = self.search_usda_food(ingredient_name)
        if fdc_id:
            nutrition = self.get_usda_nutrition(fdc_id)
            if nutrition:
                return nutrition
        
        if verbose:
            print(f"   No nutrition data found for: '{ingredient_name}'")
        return None
    
    def search_usda_food(self, food_name: str) -> Optional[str]:
        """Search USDA database (thread-safe)"""
        try:
            with self.cache_lock:
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
                with self.cache_lock:
                    self.usda_food_cache[food_name] = fdc_id
                print(f"   Found in USDA: ID {fdc_id}")
                return fdc_id
            
            return None
            
        except Exception as e:
            print(f"   USDA search error: {e}")
            return None
    
    def get_usda_nutrition(self, fdc_id: str) -> Optional[NutritionInfo]:
        """Get USDA nutrition data (thread-safe)"""
        try:
            with self.cache_lock:
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
            
            with self.cache_lock:
                self.usda_nutrition_cache[fdc_id] = nutrition
            print(f"   USDA nutrition: {nutrition.calories:.1f} kcal")
            return nutrition
            
        except Exception as e:
            print(f"   USDA nutrition error: {e}")
            return None
    
    def calculate_ingredient_nutrition(self, ingredient_text: str, verbose: bool = True) -> NutritionInfo:
        """Calculate nutrition with robust parsing"""
        try:
            if verbose:
                print(f"\nPARSING: '{ingredient_text}'")
            
            # Defensively strip any remaining parentheses and re-run deglue pass
            text_cleaned = re.sub(r'\([^)]*\)', '', ingredient_text)
            text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
            
            # Apply deglue rules again defensively
            canonical_units = ['kg', 'g', 'l', 'ml', 'tsp', 'tbsp', 'cup', 'oz', 'lb', 'pcs', 'clove', 'can', 'tin', 'pkt', 'bunch']
            for unit in canonical_units:
                # Insert spaces between unit and ingredient
                pattern = r'\b(' + re.escape(unit) + r')(?=[A-Za-z])'
                text_cleaned = re.sub(pattern, r'\1 ', text_cleaned, flags=re.IGNORECASE)
            
            text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
            
            # Parse ingredient using robust parser
            parsed = self.parser.parse_ingredient(text_cleaned)
            
            if verbose:
                print(f"   Quantity: {parsed.quantity}")
                print(f"   Unit: {parsed.unit}")
                print(f"   Ingredient: '{parsed.ingredient_name}'")
                print(f"   Grams: {parsed.grams:.1f}g")
            
            # Find nutrition data
            nutrition_per_100g = self.find_nutrition_data(parsed.ingredient_name, verbose)
            
            if not nutrition_per_100g:
                if verbose:
                    print(f"   No nutrition data found")
                return NutritionInfo()
            
            if verbose:
                print(f"   Nutrition per 100g: {nutrition_per_100g.calories:.1f} kcal")
            
            # Scale to actual grams
            scale_factor = parsed.grams / 100.0
            if verbose:
                print(f"   Scale factor: {scale_factor:.3f}")
            
            scaled_nutrition = NutritionInfo(
                calories=nutrition_per_100g.calories * scale_factor,
                protein=nutrition_per_100g.protein * scale_factor,
                carbs=nutrition_per_100g.carbs * scale_factor,
                fat=nutrition_per_100g.fat * scale_factor
            )
            
            if verbose:
                print(f"   Final: {scaled_nutrition.calories:.1f} kcal")
            
            # Add small delay for API rate limiting (reduced for threading)
            time.sleep(0.05)
            
            return scaled_nutrition
            
        except Exception as e:
            if verbose:
                print(f"   Error: {e}")
            return NutritionInfo()
    
    def calculate_recipe_nutrition(self, ingredients_str: str, verbose: bool = True) -> NutritionInfo:
        """Calculate recipe nutrition"""
        try:
            if not ingredients_str or pd.isna(ingredients_str):
                return NutritionInfo()
            
            ingredients = [ing.strip() for ing in ingredients_str.split('|') if ing.strip()]
            
            if verbose:
                print(f"\nRECIPE: Processing {len(ingredients)} ingredients")
                print("=" * 60)
            
            total_nutrition = NutritionInfo()
            
            for i, ingredient in enumerate(ingredients, 1):
                # Skip obvious instructions
                if len(ingredient) > 100 or any(word in ingredient.lower() for word in 
                                               ['enjoy', 'serve', 'garnish', 'hot with']):
                    if verbose:
                        print(f"SKIPPING {i}: '{ingredient}' (instruction)")
                    continue
                
                if verbose:
                    print(f"\n--- INGREDIENT {i}/{len(ingredients)} ---")
                ingredient_nutrition = self.calculate_ingredient_nutrition(ingredient, verbose)
                
                # Only add to totals if not skipped (calories > 0)
                if ingredient_nutrition.calories > 0:
                    total_nutrition.calories += ingredient_nutrition.calories
                    total_nutrition.protein += ingredient_nutrition.protein
                    total_nutrition.carbs += ingredient_nutrition.carbs
                    total_nutrition.fat += ingredient_nutrition.fat
                
                if verbose:
                    print(f"RUNNING TOTAL: {total_nutrition.calories:.1f} kcal")
            
            # Apply guardrails
            if self.normalizer.has_suspicious_values(total_nutrition.calories, ingredients_str):
                if verbose:
                    print(f"\nSKIPPING RECIPE: Suspicious values detected (calories: {total_nutrition.calories:.1f})")
                    print("=" * 60)
                return NutritionInfo()  # Return empty nutrition
            
            if verbose:
                print(f"\nFINAL TOTAL: {total_nutrition.calories:.1f} kcal")
                print("=" * 60)
            
            return total_nutrition
            
        except Exception as e:
            logger.error(f"Error calculating recipe nutrition: {e}")
            return NutritionInfo()
    
    def process_recipe_batch(self, recipe_batch: List[Tuple[int, str, str]]) -> List[Tuple[int, NutritionInfo]]:
        """
        Process a batch of recipes in a single thread
        Args: List of (index, recipe_title, ingredients_string) tuples
        Returns: List of (index, nutrition_info) tuples
        """
        results = []
        for idx, recipe_title, ingredients in recipe_batch:
            try:
                nutrition = self.calculate_recipe_nutrition(ingredients, verbose=False)  # Non-verbose for threads
                results.append((idx, nutrition))
                print(f"   Thread completed: {idx+1}. '{recipe_title[:30]}...' -> {nutrition.calories:.0f} kcal")
            except Exception as e:
                print(f"   Thread error {idx+1}: {e}")
                results.append((idx, NutritionInfo()))  # Return empty nutrition on error
        return results
    
    def process_csv_file(self, input_file: str, output_file: str = None,
                        ingredients_col: str = 'ingredients_per_person', max_workers: int = None):
        """Process CSV file with multi-threading"""
        try:
            logger.info(f"Loading recipes from {input_file}")
            df = pd.read_csv(input_file, encoding='utf-8')
            
            if ingredients_col not in df.columns:
                logger.error(f"Column '{ingredients_col}' not found!")
                logger.info(f"Available columns: {list(df.columns)}")
                return
            
            logger.info(f" Loaded {len(df)} recipes")
            
            # Add nutrition columns
            df['calories'] = 0.0
            df['protein_g'] = 0.0
            df['carbs_g'] = 0.0
            df['fat_g'] = 0.0
            
            # Determine number of threads
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), 4)  # Cap at 4 to avoid overwhelming APIs
            
            logger.info(f"Using {max_workers} threads for parallel processing")
            
            # Prepare recipe batches
            recipes = []
            for idx, row in df.iterrows():
                ingredients = row.get(ingredients_col, '')
                recipe_title = row.get('title', f'Recipe {idx+1}')
                recipes.append((idx, recipe_title, ingredients))
            
            # Split recipes into batches
            batch_size = max(1, len(recipes) // max_workers)
            batches = [recipes[i:i + batch_size] for i in range(0, len(recipes), batch_size)]
            
            logger.info(f"Processing {len(recipes)} recipes in {len(batches)} batches of ~{batch_size} recipes each")
            
            # Process batches in parallel
            completed_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {executor.submit(self.process_recipe_batch, batch): batch 
                                 for batch in batches}
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        
                        # Update DataFrame with results
                        for idx, nutrition in batch_results:
                            df.at[idx, 'calories'] = round(nutrition.calories, 1)
                            df.at[idx, 'protein_g'] = round(nutrition.protein, 1)
                            df.at[idx, 'carbs_g'] = round(nutrition.carbs, 1)
                            df.at[idx, 'fat_g'] = round(nutrition.fat, 1)
                        
                        completed_count += len(batch_results)
                        logger.info(f"Progress: {completed_count}/{len(df)} recipes processed")
                        
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
                        # Continue with other batches
            
            # Save results
            if output_file is None:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}_robust_nutrition.csv"
            
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f" Results saved to {output_file}")
            
            # Show summary
            self.show_summary(df)
            
            return df
            
        except Exception as e:
            logger.error(f" Error processing CSV: {e}")
            raise
    
    def show_summary(self, df: pd.DataFrame):
        """Display summary"""
        print("\n" + "="*70)
        print("ROBUST INGREDIENT PARSING + NUTRITION CALCULATION")
        print("="*70)
        
        avg_calories = df['calories'].mean()
        avg_protein = df['protein_g'].mean()
        avg_carbs = df['carbs_g'].mean()
        avg_fat = df['fat_g'].mean()
        
        print(f"\n AVERAGE NUTRITION PER RECIPE (single serving):")
        print(f"  Calories: {avg_calories:.1f} kcal")
        print(f"  Protein:  {avg_protein:.1f} g")
        print(f"  Carbs:    {avg_carbs:.1f} g")
        print(f"  Fat:      {avg_fat:.1f} g")
        
        print(f"\n SAMPLE RECIPES:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            title = row.get('title', f'Recipe {i+1}')
            calories = row.get('calories', 0)
            protein = row.get('protein_g', 0)
            carbs = row.get('carbs_g', 0)
            fat = row.get('fat_g', 0)
            
            print(f"{i+1}. {title[:45]}...")
            print(f"    {calories:.0f} kcal |  {protein:.1f}g protein |  {carbs:.1f}g carbs |  {fat:.1f}g fat")
        
        print("="*70)
    
    def test_parser(self):
        """Test the robust parser with normalizer and various ingredients"""
        test_ingredients = [
            "25 gr salted butter",
            "1/2 cup rice flour", 
            "a pinch of salt",
            "1/4 onion, diced",
            "2 tbsp coconut oil",
            "3 pieces chicken",
            "1 tsp curry powder",
            "some fresh ginger",
            "few curry leaves",
            "1 can coconut milk",
            "2 1/2 cups water",
            # NEW NORMALIZER TEST CASES
            "Beef oxtail - 1/6kg",
            "1/8cupwater", 
            "tbspcoconut oil",
            "turmeric powder 1/2 teaspoon",
            "salt – as you need",
            "½ cup rice flour",
            "2 3/4 cups water to taste"
        ]
        
        print("\nTESTING ROBUST PARSER WITH NORMALIZER")
        print("="*60)
        print("Pre-normalizes complex formats")
        print("Handles liquid densities") 
        print("Skips 'as needed' items without quantities")
        print("="*60)
        
        for ingredient in test_ingredients:
            parsed = self.parser.parse_ingredient(ingredient)
            if parsed.unit == 'skip':
                print(f"\nInput: '{ingredient}'")
                print(f"  -> SKIPPED (as needed/to taste)")
            else:
                print(f"\nInput: '{ingredient}'")
                print(f"  -> {parsed.quantity} {parsed.unit} of '{parsed.ingredient_name}' = {parsed.grams:.1f}g")
        
        print("\n" + "="*60)
        print("NORMALIZER LOG MESSAGES SHOW TRANSFORMATIONS ABOVE")
        print("="*60)

# Main execution with comprehensive testing
if __name__ == "__main__":
    print("ROBUST INGREDIENT PARSER + NUTRITION CALCULATOR")
    print("=" * 60)
    print("Handles complex parsing: 'a pinch of salt', '1/4 onion'")
    print("Standardizes everything to grams")
    print("Sri Lankan database + USDA API fallback")
    print("Most accurate nutrition calculation possible!")
    print()
    
    # Check database file
    db_file = 'ingredient-dataset_nutrition.xlsx'
    if not os.path.exists(db_file):
        print(f" Nutrition database file not found: {db_file}")
        exit(1)
    
    try:
        # Initialize calculator
        calculator = HybridNutritionCalculator(db_file)
        
        # Test parser first?
        test_parser = input("Test the ingredient parser first? (y/n): ").lower()
        if test_parser == 'y':
            calculator.test_parser()
            print()
        
        # Get input file
        input_file = input("Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(input_file):
            print(f" File not found: {input_file}")
            exit(1)
        
        # Get column name
        ingredients_col = input("Ingredients column name (default: 'ingredients_per_person'): ").strip() or 'ingredients_per_person'
        
        # Get number of threads
        max_workers_input = input(f"Number of threads (1-{multiprocessing.cpu_count()}, default: auto): ").strip()
        max_workers = None
        if max_workers_input.isdigit():
            max_workers = max(1, min(int(max_workers_input), multiprocessing.cpu_count()))
        
        # Output file
        output_file = input("Output file path (press Enter for auto-generated): ").strip()
        if not output_file:
            output_file = None
        
        threads_text = f"{max_workers} threads" if max_workers else "auto threads"
        print(f"\n Starting robust nutrition calculation with {threads_text}...")
        print(" Advanced parsing + Ingredient normalizer + Sri Lankan DB + USDA API")
        print(" Multi-threaded processing for maximum speed...")
        print(" This may take a few minutes...")
        
        # Process CSV with timing
        start_time = time.time()
        calculator.process_csv_file(
            input_file=input_file,
            output_file=output_file,
            ingredients_col=ingredients_col,
            max_workers=max_workers
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\n Robust nutrition calculation completed in {processing_time:.1f} seconds!")
        print(" Most accurate parsing and nutrition data possible!")
        print(f" Multi-threaded processing with {threads_text}")
        
    except KeyboardInterrupt:
        print("\n Calculation interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\n Make sure:")
        print("   - Internet connection for USDA API")
        print("   - Nutrition database Excel file is present")
        print("   - CSV has correct ingredients column")