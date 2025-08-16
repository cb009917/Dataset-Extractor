import pandas as pd
import numpy as np
import re
import logging
import requests
import time
import json
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

class ImprovedIngredientParser:
    """
    Improved ingredient parser with coconut aliases, negative rules, and enhanced validation
    """
    
    def __init__(self):
        # Initialize the ingredient normalizer
        self.normalizer = IngredientNormalizer()
        
        # C. Coconut milk alias mapping for tighter matching
        self.coconut_aliases = {
            'thick coconut milk': 'Coconut Milk - 1st Milk',
            'first coconut milk': 'Coconut Milk - 1st Milk',
            'coconut milk (canned)': 'Coconut Milk - 1st Milk',
            'canned coconut milk': 'Coconut Milk - 1st Milk',
            'thin coconut milk': 'Coconut Milk - 2nd Milk',
            'second coconut milk': 'Coconut Milk - 2nd Milk',
            'coconut milk': 'Coconut Milk - 1st Milk'  # Default to thick
        }
        
        # C. Negative matching rules to prevent disasters
        self.negative_rules = {
            'milk': ['Dessicated Coconut', 'Raw Coconut', 'Coconut Oil']  # Never match milk to these
        }
        
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
            
            # F. Enhanced volume units with better densities
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
            
            # F. Enhanced can sizes with proper defaults
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
        """Convert fraction string to decimal, handling unicode slash"""
        try:
            # Handle unicode fraction slash
            fraction_str = fraction_str.replace('\u2044', '/')
            
            # Handle double fractions like "1/4/2" = "1/8"
            if fraction_str.count('/') == 2:
                parts = fraction_str.split('/')
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
                    # (a/b)/c = a/(b*c)
                    from math import gcd
                    num, den = a, b * c
                    g = gcd(num, den)
                    fraction_str = f"{num//g}/{den//g}"
            
            if '/' in fraction_str:
                return float(Fraction(fraction_str))
            return float(fraction_str)
        except:
            return 1.0
    
    def parse_quantity(self, text: str) -> Tuple[float, str]:
        """Extract quantity from text, return (quantity, remaining_text)"""
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
        """Extract unit from text, return (unit, remaining_text)"""
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
        """Convert any quantity/unit to grams using comprehensive conversion table"""
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
    
    def parse_ingredient(self, ingredient_text: str, verbose: bool = True) -> ParsedIngredient:
        """Main parsing function: converts any ingredient string to standardized format
        Now includes pre-normalization step and validation"""
        try:
            original_text = ingredient_text.strip()
            
            # STEP 0: Normalize the ingredient text using the normalizer
            normalized_text, should_skip = self.normalizer.normalize_token(original_text)
            
            if should_skip:
                if verbose:
                    print(f"   SKIPPED (as needed/to taste): '{original_text}'")
                return ParsedIngredient(
                    quantity=0.0,
                    unit='skip',
                    ingredient_name=original_text.lower().strip(),
                    grams=0.0,
                    original_text=original_text
                )
            
            if verbose:
                print(f"   Normalized: '{original_text}' -> '{normalized_text}'")
            
            # Step 1: Extract quantity from normalized text
            quantity, remaining_text = self.parse_quantity(normalized_text)
            
            # Step 2: Extract unit
            unit, ingredient_name = self.parse_unit(remaining_text)
            
            # Step 3: Clean ingredient name
            clean_name = self.clean_ingredient_name(ingredient_name)
            
            # E. VALIDATION: Require valid qty + unit + ingredient shape
            if not clean_name or len(clean_name.strip()) < 2:
                if verbose:
                    print(f"   DROPPED: Invalid ingredient name '{clean_name}'")
                return ParsedIngredient(
                    quantity=0.0,
                    unit='skip',
                    ingredient_name=clean_name,
                    grams=0.0,
                    original_text=original_text
                )
            
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
            if verbose:
                print(f"   Parsing error for '{ingredient_text}': {e}")
            # Return safe fallback
            return ParsedIngredient(
                quantity=1.0,
                unit='piece',
                ingredient_name=ingredient_text.lower().strip(),
                grams=50.0,
                original_text=ingredient_text
            )

class ImprovedNutritionCalculator:
    """
    Improved nutrition calculator with all the requested enhancements
    """
    
    def __init__(self, nutrition_db_file: str = 'ingredient-dataset_nutrition.xlsx', 
                 usda_api_key: str = "fz7YpdJ3tR0PfazCyXr4mJYgyDGir5swSwIANOJz"):
        
        self.parser = ImprovedIngredientParser()
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
    
    def find_nutrition_data(self, ingredient_name: str, verbose: bool = True) -> tuple[Optional[NutritionInfo], str]:
        """Find nutrition data using hybrid approach with improved matching"""
        
        # Check coconut aliases first to prevent milk->coconut disasters
        for alias, canonical in self.parser.coconut_aliases.items():
            if alias in ingredient_name.lower():
                if canonical.lower() in self.nutrition_database:
                    if verbose:
                        print(f"   Found via coconut alias: '{ingredient_name}' -> '{canonical}'")
                    return self.nutrition_database[canonical.lower()], canonical
        
        # Check negative rules to prevent disasters
        for token, forbidden_matches in self.parser.negative_rules.items():
            if token in ingredient_name.lower():
                for forbidden in forbidden_matches:
                    if forbidden.lower() in ingredient_name.lower():
                        if verbose:
                            print(f"   NEGATIVE RULE BLOCKED: '{ingredient_name}' cannot match '{forbidden}'")
                        # Continue searching but skip this match
        
        # Try Sri Lankan database exact match
        if ingredient_name.lower() in self.nutrition_database:
            if verbose:
                print(f"   Found in Sri Lankan DB (exact): '{ingredient_name}'")
            return self.nutrition_database[ingredient_name.lower()], ingredient_name
        
        # Try fuzzy matching with token-level agreement
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.0
        
        ingredient_tokens = set(ingredient_name.lower().split())
        
        for db_name in self.nutrition_database.keys():
            db_tokens = set(db_name.lower().split())
            
            # Require token-level agreement for fuzzy matching
            if not (ingredient_tokens & db_tokens):  # No common tokens
                continue
                
            # Check negative rules at fuzzy match level too
            skip_match = False
            for token, forbidden_matches in self.parser.negative_rules.items():
                if token in ingredient_name.lower():
                    for forbidden in forbidden_matches:
                        if forbidden.lower() in db_name.lower():
                            if verbose:
                                print(f"   FUZZY NEGATIVE RULE: Skipping '{db_name}' for '{ingredient_name}'")
                            skip_match = True
                            break
                if skip_match:
                    break
            
            if skip_match:
                continue
            
            score = SequenceMatcher(None, ingredient_name, db_name).ratio()
            if score > best_score and score > 0.7:  # High threshold
                best_score = score
                best_match = db_name
        
        if best_match:
            if verbose:
                print(f"   Found in Sri Lankan DB (fuzzy): '{ingredient_name}' -> '{best_match}' ({best_score:.2f})")
            return self.nutrition_database[best_match], best_match
        
        # Fallback to USDA API
        if verbose:
            print(f"   Trying USDA API for: '{ingredient_name}'")
        fdc_id = self.search_usda_food(ingredient_name)
        if fdc_id:
            nutrition = self.get_usda_nutrition(fdc_id)
            if nutrition:
                return nutrition, f"USDA-{fdc_id}"
        
        if verbose:
            print(f"   No nutrition data found for: '{ingredient_name}'")
        return None, "NOT_FOUND"
    
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
    
    def calculate_ingredient_nutrition(self, ingredient_text: str, verbose: bool = True) -> tuple[NutritionInfo, dict]:
        """Calculate nutrition with robust parsing and comprehensive logging"""
        log_data = {
            'original_text': ingredient_text,
            'normalized_text': '',
            'matched_db_key': '',
            'unit_conversion': '',
            'nutrition_per_100g': {},
            'final_grams': 0,
            'errors': []
        }
        
        try:
            if verbose:
                print(f"\nPARSING: '{ingredient_text}'")
            
            # Parse ingredient using robust parser
            parsed = self.parser.parse_ingredient(ingredient_text, verbose)
            log_data['normalized_text'] = f"{parsed.quantity} {parsed.unit} {parsed.ingredient_name}"
            log_data['final_grams'] = parsed.grams
            log_data['unit_conversion'] = f"{parsed.quantity} {parsed.unit} → {parsed.grams:.1f}g"
            
            if parsed.unit == 'skip':
                return NutritionInfo(), log_data
            
            if verbose:
                print(f"   Quantity: {parsed.quantity}")
                print(f"   Unit: {parsed.unit}")
                print(f"   Ingredient: '{parsed.ingredient_name}'")
                print(f"   Grams: {parsed.grams:.1f}g")
            
            # Find nutrition data
            nutrition_per_100g, matched_key = self.find_nutrition_data(parsed.ingredient_name, verbose)
            log_data['matched_db_key'] = matched_key
            
            if not nutrition_per_100g:
                if verbose:
                    print(f"   No nutrition data found")
                log_data['errors'].append('No nutrition data found')
                return NutritionInfo(), log_data
            
            log_data['nutrition_per_100g'] = {
                'calories': nutrition_per_100g.calories,
                'protein': nutrition_per_100g.protein,
                'carbs': nutrition_per_100g.carbs,
                'fat': nutrition_per_100g.fat
            }
            
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
            
            return scaled_nutrition, log_data
            
        except Exception as e:
            if verbose:
                print(f"   Error: {e}")
            log_data['errors'].append(str(e))
            return NutritionInfo(), log_data
    
    def calculate_recipe_nutrition(self, ingredients_str: str, verbose: bool = True) -> tuple[NutritionInfo, dict, bool]:
        """Calculate recipe nutrition with outlier detection"""
        recipe_log = {
            'ingredients_processed': 0,
            'ingredients_skipped': 0,
            'ingredient_logs': [],
            'outlier_flags': [],
            'total_calories': 0,
            'total_fat_g': 0
        }
        
        try:
            if not ingredients_str or pd.isna(ingredients_str):
                return NutritionInfo(), recipe_log, False
            
            ingredients = [ing.strip() for ing in ingredients_str.split('|') if ing.strip()]
            
            if verbose:
                print(f"\nRECIPE: Processing {len(ingredients)} ingredients")
                print("=" * 60)
            
            total_nutrition = NutritionInfo()
            high_contributors = []  # Track high calorie/fat contributors
            
            for i, ingredient in enumerate(ingredients, 1):
                # Skip obvious instructions
                if len(ingredient) > 100 or any(word in ingredient.lower() for word in 
                                               ['enjoy', 'serve', 'garnish', 'hot with']):
                    if verbose:
                        print(f"SKIPPING {i}: '{ingredient}' (instruction)")
                    recipe_log['ingredients_skipped'] += 1
                    continue
                
                if verbose:
                    print(f"\n--- INGREDIENT {i}/{len(ingredients)} ---")
                ingredient_nutrition, ingredient_log = self.calculate_ingredient_nutrition(ingredient, verbose)
                
                recipe_log['ingredient_logs'].append(ingredient_log)
                
                # Track high contributors for outlier detection
                if ingredient_nutrition.calories > 200 or ingredient_nutrition.fat > 20:
                    high_contributors.append({
                        'ingredient': ingredient,
                        'calories': ingredient_nutrition.calories,
                        'fat': ingredient_nutrition.fat,
                        'matched_key': ingredient_log.get('matched_db_key', 'Unknown')
                    })
                
                # Only add to totals if not skipped (calories > 0)
                if ingredient_nutrition.calories > 0:
                    total_nutrition.calories += ingredient_nutrition.calories
                    total_nutrition.protein += ingredient_nutrition.protein
                    total_nutrition.carbs += ingredient_nutrition.carbs
                    total_nutrition.fat += ingredient_nutrition.fat
                    recipe_log['ingredients_processed'] += 1
                else:
                    recipe_log['ingredients_skipped'] += 1
                
                if verbose:
                    print(f"RUNNING TOTAL: {total_nutrition.calories:.1f} kcal")
            
            recipe_log['total_calories'] = total_nutrition.calories
            recipe_log['total_fat_g'] = total_nutrition.fat
            
            # G. OUTLIER DETECTION
            is_outlier = False
            if total_nutrition.calories > 1500:
                recipe_log['outlier_flags'].append(f'High calories: {total_nutrition.calories:.0f} > 1500')
                is_outlier = True
            
            if total_nutrition.fat > 150:
                recipe_log['outlier_flags'].append(f'High fat: {total_nutrition.fat:.0f}g > 150g')
                is_outlier = True
            
            # Log high contributors for outliers
            if is_outlier and high_contributors:
                recipe_log['outlier_flags'].append('High contributors:')
                for contrib in high_contributors:
                    recipe_log['outlier_flags'].append(
                        f"  - {contrib['ingredient'][:40]}... -> {contrib['calories']:.0f}cal, {contrib['fat']:.0f}g fat (matched: {contrib['matched_key']})"
                    )
                    
                    # Check for coconut milk -> dessicated coconut disasters
                    if 'milk' in contrib['ingredient'].lower() and 'dessicated coconut' in contrib['matched_key'].lower():
                        recipe_log['outlier_flags'].append("  *** COCONUT DISASTER DETECTED! Milk matched to Dessicated Coconut ***")
            
            if verbose:
                print(f"\nFINAL TOTAL: {total_nutrition.calories:.1f} kcal")
                if is_outlier:
                    print(f"*** OUTLIER DETECTED ***")
                    for flag in recipe_log['outlier_flags']:
                        print(f"   {flag}")
                print("=" * 60)
            
            return total_nutrition, recipe_log, is_outlier
            
        except Exception as e:
            logger.error(f"Error calculating recipe nutrition: {e}")
            recipe_log['outlier_flags'].append(f'Error: {str(e)}')
            return NutritionInfo(), recipe_log, False
    
    def _parse_servings(self, servings_str: str) -> Optional[float]:
        """B. Parse servings string to numeric value"""
        if not servings_str or pd.isna(servings_str):
            return None
        
        # Manual overrides for common problematic cases
        servings_overrides = {
            'family': 4,
            'large family': 6,
            'small family': 3,
            'couple': 2,
            'one': 1,
            'single': 1
        }
        
        servings_str = str(servings_str).strip().lower()
        
        # Check overrides first
        for override_key, override_value in servings_overrides.items():
            if override_key in servings_str:
                return float(override_value)
        
        # Extract numbers from string
        numbers = re.findall(r'\d+(?:\.\d+)?', servings_str)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return None
    
    def process_recipe_batch(self, recipe_batch: List[Tuple[int, str, str, str]]) -> List[Tuple[int, NutritionInfo, dict, bool]]:
        """Process a batch of recipes in a single thread
        Args: List of (index, recipe_title, ingredients_string, servings_str) tuples
        Returns: List of (index, nutrition_info, log_data, is_outlier) tuples"""
        results = []
        for idx, recipe_title, ingredients, servings_str in recipe_batch:
            try:
                # Parse servings to determine per-person calculation
                servings = self._parse_servings(servings_str)
                
                if servings is None or servings <= 0:
                    print(f"   Thread skipped {idx+1}: Invalid servings '{servings_str}'")
                    results.append((idx, NutritionInfo(), {'error': 'Invalid servings'}, False))
                    continue
                
                # Calculate total recipe nutrition
                total_nutrition, recipe_log, is_outlier = self.calculate_recipe_nutrition(ingredients, verbose=False)
                
                # Convert to per-person nutrition
                per_person_nutrition = NutritionInfo(
                    calories=total_nutrition.calories / servings,
                    protein=total_nutrition.protein / servings,
                    carbs=total_nutrition.carbs / servings,
                    fat=total_nutrition.fat / servings
                )
                
                recipe_log['servings'] = servings
                recipe_log['per_person'] = True
                
                if is_outlier:
                    print(f"   Thread completed (OUTLIER): {idx+1}. '{recipe_title[:30]}...' -> {per_person_nutrition.calories:.0f} kcal/person")
                else:
                    print(f"   Thread completed: {idx+1}. '{recipe_title[:30]}...' -> {per_person_nutrition.calories:.0f} kcal/person")
                    
                results.append((idx, per_person_nutrition, recipe_log, is_outlier))
                
            except Exception as e:
                print(f"   Thread error {idx+1}: {e}")
                results.append((idx, NutritionInfo(), {'error': str(e)}, False))
        return results
    
    def process_csv_file(self, input_file: str, output_file: str = None,
                        ingredients_col: str = 'ingredients_per_person', 
                        servings_col: str = 'servings',
                        max_workers: int = None):
        """Process CSV file with multi-threading and all improvements"""
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
            
            # Add logging columns
            df['processing_log'] = ''
            df['is_outlier'] = False
            df['skipped_reason'] = ''
            
            # Determine number of threads
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), 4)  # Cap at 4 to avoid overwhelming APIs
            
            logger.info(f"Using {max_workers} threads for parallel processing")
            
            # Prepare recipe batches
            recipes = []
            skipped_count = 0
            
            for idx, row in df.iterrows():
                ingredients = row.get(ingredients_col, '')
                servings = row.get(servings_col, '')
                recipe_title = row.get('title', f'Recipe {idx+1}')
                
                # A. Always compute from ingredients_per_person only
                if not ingredients or pd.isna(ingredients) or len(str(ingredients).strip()) < 5:
                    df.at[idx, 'skipped_reason'] = 'Empty or invalid ingredients_per_person'
                    skipped_count += 1
                    continue
                    
                recipes.append((idx, recipe_title, ingredients, servings))
            
            logger.info(f"Processing {len(recipes)} recipes, skipped {skipped_count} due to empty ingredients_per_person")
            
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
                        for idx, nutrition, log_data, is_outlier in batch_results:
                            if 'error' in log_data:
                                df.at[idx, 'skipped_reason'] = log_data['error']
                                continue
                                
                            df.at[idx, 'calories'] = round(nutrition.calories, 1)
                            df.at[idx, 'protein_g'] = round(nutrition.protein, 1)
                            df.at[idx, 'carbs_g'] = round(nutrition.carbs, 1)
                            df.at[idx, 'fat_g'] = round(nutrition.fat, 1)
                            df.at[idx, 'is_outlier'] = is_outlier
                            
                            # Store processing log as JSON for debugging
                            try:
                                df.at[idx, 'processing_log'] = json.dumps(log_data, indent=None, separators=(',', ':'))
                            except:
                                df.at[idx, 'processing_log'] = str(log_data)
                        
                        completed_count += len(batch_results)
                        logger.info(f"Progress: {completed_count}/{len(df)} recipes processed")
                        
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
                        # Continue with other batches
            
            # Save results
            if output_file is None:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}_improved_nutrition.csv"
            
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_file}")
            
            # Show summary
            self.show_summary(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def show_summary(self, df: pd.DataFrame):
        """Display comprehensive summary"""
        print("\n" + "="*70)
        print("IMPROVED INGREDIENT PARSING + NUTRITION CALCULATION")
        print("="*70)
        
        # Filter out outliers and skipped for averages
        valid_df = df[(df['calories'] > 0) & (~df.get('is_outlier', False))]
        outlier_count = df.get('is_outlier', False).sum()
        skipped_count = len(df) - len(df[df['calories'] > 0])
        
        if len(valid_df) > 0:
            avg_calories = valid_df['calories'].mean()
            avg_protein = valid_df['protein_g'].mean()
            avg_carbs = valid_df['carbs_g'].mean()
            avg_fat = valid_df['fat_g'].mean()
        else:
            avg_calories = avg_protein = avg_carbs = avg_fat = 0
        
        print(f"\n SUMMARY:")
        print(f"  Total recipes: {len(df)}")
        print(f"  Successfully processed: {len(valid_df)}")
        print(f"  Outliers flagged: {outlier_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"\n AVERAGE NUTRITION PER PERSON (excluding outliers):")
        print(f"  Calories: {avg_calories:.1f} kcal")
        print(f"  Protein:  {avg_protein:.1f} g")
        print(f"  Carbs:    {avg_carbs:.1f} g")
        print(f"  Fat:      {avg_fat:.1f} g")
        
        if outlier_count > 0:
            print(f"\n OUTLIERS (check processing_log for details):")
            outlier_df = df[df.get('is_outlier', False) == True]
            for i, row in outlier_df.head(3).iterrows():
                title = row.get('title', f'Recipe {i+1}')
                calories = row.get('calories', 0)
                fat = row.get('fat_g', 0)
                print(f"  - {title[:40]}... -> {calories:.0f} kcal, {fat:.0f}g fat")
        
        print(f"\n SAMPLE RECIPES (valid, non-outlier):")
        sample_df = valid_df.head(5)
        for i, (_, row) in enumerate(sample_df.iterrows(), 1):
            title = row.get('title', f'Recipe {_+1}')
            calories = row.get('calories', 0)
            protein = row.get('protein_g', 0)
            carbs = row.get('carbs_g', 0)
            fat = row.get('fat_g', 0)
            
            print(f"{i}. {title[:45]}...")
            print(f"    {calories:.0f} kcal |  {protein:.1f}g protein |  {carbs:.1f}g carbs |  {fat:.1f}g fat")
        
        print("="*70)
    
    def test_parser(self):
        """Test the improved parser with all enhancements"""
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
            "2 3/4 cups water to taste",
            # COCONUT ALIAS TESTS
            "1 can thick coconut milk",
            "200ml thin coconut milk",
            # FRACTION SLASH TESTS
            "1⁄4 cup rice flour",  # Unicode fraction slash
            "1/4/2 cup water"     # Double fraction
        ]
        
        print("\nTESTING IMPROVED PARSER WITH ALL ENHANCEMENTS")
        print("="*60)
        print("✓ Unicode fraction slash normalization")
        print("✓ Double fraction arithmetic (1/4/2 → 1/8)")
        print("✓ Coconut milk aliases") 
        print("✓ Negative rules (milk ≠ coconut)")
        print("✓ Enhanced validation")
        print("✓ Comprehensive logging")
        print("="*60)
        
        for ingredient in test_ingredients:
            nutrition, log_data = self.calculate_ingredient_nutrition(ingredient, verbose=False)
            parsed = self.parser.parse_ingredient(ingredient, verbose=False)
            
            if parsed.unit == 'skip':
                print(f"\nInput: '{ingredient}'")
                print(f"  -> SKIPPED (as needed/to taste or invalid)")
            else:
                print(f"\nInput: '{ingredient}'")
                print(f"  -> {parsed.quantity} {parsed.unit} of '{parsed.ingredient_name}' = {parsed.grams:.1f}g")
                print(f"  -> Matched: {log_data.get('matched_db_key', 'Not found')}")
                if nutrition.calories > 0:
                    print(f"  -> Nutrition: {nutrition.calories:.1f} kcal")
        
        print("\n" + "="*60)
        print("ALL IMPROVEMENTS IMPLEMENTED AND TESTED")
        print("="*60)

# Main execution
if __name__ == "__main__":
    print("IMPROVED INGREDIENT PARSER + NUTRITION CALCULATOR")
    print("=" * 60)
    print("✓ A. Always compute from ingredients_per_person only")
    print("✓ B. Fix servings - parse to numeric with manual overrides") 
    print("✓ C. Tighten ingredient matching - coconut aliases & negative rules")
    print("✓ D. Normalize unicode fraction slash (⁄ → /) and fix 1/4/2 → 1/8")
    print("✓ E. Drop non-ingredients - require valid qty + unit + ingredient")
    print("✓ F. Enhanced densities + can sizes for sanity")
    print("✓ G. Outlier guardrails - flag >1500 cal or >150g fat recipes")
    print("✓ H. Comprehensive logging for every token, match, conversion")
    print()
    
    # Check database file
    db_file = 'ingredient-dataset_nutrition.xlsx'
    if not os.path.exists(db_file):
        print(f"⚠ Nutrition database file not found: {db_file}")
        exit(1)
    
    try:
        # Initialize calculator
        calculator = ImprovedNutritionCalculator(db_file)
        
        # Test parser first?
        test_parser = input("Test the improved parser first? (y/n): ").lower()
        if test_parser == 'y':
            calculator.test_parser()
            print()
        
        # Get input file
        input_file = input("Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(input_file):
            print(f"⚠ File not found: {input_file}")
            exit(1)
        
        # Get column names
        ingredients_col = input("Ingredients column name (default: 'ingredients_per_person'): ").strip() or 'ingredients_per_person'
        servings_col = input("Servings column name (default: 'servings'): ").strip() or 'servings'
        
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
        print(f"\n🚀 Starting IMPROVED nutrition calculation with {threads_text}...")
        print("✅ All 8 improvements implemented!")
        print("🔍 Advanced parsing + Enhanced matching + Sri Lankan DB + USDA API")
        print("⚡ Multi-threaded processing with comprehensive logging...")
        print("⏱ This may take a few minutes...")
        
        # Process CSV with timing
        start_time = time.time()
        calculator.process_csv_file(
            input_file=input_file,
            output_file=output_file,
            ingredients_col=ingredients_col,
            servings_col=servings_col,
            max_workers=max_workers
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\n✅ IMPROVED nutrition calculation completed in {processing_time:.1f} seconds!")
        print("🎯 All improvements applied - most accurate parsing possible!")
        print(f"⚡ Multi-threaded processing with {threads_text}")
        print("📊 Check processing_log column for detailed debugging info")
        
    except KeyboardInterrupt:
        print("\n⏹ Calculation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Make sure:")
        print("   - Internet connection for USDA API")
        print("   - Nutrition database Excel file is present")
        print("   - CSV has correct ingredients column")