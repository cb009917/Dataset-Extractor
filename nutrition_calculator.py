#!/usr/bin/env python3
"""
Nutrition Calculator with Defensive Cleaning and Local Alias Support
Calculates nutrition data using Sri Lankan database with local aliases.
Excludes A & B items and applies safety guards.
"""

import pandas as pd
import numpy as np
import re
import logging
import os
import csv
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Set UTF-8 encoding at process start
os.environ["PYTHONIOENCODING"] = "utf-8"

from common_text import load_alias, save_alias, canonicalize_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token cleanup patterns
UNIT_CANON = r"(?:kg|g|l|ml|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)"
_PAT_NUM_TO_UNIT = rf'(?<![A-Za-z])(\d+(?:[.,]\d+)?(?:\s+\d/\d)?|\d/\d)\s*({UNIT_CANON})\b'
_PAT_UNIT_TO_WORD = rf'\b({UNIT_CANON})(?=[A-Za-z])'

def normalize_ing_token(s: str) -> str:
    if not isinstance(s, str): return ""
    # strip (...) and extra spaces
    s = re.sub(r"\([^)]*\)", "", s)
    
    # More aggressive de-gluing for cases like "1tspSalt", "1CupWater", "450gFresh"
    # Handle digit+unit+word combinations
    s = re.sub(r'(\d+(?:[.,]\d+)?)(tsp|tbsp|cup|g|kg|ml|l|oz|lb)([a-zA-Z])', r'\1 \2 \3', s, flags=re.IGNORECASE)
    
    # de-glue: unit→word and number→unit (original patterns)
    s = re.sub(_PAT_UNIT_TO_WORD, r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(_PAT_NUM_TO_UNIT,  r"\1 \2", s, flags=re.IGNORECASE)
    
    # Handle fractions like "/4tsp", "/2cup"
    s = re.sub(r'(/\d+)(tsp|tbsp|cup|g|kg|ml|l)', r'\1 \2', s, flags=re.IGNORECASE)
    
    # common split-letter/typos
    fixes = [
        (r"\bl\s*eeks\b","leeks"), (r"\bl\s*ime\b","lime"), (r"\bg\s*arlic\b","garlic"),
        (r"\bg\s*inger\b","ginger"), (r"\bg\s*rated\b","grated"),
        (r"\bg\s*reen\s+bean\b","green bean"), (r"\bg\s*reen\s+chil", "green chil"),
        (r"\bg\s*reen\s+gram", "green gram"), (r"\bg\s*round\s+", "ground "),
        (r"\bl\s*arge\b", "large"), (r"\bl\s*eaf\b", "leaf"), (r"\bl\s*eaves\b", "leaves"),
        (r"\bl\s*entil", "lentil"), (r"\bl\s*emon\b", "lemon"), (r"\bl\s*ong\s+", "long "),
        (r"\btspsalt\b","tsp salt"), (r"\btbspwater\b","tbsp water"), (r"\bcupwater\b","cup water"),
        (r"\bcupsalt\b","cup salt"), (r"\bcupsugar\b","cup sugar"), (r"\bcupoil\b","cup oil")
    ]
    for pat, rep in fixes: s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return re.sub(r"\s+"," ", s).strip()

# Category-based fallback templates
CATEGORY_KEYWORDS = {
  "vegetable": ["brinjal","eggplant","okra","beans","leek","cabbage","pumpkin","beetroot","carrot","tomato","capsicum","bell pepper","gourd","drumstick","banana flower","ash plantain","green beans","long beans","gotukola","mukunuwenna","kangkung","pennywort","amaranth","onion","onions","potato","potatoes","shallot","ginger","garlic","tapioca","cassava","yam","vinegar","keerai","leaves","flower","agathi","ponnanganni"],
  "fruit": ["banana","lime","lemon","mango","pineapple","papaya"],
  "staple": ["rice","rice flour","wheat flour","noodles","pasta","vermicelli","semolina","bread","flour","breadcrumbs","cornstarch","sago","rava","parotta","roti","all-purpose flour","all purpose flour","steamed flour","whole wheat flour","palmyra tuber flour"],
  "protein": ["egg","chicken","beef","pork","fish","prawns","sprats","dried fish","tofu","soya meat","chickpeas","red lentils","mung beans","cowpea","urad dal","crab","shrimp","mutton","eggs","tuna","lentils","dal","dhal","squid","cuttlefish","mackerel","salmon"],
  "dairy": ["milk","curd","yogurt","cheese","milk powder"],
  "fat": ["oil","coconut oil","ghee","butter","margarine"],
  "seasoning": ["sugar","jaggery","vinegar","vanilla","cocoa","nutella","cashew","nuts","powder","paste","extract","flakes","juice"]
}

CATEGORY_TEMPLATES = {
  "vegetable": {"kcal": 35,  "protein": 1.5, "carbs": 7,  "fat": 0.2},
  "fruit":     {"kcal": 60,  "protein": 0.7, "carbs": 15, "fat": 0.2},
  "staple":    {"kcal": 360, "protein": 9,   "carbs": 75, "fat": 1.5},
  "protein":   {"kcal": 200, "protein": 20,  "carbs": 0,  "fat": 13},
  "dairy":     {"kcal": 60,  "protein": 3.2, "carbs": 5,  "fat": 3.3},
  "fat":       {"kcal": 900, "protein": 0,   "carbs": 0,  "fat": 100},
  "seasoning": {"kcal": 300, "protein": 3,   "carbs": 75, "fat": 2}
}

def _category_from_name(name: str):
    nl = (name or "").lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in nl for k in kws): return cat
    return None

@dataclass
class NutritionInfo:
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0

class NutritionCalculator:
    """
    Nutrition calculator with defensive pre-processing and local alias support
    """
    
    def __init__(self, nutrition_db_file: str = 'config/ingredient-dataset_nutrition.xlsx'):
        self.nutrition_db_file = nutrition_db_file
        self.nutrition_database = {}
        self.shared_aliases = load_alias()
        if self.shared_aliases:
            logger.info(f"Loaded {len(self.shared_aliases)} user aliases from config/item_alias_user.json")
        self.unmatched_ingredients = {}
        self.unmatched_reports = []  # For comprehensive unmatched reporting
        
        # Canonical units for degluing
        self.canonical_units = ['kg', 'g', 'l', 'ml', 'tsp', 'tbsp', 'cup', 'oz', 'lb', 'pcs', 'clove', 'can', 'tin', 'pkt', 'bunch']
        
        # A-list: fillers & meta (exclude from calculations and denominators)
        self.a_list = [
            'salt', 'pepper', 'water', 'hot water', 'cold water', 'warm water', 'lukewarm water',
            'boiled water', 'normal water', 'boiling water', 'to taste', 'as required', 
            'optional', 'for garnish', 'for serving', 'for tempering', 'for marinade',
            'as you need', 'as you want', 'adjust to your taste', 'as needed'
        ]
        
        # B-list: micro-spices/herbs (exclude from calculations and denominators)
        self.b_list = [
            'turmeric', 'cumin seed', 'cumin powder', 'cumin', 'mustard seed', 'mustard seeds',
            'fenugreek', 'coriander powder', 'coriander seed', 'coriander seeds', 'chilli powder',
            'chili powder', 'chilli flakes', 'chili flakes', 'curry leaves', 'curry leaf',
            'pandan', 'rampe', 'cardamom', 'clove', 'cloves', 'cinnamon', 'cinnamon stick',
            'nutmeg', 'mace', 'black pepper', 'white pepper', 'red chili powder', 'green chili powder',
            'paprika', 'cayenne', 'bay leaves', 'bay leaf', 'thyme', 'oregano', 'basil',
            'rosemary', 'sage', 'mint', 'cilantro', 'parsley', 'dill', 'chilli', 'chili'
        ]
        
        # Local ingredient aliases (kept small, no web calls)
        self.local_aliases = {
            'red chili': 'dried chillies',
            'red chili powder': 'dried chillies',
            'green chili': 'green chillies',
            'sprat': 'dried sprats',
            'coconut milk': 'coconut',
            'thick coconut milk': 'coconut',
            'thin coconut milk': 'coconut',
            'fresh milk': 'milk',
            'garlic paste': 'garlic',
            'ginger paste': 'ginger',
            'tomato paste': 'tomatoes',
            'cooking oil': 'vegetable oil',
            'vegetable oil': 'sunflower oil',
            'spring onion': 'onions',
            'green onion': 'onions',
            'scallion': 'onions'
        }
        
        # Densities (ml → g)
        self.densities = {
            'milk': 1.03, 'curd': 1.03, 'yogurt': 1.03, 'coconut milk': 1.03,
            'soy sauce': 1.20, 'vinegar': 1.01, 'treacle': 1.35, 'honey': 1.35,
            'oil': 0.92, 'coconut oil': 0.92, 'vegetable oil': 0.92,
            'water': 1.00
        }
        
        # Piece weights (pcs → g)
        self.piece_weights = {
            'egg': 50, 'onion': 150, 'tomato': 120, 'lime': 30, 'lemon': 70,
            'green chili': 6, 'green chilli': 6, 'garlic clove': 3, 'garlic': 3
        }
        
        # Conversion factors to grams
        self.to_grams = {
            'g': 1.0, 'kg': 1000.0,
            'tsp': 2.0, 'tbsp': 6.0,  # Updated for better spice estimates
            'oz': 28.35, 'lb': 453.6,
            'can': 400.0, 'tin': 400.0, 'pkt': 50.0, 'bunch': 30.0, 'clove': 3.0
        }
        
        self.load_nutrition_database()
    
    def load_nutrition_database(self):
        """Load Sri Lankan nutrition database"""
        try:
            logger.info(f"Loading nutrition database from {self.nutrition_db_file}")
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
                
                # Estimate protein from remaining calories
                carb_calories = carbs * 4
                fat_calories = fat * 9
                remaining_calories = max(0, calories - carb_calories - fat_calories)
                protein = max(0, round(remaining_calories / 4, 1))
                
                # Only accept DB rows where macros are not all zero/NaN
                if calories <= 0 and protein <= 0 and carbs <= 0 and fat <= 0:
                    continue
                
                nutrition = NutritionInfo(calories=calories, protein=protein, carbs=carbs, fat=fat)
                
                # Store with canonicalized name for consistent matching
                canonical_name = canonicalize_name(name)
                if canonical_name:
                    self.nutrition_database[canonical_name] = nutrition
            
            logger.info(f"Processed {len(self.nutrition_database)} nutrition entries")
            
        except Exception as e:
            logger.error(f"Error loading nutrition database: {e}")
            raise
    
    def map_to_db_key(self, name: str) -> str:
        """Map ingredient name to DB key using shared aliases"""
        if not name:
            return ""
        
        # Apply shared canonicalization 
        canonical = canonicalize_name(name)
        
        # Try alias -> exact (canonicalized)
        if canonical in self.shared_aliases:
            canonical = self.shared_aliases[canonical]
        
        return canonical
    
    def fuzzy_match_ingredient(self, ingredient_name: str) -> Optional[str]:
        """Try fuzzy matching with Jaccard similarity (threshold >= 0.34)"""
        if not ingredient_name:
            return None
        
        canonical_name = canonicalize_name(ingredient_name)
        ingredient_tokens = set(canonical_name.split())
        if not ingredient_tokens:
            return None
        
        best_match = None
        best_score = 0.0
        
        for db_key in self.nutrition_database.keys():
            db_tokens = set(db_key.split())
            if not db_tokens:
                continue
            
            # Jaccard similarity
            intersection = len(ingredient_tokens & db_tokens)
            union = len(ingredient_tokens | db_tokens)
            score = intersection / union if union > 0 else 0
            
            # Use threshold of 0.34
            if score >= 0.34 and score > best_score:
                best_score = score
                best_match = db_key
        
        return best_match
    
    def get_category_nutrition(self, ingredient_name: str, grams: float) -> Optional[NutritionInfo]:
        """Get nutrition from category template"""
        category = _category_from_name(ingredient_name)
        if not category or category not in CATEGORY_TEMPLATES:
            return None
        
        template = CATEGORY_TEMPLATES[category]
        scale_factor = grams / 100.0
        
        return NutritionInfo(
            calories=template["kcal"] * scale_factor,
            protein=template["protein"] * scale_factor,
            carbs=template["carbs"] * scale_factor,
            fat=template["fat"] * scale_factor
        )
    
    def should_exclude_token(self, ingredient_name: str, grams: float) -> bool:
        """
        Check if token should be excluded from calculations and denominators
        Exclude A & B & <2g items
        """
        if not ingredient_name:
            return True
            
        ingredient_lower = ingredient_name.lower().strip()
        
        # Exclude A-list items (salt/pepper/water and filler phrases)
        for a_item in self.a_list:
            if a_item in ingredient_lower:
                return True
        
        # Exclude B-list items (micro-spices/herbs)
        for b_item in self.b_list:
            if b_item in ingredient_lower:
                return True
        
        # Exclude <2g items
        if grams < 2.0:
            return True
        
        # Exclude water even if quantified
        if 'water' in ingredient_lower and ('water' == ingredient_lower or ingredient_lower.endswith(' water')):
            return True
        
        return False
    
    def parse_ingredient_token(self, token: str) -> Tuple[float, str, str]:
        """
        Parse ingredient token to extract quantity, unit, and ingredient name
        Returns (grams, unit, ingredient_name)
        """
        if not token:
            return 0.0, '', ''
        
        # Clean token with improved normalization
        token = normalize_ing_token(token)
        
        # Pattern to match: quantity unit ingredient_name
        pattern = r'^(\d+(?:\.\d+)?(?:\s+\d+/\d+|\d+/\d+)?)\s+(\w+)\s+(.*?)$'
        match = re.match(pattern, token.strip(), re.IGNORECASE)
        
        if not match:
            return 0.0, '', token
        
        quantity_str = match.group(1).strip()
        unit = match.group(2).strip().lower()
        ingredient_name = match.group(3).strip()
        
        # Parse quantity (support fractions)
        quantity = self.parse_quantity_string(quantity_str)
        
        # Convert to grams
        grams = self.convert_to_grams(quantity, unit, ingredient_name)
        
        return grams, unit, ingredient_name
    
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
    
    def convert_to_grams(self, quantity: float, unit: str, ingredient_name: str = '') -> float:
        """Convert quantity and unit to grams with density and piece weight handling"""
        if quantity <= 0:
            return 0.0
        
        unit_lower = unit.lower().strip()
        ingredient_lower = ingredient_name.lower()
        
        # Mass units
        if unit_lower == 'g':
            return quantity
        if unit_lower == 'kg':
            return quantity * 1000.0
            
        # Volume units - use density
        if unit_lower == 'ml':
            density = 1.0
            for item, dens in self.densities.items():
                if item in ingredient_lower:
                    density = dens
                    break
            return quantity * density
        if unit_lower == 'l':
            density = 1.0
            for item, dens in self.densities.items():
                if item in ingredient_lower:
                    density = dens
                    break
            return quantity * 1000.0 * density
            
        # Cup uses density
        if unit_lower == 'cup':
            density = 1.0
            for item, dens in self.densities.items():
                if item in ingredient_lower:
                    density = dens
                    break
            return quantity * 240.0 * density
            
        # Pieces - use piece weights
        if unit_lower == 'pcs' or unit_lower == '':
            weight = 10.0  # default
            for item, w in self.piece_weights.items():
                if item in ingredient_lower:
                    weight = w
                    break
            return quantity * weight
            
        # Other units
        conversion_factor = self.to_grams.get(unit_lower, 10.0)
        return quantity * conversion_factor
    
    def find_nutrition_data(self, ingredient_name: str) -> Optional[NutritionInfo]:
        """
        Find nutrition data using improved matching with user aliases and fallbacks
        """
        if not ingredient_name:
            return None
        
        ingredient_lower = ingredient_name.lower().strip()
        
        # 1. Try exact match with mapped key
        mapped_key = self.map_to_db_key(ingredient_lower)
        if mapped_key in self.nutrition_database:
            return self.nutrition_database[mapped_key]
        
        # 2. Try exact match with original name
        if ingredient_lower in self.nutrition_database:
            return self.nutrition_database[ingredient_lower]
        
        # 3. Try fuzzy matching
        fuzzy_match = self.fuzzy_match_ingredient(ingredient_name)
        if fuzzy_match:
            return self.nutrition_database[fuzzy_match]
        
        # Track unmatched ingredients for reporting (only true misses)
        if ingredient_lower not in self.unmatched_ingredients:
            self.unmatched_ingredients[ingredient_lower] = 0
        self.unmatched_ingredients[ingredient_lower] += 1
        
        return None
    
    def calculate_ingredient_nutrition(self, ingredient_token: str) -> NutritionInfo:
        """
        Calculate nutrition for a single ingredient token with category fallback
        """
        if not ingredient_token:
            return NutritionInfo()
        
        # Parse ingredient token
        grams, unit, ingredient_name = self.parse_ingredient_token(ingredient_token)
        
        # Check exclusion criteria
        if self.should_exclude_token(ingredient_name, grams):
            return NutritionInfo()
        
        # Find nutrition data
        nutrition_per_100g = self.find_nutrition_data(ingredient_name)
        
        # If no direct match, try category fallback
        if not nutrition_per_100g:
            nutrition_per_100g = self.get_category_nutrition(ingredient_name, 100.0)
            if nutrition_per_100g:
                # Scale from 100g template to actual grams
                scale_factor = grams / 100.0
                return NutritionInfo(
                    calories=nutrition_per_100g.calories * scale_factor,
                    protein=nutrition_per_100g.protein * scale_factor,
                    carbs=nutrition_per_100g.carbs * scale_factor,
                    fat=nutrition_per_100g.fat * scale_factor
                )
            # Last resort: use category based on simple heuristics
            if grams >= 0.5:  # Be very inclusive
                scale_factor = grams / 100.0
                ingredient_lower = ingredient_name.lower()
                # Simple heuristics for category assignment
                if any(word in ingredient_lower for word in ['chicken', 'fish', 'meat', 'egg', 'beef', 'pork', 'mutton', 'protein', 'dal', 'lentil', 'prawn', 'crab', 'salmon', 'mackerel']):
                    template = CATEGORY_TEMPLATES["protein"]
                elif any(word in ingredient_lower for word in ['flour', 'rice', 'bread', 'noodle', 'pasta', 'roti', 'grain', 'semolina', 'rava', 'cornstarch']):
                    template = CATEGORY_TEMPLATES["staple"]
                elif any(word in ingredient_lower for word in ['milk', 'curd', 'yogurt', 'cheese']):
                    template = CATEGORY_TEMPLATES["dairy"]
                elif any(word in ingredient_lower for word in ['oil', 'ghee', 'butter']):
                    template = CATEGORY_TEMPLATES["fat"]
                elif any(word in ingredient_lower for word in ['sugar', 'sweet', 'honey', 'jaggery', 'vanilla', 'cashew', 'nuts']):
                    template = CATEGORY_TEMPLATES["seasoning"]
                elif any(word in ingredient_lower for word in ['onion', 'carrot', 'tomato', 'shallot', 'leek', 'potato']):
                    template = CATEGORY_TEMPLATES["vegetable"]
                # Even more aggressive fallback for anything with reasonable grams
                elif grams >= 5.0:  # Only assign for substantial amounts
                    template = CATEGORY_TEMPLATES["vegetable"]  # Default to vegetable for substantial amounts
                else:
                    template = None  # Don't assign for very small amounts
                
                if template:
                    return NutritionInfo(
                        calories=template["kcal"] * scale_factor,
                        protein=template["protein"] * scale_factor,
                        carbs=template["carbs"] * scale_factor,
                        fat=template["fat"] * scale_factor
                    )
            return NutritionInfo()
        
        # Scale to actual grams
        scale_factor = grams / 100.0
        
        scaled_nutrition = NutritionInfo(
            calories=nutrition_per_100g.calories * scale_factor,
            protein=nutrition_per_100g.protein * scale_factor,
            carbs=nutrition_per_100g.carbs * scale_factor,
            fat=nutrition_per_100g.fat * scale_factor
        )
        
        return scaled_nutrition
    
    def calculate_recipe_nutrition(self, ingredients_str: str, title: str = "") -> NutritionInfo:
        """
        Calculate total nutrition for a recipe
        """
        if not ingredients_str or pd.isna(ingredients_str):
            return NutritionInfo()
        
        # Split into ingredient tokens
        ingredient_tokens = [token.strip() for token in ingredients_str.split('|') if token.strip()]
        
        total_nutrition = NutritionInfo()
        included_tokens = []
        unmatched_materials = []
        
        for token in ingredient_tokens:
            # Parse the token to get ingredient name for unmatched tracking
            grams, unit, ingredient_name = self.parse_ingredient_token(token)
            
            # Skip A/B list items and <2g items from unmatched tracking
            if not self.should_exclude_token(ingredient_name, grams):
                ingredient_nutrition = self.calculate_ingredient_nutrition(token)
                
                # Only add if nutrition was found (non-zero)
                if ingredient_nutrition.calories > 0 or ingredient_nutrition.protein > 0 or ingredient_nutrition.carbs > 0 or ingredient_nutrition.fat > 0:
                    total_nutrition.calories += ingredient_nutrition.calories
                    total_nutrition.protein += ingredient_nutrition.protein
                    total_nutrition.carbs += ingredient_nutrition.carbs
                    total_nutrition.fat += ingredient_nutrition.fat
                    included_tokens.append(token)
                else:
                    # This is a material token that failed to get nutrition
                    if ingredient_name and grams >= 2.0:  # Only track substantial ingredients
                        unmatched_materials.append(self._primary_name_nutrition(ingredient_name))
        
        # If recipe has 0 calories, collect unmatched material tokens
        if total_nutrition.calories == 0 and unmatched_materials:
            self.collect_unmatched_nutrition(unmatched_materials, title)
        
        # Apply per-serving kcal guard and auto-skip outliers
        if total_nutrition.calories > 2500:
            # Log outlier for review and return empty nutrition
            self.log_nutrition_outlier(title or ingredients_str, total_nutrition.calories, included_tokens)
            return NutritionInfo()  # Skip outlier recipe
        
        return total_nutrition
    
    def _primary_name_nutrition(self, name: str) -> str:
        """Extract primary ingredient name by removing descriptors"""
        name_l = name.lower()
        name_l = re.sub(r'\b(fresh|large|small|medium|finely|coarsely|ground|crushed|dried|red|green|yellow|ripe|chopped|sliced|minced)\b', '', name_l)
        return re.sub(r'\s+', ' ', name_l).strip()
    
    def collect_unmatched_nutrition(self, unmatched_materials: List[str], title: str = ""):
        """Collect unmatched material ingredients for comprehensive reporting"""
        for ingredient in unmatched_materials:
            ingredient_head = ingredient.strip().lower()
            if ingredient_head and 'water' not in ingredient_head:  # Skip water
                # Apply canonicalization for suggested mapping
                canonical_head = canonicalize_name(ingredient_head)
                
                self.unmatched_reports.append({
                    'raw_token': ingredient,
                    'canonical_head': canonical_head,
                    'reason': 'no_nutrition_match',
                    'recipe_title': title
                })
    
    def write_unmatched_nutrition_report(self):
        """Write comprehensive unmatched nutrition report in standardized format"""
        try:
            from pathlib import Path
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            # Standardized column format for learner compatibility
            fieldnames = ['raw_token', 'ingredient_head', 'canonical_suggested', 'reason', 'recipe_title']
            output_file = reports_dir / 'unmatched_nutrition.csv'
            
            if self.unmatched_reports:
                # Enhance existing reports with suggested matches
                enhanced_reports = []
                for report in self.unmatched_reports:
                    enhanced = {
                        'raw_token': report.get('raw_token', ''),
                        'ingredient_head': canonicalize_name(report.get('canonical_head', report.get('raw_token', ''))),
                        'canonical_suggested': self.suggest_canonical(report.get('canonical_head', report.get('raw_token', ''))),
                        'reason': report.get('reason', 'no_match'),
                        'recipe_title': report.get('recipe_title', '')
                    }
                    enhanced_reports.append(enhanced)
                
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(enhanced_reports)
                logger.info(f"Wrote {len(enhanced_reports)} unmatched nutrition entries to {output_file}")
            else:
                # Write empty file with headers
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
        except Exception as e:
            logger.warning(f"Could not write unmatched nutrition report: {e}")
    
    def suggest_canonical(self, ingredient_name: str) -> str:
        """Suggest canonical name from nutrition database using fuzzy matching"""
        if not ingredient_name or not hasattr(self, 'nutrition_database'):
            return ''
        
        canonical = canonicalize_name(ingredient_name)
        # Simple token overlap matching against nutrition DB
        db_names = list(self.nutrition_database.keys())
        
        canonical_tokens = set(canonical.split())
        best_match = ''
        best_score = 0.0
        
        for db_name in db_names:
            db_tokens = set(canonicalize_name(db_name).split())
            if db_tokens and canonical_tokens:
                # Jaccard similarity
                intersection = len(canonical_tokens & db_tokens)
                union = len(canonical_tokens | db_tokens)
                score = intersection / union if union > 0 else 0.0
                
                if score > best_score and score >= 0.34:
                    best_score = score
                    best_match = db_name
        
        return best_match
    
    def log_nutrition_outlier(self, title: str, calories: float, tokens: List[str]):
        """
        Log high-calorie outliers to CSV for review
        """
        try:
            file_exists = os.path.exists('reports/nutrition_outliers.csv')
            with open('reports/nutrition_outliers.csv', 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # Write header if file is new
                if not file_exists:
                    writer.writerow(['recipe_title', 'calories', 'included_tokens'])
                writer.writerow([title, round(calories, 1), ' | '.join(tokens)])
        except Exception as e:
            logger.warning(f"Could not log nutrition outlier: {e}")
    
    def process_csv_file(self, input_file: str, output_file: str = None,
                        ingredients_col: str = 'ingredients_per_person'):
        """
        Process CSV file and add nutrition columns
        """
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
            
            # Process recipes
            for idx, row in df.iterrows():
                ingredients = row.get(ingredients_col, '')
                title = row.get('title', f'Recipe {idx+1}')
                nutrition = self.calculate_recipe_nutrition(ingredients, title)
                
                df.at[idx, 'calories'] = round(nutrition.calories, 1)
                df.at[idx, 'protein_g'] = round(nutrition.protein, 1)
                df.at[idx, 'carbs_g'] = round(nutrition.carbs, 1)
                df.at[idx, 'fat_g'] = round(nutrition.fat, 1)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} recipes")
            
            # Save results
            if output_file is None:
                base_name = input_file.replace('.csv', '')
                output_file = f"{base_name}_improved_nutrition.csv"
            
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_file}")
            
            # Write comprehensive unmatched report
            self.write_unmatched_nutrition_report()
            
            # Generate coverage report
            covered = int((df['calories'] > 0).sum())
            total = int(len(df))
            coverage_pct = round(covered/total*100, 1)
            report = {
                "coverage": coverage_pct / 100.0,
                "matched": covered,
                "total": total
            }
            os.makedirs('reports', exist_ok=True)
            with open("reports/nutrition_coverage.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Nutrition coverage: {coverage_pct}% ({covered}/{total} recipes)")
            
            # Show summary
            self.show_summary(df)
            
            return df, coverage_pct
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def show_summary(self, df: pd.DataFrame):
        """Display nutrition summary"""
        print("\n" + "="*70)
        print("NUTRITION CALCULATION SUMMARY")
        print("="*70)
        
        avg_calories = df['calories'].mean()
        avg_protein = df['protein_g'].mean()
        avg_carbs = df['carbs_g'].mean()
        avg_fat = df['fat_g'].mean()
        
        print(f"\nAVERAGE NUTRITION PER RECIPE (per person):")
        print(f"  Calories: {avg_calories:.1f} kcal")
        print(f"  Protein:  {avg_protein:.1f} g")
        print(f"  Carbs:    {avg_carbs:.1f} g")
        print(f"  Fat:      {avg_fat:.1f} g")
        
        # Count recipes with valid nutrition
        valid_recipes = len(df[df['calories'] > 0])
        print(f"\nRecipes with nutrition data: {valid_recipes}/{len(df)} ({valid_recipes/len(df)*100:.1f}%)")
        
        print("="*70)

def main():
    """Main entry point for standalone execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python nutrition_calculator.py <input_csv> [output_csv] [ingredients_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    ingredients_col = sys.argv[3] if len(sys.argv) > 3 else 'ingredients_per_person'
    
    calculator = NutritionCalculator()
    result, coverage_pct = calculator.process_csv_file(input_file, output_file, ingredients_col)
    
    # Print coverage for pipeline use
    print(f"NUTRITION_COVERAGE: {coverage_pct}")
    return coverage_pct

if __name__ == "__main__":
    main()