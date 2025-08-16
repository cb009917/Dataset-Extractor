import pandas as pd
import numpy as np
from transformers import pipeline
import logging
import time
from typing import Tuple, List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecipeClassifier:
    def __init__(self, max_workers=None):
        """
        Initialize the recipe classifier using Hugging Face transformers
        """
        logger.info("🤖 Initializing Hugging Face classification model...")
        
        try:
            # Use zero-shot classification model (works great for this task)
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.has_gpu() else -1  # Use GPU if available
            )
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("💡 Try installing: pip install torch transformers")
            raise
        
        # Threading configuration
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.progress_lock = Lock()
        self.processed_count = 0
        
        # Define classification labels
        self.meal_labels = ["breakfast", "lunch", "dinner"]
        self.course_labels = ["main", "side", "snack", "beverage", "dessert", "soup"]
        
        # Sri Lankan specific enhancements
        self.sri_lankan_meal_rules = {
            'breakfast': ['hopper', 'string hopper', 'kiribath', 'milk rice', 'pittu', 'roti'],
            'lunch': ['rice and curry', 'kottu', 'fried rice', 'lamprais'],
            'dinner': ['curry', 'devilled', 'fish curry', 'chicken curry']
        }
        
        self.sri_lankan_course_rules = {
            'dessert': ['wattalappam', 'custard', 'sweet', 'kokis', 'kavum', 'cake', 'pudding', 'ice cream'],
            'side': ['sambol', 'pol sambol', 'seeni sambol', 'pickle', 'chutney', 'salad', 'moju', 'achcharu', 'papadam', 'curry', 'chicken curry', 'fish curry', 'beef curry', 'pork curry', 'dal curry', 'vegetable curry', 'beetroot curry', 'potato curry', 'mutton curry', 'prawn curry', 'sodhi', 'fish sodhi', 'chicken sodhi', 'vegetable sodhi', 'brinjal sodhi', 'okra sodhi'],
            'main': ['rice and curry', 'kottu', 'hoppers', 'pittu', 'lamprais', 'biryani', 'fried rice', 'noodles'],
            'beverage': ['tea', 'coffee', 'juice', 'drink', 'smoothie', 'lassi', 'shake', 'water', 'milk', 'soda', 'beer', 'wine', 'cocktail', 'faluda', 'thambili', 'king coconut', 'ice tea', 'iced tea', 'cold drink'],
            'snack': ['rolls', 'patties', 'wade', 'isso wade', 'fish bun', 'short eats', 'cutlet', 'samosa', 'roti', 'sandwich', 'toast'],
            'soup': ['soup', 'broth', 'rasam', 'clear soup', 'vegetable soup', 'chicken soup', 'mushroom soup', 'lentil soup', 'bone broth', 'stock']
        }
    
    def has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def preprocess_text(self, title: str, ingredients: str, instructions: str) -> str:
        """
        Combine and preprocess recipe text for classification
        """
        # Handle NaN values
        title = str(title) if pd.notna(title) else ""
        ingredients = str(ingredients) if pd.notna(ingredients) else ""
        instructions = str(instructions) if pd.notna(instructions) else ""
        
        # Combine text (title is most important, then ingredients)
        combined_text = f"{title}. {ingredients[:200]}..."  # Limit length for efficiency
        
        return combined_text.strip()
    
    def apply_sri_lankan_rules(self, title: str, ml_prediction: str, rule_type: str) -> str:
        """
        Apply Sri Lankan specific rules to enhance ML predictions
        """
        title_lower = title.lower()
        rules = self.sri_lankan_meal_rules if rule_type == 'meal' else self.sri_lankan_course_rules
        
        # Priority-based rule checking for course type
        if rule_type == 'course':
            # Check beverage keywords (high priority)
            beverage_keywords = rules.get('beverage', [])
            for keyword in beverage_keywords:
                if keyword in title_lower:
                    logger.debug(f"Beverage rule applied: '{keyword}' -> beverage")
                    return 'beverage'
            
            # Check other categories in order of specificity
            priority_order = ['soup', 'dessert', 'snack', 'side', 'main']
            for category in priority_order:
                keywords = rules.get(category, [])
                for keyword in keywords:
                    if keyword in title_lower:
                        logger.debug(f"Sri Lankan rule applied: '{keyword}' -> {category}")
                        return category
        else:
            # For meal type, check all categories
            for category, keywords in rules.items():
                for keyword in keywords:
                    if keyword in title_lower:
                        logger.debug(f"Sri Lankan rule applied: '{keyword}' -> {category}")
                        return category.replace('_', ' ')
        
        # If no specific rule matches, return ML prediction
        return ml_prediction
    
    def classify_meal_type(self, title: str, ingredients: str, instructions: str) -> str:
        """
        Classify recipe into breakfast, lunch, or dinner
        """
        try:
            text = self.preprocess_text(title, ingredients, instructions)
            
            # Get ML prediction
            result = self.classifier(text, self.meal_labels)
            ml_prediction = result['labels'][0]
            
            # Apply Sri Lankan specific rules
            final_prediction = self.apply_sri_lankan_rules(title, ml_prediction, 'meal')
            
            logger.debug(f"Meal classification: '{title}' -> {final_prediction} (ML: {ml_prediction})")
            return final_prediction
            
        except Exception as e:
            logger.warning(f"Error classifying meal type for '{title}': {e}")
            return "lunch"  # Default fallback
    
    def classify_course_type(self, title: str, ingredients: str, instructions: str) -> str:
        """
        Classify recipe into main dish, side dish, or dessert
        """
        try:
            text = self.preprocess_text(title, ingredients, instructions)
            
            # Get ML prediction
            result = self.classifier(text, self.course_labels)
            ml_prediction = result['labels'][0]
            
            # Apply Sri Lankan specific rules
            final_prediction = self.apply_sri_lankan_rules(title, ml_prediction, 'course')
            
            logger.debug(f"Course classification: '{title}' -> {final_prediction} (ML: {ml_prediction})")
            return final_prediction
            
        except Exception as e:
            logger.warning(f"Error classifying course type for '{title}': {e}")
            return "main"  # Default fallback
    
    def classify_recipe(self, title: str, ingredients: str, instructions: str) -> Tuple[str, str]:
        """
        Classify both meal type and course type for a single recipe
        """
        meal_type = self.classify_meal_type(title, ingredients, instructions)
        course_type = self.classify_course_type(title, ingredients, instructions)
        
        return meal_type, course_type
    
    def classify_recipe_batch(self, recipe_batch: List[Tuple[int, str, str, str]], total_recipes: int) -> List[Tuple[int, str, str]]:
        """
        Classify a batch of recipes (thread worker function)
        """
        results = []
        
        for idx, title, ingredients, instructions in recipe_batch:
            try:
                meal_type, course_type = self.classify_recipe(title, ingredients, instructions)
                results.append((idx, meal_type, course_type))
                
                # Thread-safe progress tracking
                with self.progress_lock:
                    self.processed_count += 1
                    if self.processed_count % 5 == 0 or self.processed_count == total_recipes:
                        logger.info(f"Progress: {self.processed_count}/{total_recipes} recipes classified ({(self.processed_count/total_recipes)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error processing recipe {idx+1}: {e}")
                results.append((idx, 'lunch', 'main'))
        
        return results
    
    def classify_dataframe(self, df: pd.DataFrame, title_col: str = 'title', 
                          ingredients_col: str = 'ingredients', 
                          instructions_col: str = 'instructions') -> pd.DataFrame:
        """
        Add meal_type and course_type columns to a DataFrame of recipes using multithreading
        """
        total_recipes = len(df)
        logger.info(f"🍽️ Classifying {total_recipes} recipes using {self.max_workers} threads...")
        
        # Initialize new columns
        df['meal_type'] = ''
        df['course_type'] = ''
        
        # Reset progress counter
        self.processed_count = 0
        
        # Prepare recipe data for threading
        recipe_data = []
        for idx, row in df.iterrows():
            title = row.get(title_col, '')
            ingredients = row.get(ingredients_col, '')
            instructions = row.get(instructions_col, '')
            recipe_data.append((idx, title, ingredients, instructions))
        
        # Split recipes into batches for threads
        batch_size = max(1, total_recipes // self.max_workers)
        recipe_batches = []
        for i in range(0, total_recipes, batch_size):
            batch = recipe_data[i:i + batch_size]
            recipe_batches.append(batch)
        
        logger.info(f"📦 Split into {len(recipe_batches)} batches of ~{batch_size} recipes each")
        
        # Process batches with ThreadPoolExecutor
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.classify_recipe_batch, batch, total_recipes): batch_idx
                for batch_idx, batch in enumerate(recipe_batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Update DataFrame with results
                    for idx, meal_type, course_type in batch_results:
                        df.at[idx, 'meal_type'] = meal_type
                        df.at[idx, 'course_type'] = course_type
                        
                        # Log sample results
                        if idx % 20 == 0:  # Show every 20th recipe
                            title = df.at[idx, title_col]
                            logger.info(f"✅ {idx+1}. '{title[:30]}...' -> {meal_type} | {course_type}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
        
        elapsed_time = time.time() - start_time
        recipes_per_second = total_recipes / elapsed_time
        
        logger.info("🎉 Classification completed!")
        logger.info(f"⚡ Processed {total_recipes} recipes in {elapsed_time:.1f}s ({recipes_per_second:.1f} recipes/sec)")
        
        return df
    
    def classify_csv_file(self, input_file: str, output_file: str = None,
                         title_col: str = 'title', ingredients_col: str = 'ingredients', 
                         instructions_col: str = 'instructions'):
        """
        Load CSV, classify recipes, and save results
        """
        try:
            # Load CSV
            logger.info(f"📂 Loading recipes from {input_file}")
            df = pd.read_csv(input_file, encoding='utf-8')
            
            # Validate columns
            required_cols = [title_col, ingredients_col, instructions_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"❌ Missing columns: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                return
            
            logger.info(f"✅ Loaded {len(df)} recipes")
            
            # Classify recipes
            df_classified = self.classify_dataframe(df, title_col, ingredients_col, instructions_col)
            
            # Generate output filename if not provided
            if output_file is None:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}_classified.csv"
            
            # Save results
            df_classified.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"💾 Classified recipes saved to {output_file}")
            
            # Show summary statistics
            self.show_classification_summary(df_classified)
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV file: {e}")
            raise
    
    def show_classification_summary(self, df: pd.DataFrame):
        """
        Display classification results summary
        """
        print("\n" + "="*50)
        print("📊 CLASSIFICATION SUMMARY")
        print("="*50)
        
        # Meal type distribution
        print("\n🍽️ MEAL TYPE DISTRIBUTION:")
        meal_counts = df['meal_type'].value_counts()
        for meal, count in meal_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {meal.title()}: {count} recipes ({percentage:.1f}%)")
        
        # Course type distribution
        print("\n🥘 COURSE TYPE DISTRIBUTION:")
        course_counts = df['course_type'].value_counts()
        for course, count in course_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {course.title()}: {count} recipes ({percentage:.1f}%)")
        
        # Sample classifications
        print("\n📝 SAMPLE CLASSIFICATIONS:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            print(f"{i+1}. {row['title'][:40]}...")
            print(f"   → {row['meal_type']} | {row['course_type']}")
        
        print("="*50)

# Standalone execution
if __name__ == "__main__":
    print("🇱🇰 Sri Lankan Recipe Classifier")
    print("=" * 40)
    print("Uses Hugging Face BART model + Sri Lankan food rules")
    print("Adds meal_type and course_type columns to your CSV")
    print()
    
    # Get input file
    input_file = input("Enter CSV file path (or drag & drop file): ").strip().strip('"')
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        exit(1)
    
    # Optional: Custom column names
    print("\nColumn names (press Enter for defaults):")
    title_col = input("Title column name (default: 'title'): ").strip() or 'title'
    ingredients_col = input("Ingredients column name (default: 'ingredients'): ").strip() or 'ingredients'
    instructions_col = input("Instructions column name (default: 'instructions'): ").strip() or 'instructions'
    
    # Optional: Custom output file
    output_file = input("Output file path (press Enter for auto-generated): ").strip()
    if not output_file:
        output_file = None
    
    print(f"\n🚀 Starting classification...")
    print("⏰ This may take a few minutes for large files...")
    
    try:
        # Initialize classifier with multithreading
        logger.info(f"🚀 Using {multiprocessing.cpu_count()} CPU cores for multithreaded classification")
        classifier = RecipeClassifier(max_workers=min(8, multiprocessing.cpu_count()))
        
        # Process CSV
        classifier.classify_csv_file(
            input_file=input_file,
            output_file=output_file,
            title_col=title_col,
            ingredients_col=ingredients_col,
            instructions_col=instructions_col
        )
        
        print("\n🎉 Classification completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Classification interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have installed: pip install transformers torch pandas")