import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import re
from typing import List, Dict, Tuple

class SriLankanIngredientSeasonalityClassifier:
    def __init__(self, csv_file_path=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.csv_file_path = csv_file_path
        
        # Sri Lankan seasonal keywords for feature extraction
        # Based on Sri Lankan climate: Dry Season (Dec-Mar), Southwest Monsoon (May-Sep), Northeast Monsoon (Oct-Jan), Inter-monsoon (Apr, Oct)
        self.seasonal_keywords = {
            'Dry_Season': ['dry', 'hot', 'harvest', 'stored', 'preserved', 'coconut', 'cashew', 'mango', 'avocado', 'jackfruit', 'papaya', 'passion'],
            'Southwest_Monsoon': ['wet', 'monsoon', 'fresh', 'green', 'leafy', 'rice', 'paddy', 'kurukan', 'gotukola', 'mukunuwenna', 'kankun', 'nivithi'],
            'Northeast_Monsoon': ['cool', 'wet', 'root', 'tuber', 'yam', 'sweet potato', 'cassava', 'beetroot', 'carrot', 'radish', 'leeks'],
            'Inter_Monsoon': ['transition', 'mixed', 'varied', 'seasonal', 'changing', 'brinjal', 'okra', 'beans', 'drumstick', 'bitter gourd'],
            'Year_Round': ['year-round', 'always', 'staple', 'basic', 'common', 'rice', 'coconut', 'onion', 'garlic', 'ginger', 'chili', 'curry', 'dhal', 'lentils']
        }
        
        # Sri Lankan ingredient training data with seasonal patterns
        self.sample_data = [
            # Dry Season (December - March): Peak harvest for tree fruits and preserved items
            ('mango', 'Dry_Season'), ('avocado', 'Dry_Season'), ('jackfruit', 'Dry_Season'), ('papaya', 'Dry_Season'), 
            ('cashew', 'Dry_Season'), ('coconut', 'Dry_Season'), ('passion fruit', 'Dry_Season'), ('wood apple', 'Dry_Season'),
            ('rambutan', 'Dry_Season'), ('durian', 'Dry_Season'), ('tamarind', 'Dry_Season'), ('lime', 'Dry_Season'),
            
            # Southwest Monsoon (May - September): Fresh greens and rice cultivation
            ('rice', 'Southwest_Monsoon'), ('gotukola', 'Southwest_Monsoon'), ('mukunuwenna', 'Southwest_Monsoon'), ('kankun', 'Southwest_Monsoon'),
            ('nivithi', 'Southwest_Monsoon'), ('kurakan', 'Southwest_Monsoon'), ('sarana', 'Southwest_Monsoon'), ('hathawariya', 'Southwest_Monsoon'),
            ('green leaves', 'Southwest_Monsoon'), ('spinach', 'Southwest_Monsoon'), ('lettuce', 'Southwest_Monsoon'), ('cabbage', 'Southwest_Monsoon'),
            
            # Northeast Monsoon (October - January): Root vegetables and tubers
            ('sweet potato', 'Northeast_Monsoon'), ('cassava', 'Northeast_Monsoon'), ('yam', 'Northeast_Monsoon'), ('beetroot', 'Northeast_Monsoon'),
            ('carrot', 'Northeast_Monsoon'), ('radish', 'Northeast_Monsoon'), ('leeks', 'Northeast_Monsoon'), ('turnip', 'Northeast_Monsoon'),
            ('potato', 'Northeast_Monsoon'), ('innala', 'Northeast_Monsoon'), ('kiriala', 'Northeast_Monsoon'), ('kohila', 'Northeast_Monsoon'),
            
            # Inter-monsoon (April & October): Transitional vegetables
            ('brinjal', 'Inter_Monsoon'), ('okra', 'Inter_Monsoon'), ('beans', 'Inter_Monsoon'), ('drumstick', 'Inter_Monsoon'),
            ('bitter gourd', 'Inter_Monsoon'), ('snake gourd', 'Inter_Monsoon'), ('ridge gourd', 'Inter_Monsoon'), ('pumpkin', 'Inter_Monsoon'),
            ('ash plantain', 'Inter_Monsoon'), ('green banana', 'Inter_Monsoon'), ('luffa', 'Inter_Monsoon'), ('wing beans', 'Inter_Monsoon'),
            
            # Year-round: Staples and always available ingredients
            ('onion', 'Year_Round'), ('garlic', 'Year_Round'), ('ginger', 'Year_Round'), ('chili', 'Year_Round'),
            ('curry leaves', 'Year_Round'), ('dhal', 'Year_Round'), ('lentils', 'Year_Round'), ('tomato', 'Year_Round'),
            ('cucumber', 'Year_Round'), ('green chili', 'Year_Round'), ('red onion', 'Year_Round'), ('coriander', 'Year_Round'),
            ('mint', 'Year_Round'), ('basil', 'Year_Round'), ('pandan', 'Year_Round'), ('lemongrass', 'Year_Round')
        ]
    
    def extract_features(self, ingredients: List[str]) -> np.ndarray:
        """Extract features from ingredient names and descriptions"""
        features = []
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            feature_vector = []
            
            # Seasonal keyword features
            for season, keywords in self.seasonal_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in ingredient_lower)
                feature_vector.append(keyword_count)
            
            # Text length feature
            feature_vector.append(len(ingredient))
            
            # Number of words feature
            feature_vector.append(len(ingredient.split()))
            
            # Has color words
            color_words = ['red', 'green', 'yellow', 'orange', 'purple', 'white', 'black', 'pink']
            has_color = 1 if any(color in ingredient_lower for color in color_words) else 0
            feature_vector.append(has_color)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def load_csv_data(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load training data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            # Assume CSV has columns: 'ingredient' and 'season'
            if 'ingredient' not in df.columns or 'season' not in df.columns:
                raise ValueError("CSV must have 'ingredient' and 'season' columns")
            return list(zip(df['ingredient'].values, df['season'].values))
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Using default sample data instead")
            return self.sample_data
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from CSV file or sample ingredients"""
        if self.csv_file_path and pd.io.common.file_exists(self.csv_file_path):
            training_data = self.load_csv_data(self.csv_file_path)
            print(f"Loaded {len(training_data)} ingredients from CSV")
        else:
            training_data = self.sample_data
            print(f"Using default sample data with {len(training_data)} ingredients")
        
        ingredients, seasons = zip(*training_data)
        
        # Extract numerical features
        numerical_features = self.extract_features(list(ingredients))
        
        # Extract text features
        text_features = self.vectorizer.fit_transform(ingredients)
        
        # Combine features
        X = np.hstack([numerical_features, text_features.toarray()])
        
        # Encode labels
        y = self.label_encoder.fit_transform(seasons)
        
        return X, y
    
    def train(self) -> Dict[str, float]:
        """Train the seasonality classification model"""
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        }
    
    def predict_seasonality(self, ingredient: str) -> Dict[str, float]:
        """Predict the seasonality of a single ingredient"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        numerical_features = self.extract_features([ingredient])
        text_features = self.vectorizer.transform([ingredient])
        X = np.hstack([numerical_features, text_features.toarray()])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Map to season names
        season_probs = {}
        for i, season in enumerate(self.label_encoder.classes_):
            season_probs[season] = probabilities[i]
        
        return season_probs
    
    def predict_multiple(self, ingredients: List[str]) -> List[Dict[str, float]]:
        """Predict seasonality for multiple ingredients"""
        return [self.predict_seasonality(ingredient) for ingredient in ingredients]
    
    def get_most_likely_season(self, ingredient: str) -> str:
        """Get the most likely season for an ingredient"""
        probs = self.predict_seasonality(ingredient)
        return max(probs, key=probs.get)
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'seasonal_keywords': self.seasonal_keywords
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.seasonal_keywords = model_data['seasonal_keywords']
        self.is_trained = True
    
    def add_seasonality_to_csv(self, input_csv_path: str, output_csv_path: str = None, ingredient_column: str = None):
        """Add seasonality predictions to an existing ingredient CSV file"""
        if not self.is_trained:
            print("Training model first...")
            self.train()
        
        try:
            # Read the existing CSV
            df = pd.read_csv(input_csv_path)
            print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Auto-detect ingredient column if not specified
            if ingredient_column is None:
                # Look for common ingredient column names
                possible_names = ['ingredient', 'ingredients', 'name', 'item', 'food']
                ingredient_column = None
                for col in df.columns:
                    if col.lower() in possible_names:
                        ingredient_column = col
                        break
                
                if ingredient_column is None:
                    print("\nAvailable columns:")
                    for i, col in enumerate(df.columns):
                        print(f"{i+1}. {col}")
                    
                    while True:
                        try:
                            choice = input(f"\nWhich column contains the ingredients? (1-{len(df.columns)}): ")
                            idx = int(choice) - 1
                            if 0 <= idx < len(df.columns):
                                ingredient_column = df.columns[idx]
                                break
                            else:
                                print("Invalid choice. Please try again.")
                        except ValueError:
                            print("Please enter a valid number.")
            
            print(f"Using '{ingredient_column}' as ingredient column")
            
            # Get predictions for all ingredients
            ingredients = df[ingredient_column].astype(str).tolist()
            predictions = []
            prediction_probabilities = []
            
            print("Predicting seasonality for all ingredients...")
            for ingredient in ingredients:
                try:
                    season_probs = self.predict_seasonality(ingredient)
                    most_likely_season = max(season_probs, key=season_probs.get)
                    confidence = season_probs[most_likely_season]
                    
                    predictions.append(most_likely_season)
                    prediction_probabilities.append(confidence)
                except:
                    predictions.append("Unknown")
                    prediction_probabilities.append(0.0)
            
            # Add new columns
            df['seasonality'] = predictions
            df['seasonality_confidence'] = [round(prob, 3) for prob in prediction_probabilities]
            
            # Save to output file
            if output_csv_path is None:
                # Create output filename by adding '_with_seasonality' to the input filename
                input_name = input_csv_path.rsplit('.', 1)[0]
                output_csv_path = f"{input_name}_with_seasonality.csv"
            
            df.to_csv(output_csv_path, index=False)
            print(f"\nSeasonality predictions added successfully!")
            print(f"Output saved to: {output_csv_path}")
            
            # Show detailed summary
            season_counts = df['seasonality'].value_counts()
            total_ingredients = len(df)
            
            print(f"\n" + "="*60)
            print(f"SEASONALITY DISTRIBUTION ({total_ingredients} total ingredients)")
            print(f"="*60)
            
            # Season descriptions
            season_descriptions = {
                'Dry_Season': 'December - March (Hot & Dry)',
                'Southwest_Monsoon': 'May - September (Wet Season)', 
                'Northeast_Monsoon': 'October - January (Cool & Wet)',
                'Inter_Monsoon': 'April & October (Transition)',
                'Year_Round': 'Available All Year'
            }
            
            for season, count in season_counts.items():
                percentage = (count / total_ingredients) * 100
                description = season_descriptions.get(season, '')
                bar = "█" * min(int(percentage / 2), 50)  # Visual bar
                print(f"{season:17s} | {count:3d} ({percentage:5.1f}%) {bar}")
                if description:
                    print(f"                   | {description}")
                print()
            
            print("="*60)
            
            return output_csv_path
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None

def main():
    """Interactive CSV seasonality predictor for Sri Lankan ingredients"""
    print("=== Sri Lankan Ingredient Seasonality Classifier ===\n")
    
    # Ask user for CSV file path
    while True:
        csv_path = input("Enter the path to your ingredient CSV file: ").strip()
        if csv_path:
            # Remove quotes if user added them
            csv_path = csv_path.strip('"').strip("'")
            try:
                # Test if file exists and can be read
                pd.read_csv(csv_path, nrows=1)
                break
            except FileNotFoundError:
                print(f"Error: File not found at '{csv_path}'. Please check the path and try again.\n")
            except Exception as e:
                print(f"Error reading file: {e}\nPlease try again.\n")
        else:
            print("Please enter a valid file path.\n")
    
    # Create classifier and train
    classifier = SriLankanIngredientSeasonalityClassifier()
    print("\nTraining the Sri Lankan ingredient seasonality classifier...")
    results = classifier.train()
    print(f"Training completed with accuracy: {results['accuracy']:.3f}")
    
    # Process the CSV file
    print(f"\nProcessing your ingredient CSV file...")
    output_path = classifier.add_seasonality_to_csv(csv_path)
    
    if output_path:
        print(f"\n✅ Successfully processed your ingredients!")
        print(f"📁 Original file: {csv_path}")
        print(f"📁 Output file: {output_path}")
        print("\nThe output file contains:")
        print("  - All your original data")
        print("  - 'seasonality' column with predicted seasons")
        print("  - 'seasonality_confidence' column with confidence scores")
        
        # Save the trained model
        model_path = 'srilankan_seasonality_model.joblib'
        classifier.save_model(model_path)
        print(f"  - Trained model saved as: {model_path}")
    else:
        print("\n❌ Failed to process the CSV file. Please check the file format and try again.")

if __name__ == "__main__":
    main()