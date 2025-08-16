import requests
from bs4 import BeautifulSoup
import json
import time
import random
import csv
from urllib.parse import urljoin
import re
from typing import Dict, List, Optional
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetedSriLankanScraper:
    def __init__(self, max_recipes=500, delay_range=(2, 4)):
        """
        Targeted scraper that looks for authentic Sri Lankan recipes using specific search strategies
        """
        self.max_recipes = max_recipes
        self.delay_range = delay_range
        self.base_url = "https://cookpad.com"
        self.scraped_recipes = []
        self.scraped_urls = set()
        self.failed_urls = []
        
        # Create output directory
        self.output_dir = f"sri_lankan_targeted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Authentic Sri Lankan dish names for targeted searching
        self.authentic_dishes = [
            "hoppers", "string hoppers", "kottu roti", "kottu", 
            "pol sambol", "coconut sambol", "kaha bath", "yellow rice",
            "kiribath", "milk rice", "pittu", "ambul thiyal", "sour fish curry",
            "parippu", "dhal curry", "kukul mas curry", "chicken curry sri lankan",
            "fish curry sri lankan", "devilled prawns", "devilled chicken",
            "gotukola sambol", "murunga curry", "drumstick curry",
            "wattalappam", "coconut custard", "seeni sambol", "onion sambol",
            "batu moju", "brinjal moju", "coconut roti", "pol roti",
            "lamprais", "sri lankan rice", "curry leaves", "pandan leaf"
        ]
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def debug_recipe_structure(self, recipe_url: str):
        """
        Debug function to inspect the actual HTML structure of a recipe page
        """
        try:
            response = self.session.get(recipe_url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print(f"\n🔍 DEBUG: Analyzing structure of {recipe_url}")
            print("=" * 60)
            
            # Look for ingredients section
            print("\n📝 INGREDIENTS SECTION ANALYSIS:")
            ingredients_candidates = [
                soup.select('.ingredients'),
                soup.select('[class*="ingredient"]'),
                soup.select('ul'),
                soup.select('ol')
            ]
            
            for i, candidates in enumerate(ingredients_candidates):
                if candidates:
                    print(f"  Found {len(candidates)} elements with selector {i+1}")
                    for j, elem in enumerate(candidates[:2]):  # Show first 2
                        text_preview = elem.get_text()[:100].replace('\n', ' ')
                        print(f"    {j+1}: {text_preview}...")
                        print(f"    Classes: {elem.get('class', [])}")
                        print(f"    ID: {elem.get('id', 'None')}")
            
            # Look for instructions section  
            print("\n📋 INSTRUCTIONS SECTION ANALYSIS:")
            instructions_candidates = [
                soup.select('.steps'),
                soup.select('[class*="step"]'),
                soup.select('[class*="instruction"]'),
                soup.select('ol li'),
                soup.select('.cooking-instructions')
            ]
            
            for i, candidates in enumerate(instructions_candidates):
                if candidates:
                    print(f"  Found {len(candidates)} elements with selector {i+1}")
                    for j, elem in enumerate(candidates[:2]):
                        text_preview = elem.get_text()[:100].replace('\n', ' ')
                        print(f"    {j+1}: {text_preview}...")
                        print(f"    Classes: {elem.get('class', [])}")
            
            # Look for serving section
            print("\n👥 SERVING SIZE ANALYSIS:")
            serving_candidates = [
                soup.select('[class*="serving"]'),
                soup.select('[id*="serving"]'),
                soup.select('.servings'),
                soup.select('[class*="yield"]')
            ]
            
            for i, candidates in enumerate(serving_candidates):
                if candidates:
                    print(f"  Found {len(candidates)} elements with selector {i+1}")
                    for elem in candidates:
                        text_preview = elem.get_text()[:50].replace('\n', ' ')
                        print(f"    Text: {text_preview}")
                        print(f"    Classes: {elem.get('class', [])}")
                        print(f"    ID: {elem.get('id', 'None')}")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"Debug error: {e}")
    
    def get_targeted_recipe_urls(self) -> List[str]:
        """
        Get recipe URLs using targeted searches for authentic Sri Lankan dishes
        """
        all_urls = []
        
        for dish in self.authentic_dishes:
            try:
                # Search for exact dish names
                search_url = f"https://cookpad.com/eng/search/{dish.replace(' ', '%20')}"
                logger.info(f"Searching for authentic dish: {dish}")
                
                response = self.session.get(search_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                recipe_links = soup.select('a[href*="/recipes/"]')
                
                dish_urls = []
                for link in recipe_links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        if self.is_valid_recipe_url(full_url):
                            all_urls.append(full_url)
                            dish_urls.append(full_url)
                
                logger.info(f"  Found {len(dish_urls)} recipes for '{dish}'")
                
                # Delay between searches
                time.sleep(random.uniform(1, 2))
                
                # Stop if we have enough URLs
                if len(all_urls) >= self.max_recipes * 2:
                    logger.info(f"Collected enough URLs ({len(all_urls)}), stopping search")
                    break
                    
            except Exception as e:
                logger.error(f"Error searching for '{dish}': {e}")
                continue
        
        # Remove duplicates
        unique_urls = list(dict.fromkeys(all_urls))
        logger.info(f"Total unique recipe URLs found: {len(unique_urls)}")
        return unique_urls
    
    def is_valid_recipe_url(self, url: str) -> bool:
        """Check if URL is valid"""
        return (
            'cookpad.com' in url and
            '/recipes/' in url and
            url not in self.scraped_urls
        )
    
    def extract_ingredients_flexible(self, soup: BeautifulSoup) -> List[str]:
        """
        Flexible ingredient extraction using multiple strategies
        """
        ingredients = []
        
        # Strategy 1: Try your provided selector
        ingredients_section = soup.select_one('.ingredients')
        if ingredients_section:
            items = ingredients_section.select('li, p, div')
            for item in items:
                text = item.get_text().strip()
                if text and len(text) > 2 and len(text) < 200:
                    ingredients.append(text)
        
        # Strategy 2: Look for ingredient-like patterns in all lists
        if not ingredients:
            all_lists = soup.select('ul li, ol li')
            for item in all_lists:
                text = item.get_text().strip()
                # Look for measurement patterns typical of ingredients
                if re.search(r'\d+.*(?:cup|tsp|tbsp|tablespoon|teaspoon|gram|ml|oz|clove|leaf|stick|piece)', text, re.IGNORECASE):
                    ingredients.append(text)
                elif re.search(r'^\d+[\s\-\/]*[a-zA-Z]', text) and len(text) < 100:
                    ingredients.append(text)
        
        # Strategy 3: Text-based extraction
        if not ingredients:
            all_text = soup.get_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            in_ingredients_section = False
            for line in lines:
                # Check if we're entering ingredients section
                if re.search(r'\bingredients?\b', line, re.IGNORECASE) and len(line) < 50:
                    in_ingredients_section = True
                    continue
                
                # Check if we're leaving ingredients section
                if in_ingredients_section and re.search(r'\b(?:instructions?|method|steps?|cooking|directions?)\b', line, re.IGNORECASE):
                    break
                
                # Collect ingredients
                if in_ingredients_section and line:
                    if re.search(r'\d+.*(?:cup|tsp|tbsp|gram|ml)', line, re.IGNORECASE) or len(line) < 80:
                        ingredients.append(line)
        
        # Clean and deduplicate
        clean_ingredients = []
        seen = set()
        for ingredient in ingredients:
            clean = re.sub(r'\s+', ' ', ingredient).strip()
            if clean and clean.lower() not in seen and len(clean) > 2:
                clean_ingredients.append(clean)
                seen.add(clean.lower())
        
        return clean_ingredients[:20]  # Limit to 20 ingredients max
    
    def extract_instructions_flexible(self, soup: BeautifulSoup) -> List[str]:
        """
        Flexible instruction extraction using multiple strategies  
        """
        instructions = []
        
        # Strategy 1: Try your provided selector
        steps_section = soup.select_one('.steps')
        if steps_section:
            items = steps_section.select('li, p, div')
            for item in items:
                text = item.get_text().strip()
                text = re.sub(r'^\d+\.?\s*', '', text)  # Remove step numbers
                if text and len(text) > 15:
                    instructions.append(text)
        
        # Strategy 2: Look for ordered lists (common for instructions)
        if not instructions:
            ol_items = soup.select('ol li')
            for item in ol_items:
                text = item.get_text().strip()
                text = re.sub(r'^\d+\.?\s*', '', text)
                if len(text) > 15:
                    instructions.append(text)
        
        # Strategy 3: Look for numbered instructions in text
        if not instructions:
            all_text = soup.get_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            for line in lines:
                # Look for numbered steps with cooking verbs
                if re.match(r'^\d+\.?\s+', line) and len(line) > 25:
                    cooking_verbs = ['add', 'mix', 'cook', 'heat', 'stir', 'wash', 'cover', 'drain', 'serve', 'boil', 'fry']
                    if any(verb in line.lower() for verb in cooking_verbs):
                        clean_instruction = re.sub(r'^\d+\.?\s*', '', line)
                        instructions.append(clean_instruction)
        
        # Clean instructions
        clean_instructions = []
        for instruction in instructions:
            clean = re.sub(r'\s+', ' ', instruction).strip()
            if clean and len(clean) > 10:
                clean_instructions.append(clean)
        
        return clean_instructions[:15]  # Limit to 15 instructions max
    
    def extract_serving_size_flexible(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Flexible serving size extraction
        """
        # Strategy 1: Try your provided pattern
        serving_elements = soup.select('[class*="serving_recipe_"], [id*="serving_recipe_"]')
        for elem in serving_elements:
            text = elem.get_text().strip()
            numbers = re.findall(r'\d+(?:-\d+)?', text)
            if numbers:
                return numbers[0]
        
        # Strategy 2: Common serving patterns in text
        all_text = soup.get_text()
        serving_patterns = [
            r'(\d+(?:-\d+)?)\s*people',
            r'(\d+(?:-\d+)?)\s*servings?',
            r'(\d+(?:-\d+)?)\s*portions?',
            r'serves?\s*(\d+(?:-\d+)?)',
            r'for\s*(\d+(?:-\d+)?)\s*people',
            r'makes?\s*(\d+(?:-\d+)?)\s*servings?'
        ]
        
        for pattern in serving_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Strategy 3: Look in common serving elements
        serving_selectors = ['.servings', '.recipe-yield', '.serves', '[class*="yield"]', '[class*="serving"]']
        for selector in serving_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text().strip()
                numbers = re.findall(r'\d+', text)
                if numbers and 1 <= int(numbers[0]) <= 20:  # Reasonable serving range
                    return numbers[0]
        
        return None
    
    def scrape_recipe(self, recipe_url: str) -> Optional[Dict]:
        """
        Scrape individual recipe with flexible extraction
        """
        try:
            logger.debug(f"Scraping: {recipe_url}")
            
            response = self.session.get(recipe_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            recipe_data = {}
            
            # Extract title
            title_elem = soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
                title = re.sub(r'\s*\|\s*Cookpad.*$', '', title)
                recipe_data['title'] = title
            
            if not recipe_data.get('title'):
                logger.warning(f"No title found for {recipe_url}")
                return None
            
            # Extract data using flexible methods
            recipe_data['ingredients'] = self.extract_ingredients_flexible(soup)
            recipe_data['instructions'] = self.extract_instructions_flexible(soup)
            recipe_data['servings'] = self.extract_serving_size_flexible(soup)
            
            # Extract cook time
            all_text = soup.get_text().lower()
            time_patterns = [r'(\d+)\s*mins?', r'(\d+)\s*minutes?', r'(\d+)\s*hours?']
            
            for pattern in time_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    time_value = matches[0]
                    if 'hour' in all_text[all_text.find(time_value):all_text.find(time_value)+20]:
                        recipe_data['cook_time'] = f"{time_value} hours"
                    else:
                        recipe_data['cook_time'] = f"{time_value} mins"
                    break
            
            # Add metadata
            recipe_data['url'] = recipe_url
            recipe_data['scraped_at'] = datetime.now().isoformat()
            
            # Validate essential data
            has_ingredients = len(recipe_data.get('ingredients', [])) > 0
            has_instructions = len(recipe_data.get('instructions', [])) > 0
            
            if recipe_data.get('title') and (has_ingredients or has_instructions):
                self.scraped_urls.add(recipe_url)
                
                # Log what we found for debugging
                logger.debug(f"✓ {recipe_data['title']}")
                logger.debug(f"  Ingredients: {len(recipe_data.get('ingredients', []))}")
                logger.debug(f"  Instructions: {len(recipe_data.get('instructions', []))}")
                logger.debug(f"  Servings: {recipe_data.get('servings', 'None')}")
                
                return recipe_data
            else:
                logger.warning(f"Insufficient data for {recipe_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {recipe_url}: {e}")
            self.failed_urls.append((recipe_url, str(e)))
            return None
    
    def scrape_recipes(self) -> List[Dict]:
        """
        Main scraping method with targeted approach
        """
        logger.info(f"🎯 Starting targeted scrape for {self.max_recipes} authentic Sri Lankan recipes")
        
        # Get targeted recipe URLs
        recipe_urls = self.get_targeted_recipe_urls()
        
        if not recipe_urls:
            logger.error("No recipe URLs found")
            return []
        
        logger.info(f"Found {len(recipe_urls)} recipe URLs from authentic dish searches")
        
        # Optional: Debug first recipe to understand structure
        if recipe_urls:
            debug_choice = input(f"\nDebug first recipe structure? (y/n): ").lower()
            if debug_choice == 'y':
                self.debug_recipe_structure(recipe_urls[0])
                continue_choice = input("Continue with scraping? (y/n): ").lower()
                if continue_choice != 'y':
                    return []
        
        # Scrape recipes
        for i, url in enumerate(recipe_urls[:self.max_recipes * 2]):  # Check up to 2x target
            if len(self.scraped_recipes) >= self.max_recipes:
                break
            
            try:
                recipe_data = self.scrape_recipe(url)
                
                if recipe_data:
                    self.scraped_recipes.append(recipe_data)
                    logger.info(f"✓ Recipe {len(self.scraped_recipes)}/{self.max_recipes}: {recipe_data['title']}")
                
                # Progress update
                if (i + 1) % 25 == 0:
                    success_rate = (len(self.scraped_recipes) / (i + 1)) * 100
                    logger.info(f"Progress: {i+1} URLs checked, {len(self.scraped_recipes)} recipes scraped ({success_rate:.1f}% success)")
                
                time.sleep(random.uniform(*self.delay_range))
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing URL {i+1}: {e}")
                continue
        
        logger.info(f"🎉 Completed! Scraped {len(self.scraped_recipes)} authentic Sri Lankan recipes")
        return self.scraped_recipes
    
    def save_results(self):
        """Save results to files"""
        # JSON
        json_file = os.path.join(self.output_dir, 'authentic_sri_lankan_recipes.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_recipes, f, indent=2, ensure_ascii=False)
        
        # CSV
        if self.scraped_recipes:
            csv_file = os.path.join(self.output_dir, 'authentic_sri_lankan_recipes.csv')
            
            flattened = []
            for recipe in self.scraped_recipes:
                flat_recipe = {}
                for key, value in recipe.items():
                    if isinstance(value, list):
                        flat_recipe[key] = ' | '.join(value)
                    else:
                        flat_recipe[key] = value
                flattened.append(flat_recipe)
            
            if flattened:
                fieldnames = flattened[0].keys()
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened)
        
        # Summary
        report_file = os.path.join(self.output_dir, 'summary.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🇱🇰 AUTHENTIC SRI LANKAN RECIPE SCRAPER RESULTS\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Recipes Scraped: {len(self.scraped_recipes)}\n")
            f.write(f"Failed URLs: {len(self.failed_urls)}\n\n")
            
            if self.scraped_recipes:
                # Data completeness stats
                with_ingredients = sum(1 for r in self.scraped_recipes if r.get('ingredients'))
                with_instructions = sum(1 for r in self.scraped_recipes if r.get('instructions'))
                with_servings = sum(1 for r in self.scraped_recipes if r.get('servings'))
                with_cook_time = sum(1 for r in self.scraped_recipes if r.get('cook_time'))
                
                f.write("DATA COMPLETENESS:\n")
                f.write(f"- Recipes with ingredients: {with_ingredients} ({with_ingredients/len(self.scraped_recipes)*100:.1f}%)\n")
                f.write(f"- Recipes with instructions: {with_instructions} ({with_instructions/len(self.scraped_recipes)*100:.1f}%)\n")
                f.write(f"- Recipes with servings: {with_servings} ({with_servings/len(self.scraped_recipes)*100:.1f}%)\n")
                f.write(f"- Recipes with cook time: {with_cook_time} ({with_cook_time/len(self.scraped_recipes)*100:.1f}%)\n\n")
                
                f.write("SAMPLE RECIPES:\n")
                for i, recipe in enumerate(self.scraped_recipes[:10], 1):
                    f.write(f"{i}. {recipe['title']}\n")
                    f.write(f"   Ingredients: {len(recipe.get('ingredients', []))}\n")
                    f.write(f"   Instructions: {len(recipe.get('instructions', []))}\n")
                    f.write(f"   Servings: {recipe.get('servings', 'N/A')}\n")
                    f.write(f"   Cook Time: {recipe.get('cook_time', 'N/A')}\n")
                    f.write(f"   URL: {recipe['url']}\n\n")
        
        logger.info(f"📁 Results saved to {self.output_dir}/")

# Main execution
if __name__ == "__main__":
    print("🇱🇰 TARGETED Sri Lankan Recipe Scraper")
    print("=" * 45)
    print("✓ Searches for specific Sri Lankan dishes")
    print("✓ Flexible ingredient/instruction extraction")  
    print("✓ Better authenticity through targeted searches")
    print("✓ Debug mode to understand recipe structure")
    print()
    
    max_recipes = int(input("Number of recipes to scrape (default 100): ") or "100")
    
    print(f"\n🎯 Targeting authentic Sri Lankan dishes like:")
    print("   hoppers, kottu, pol sambol, kaha bath, string hoppers...")
    print("🔧 Includes debug mode to analyze recipe structure")
    
    scraper = TargetedSriLankanScraper(max_recipes=max_recipes)
    
    try:
        recipes = scraper.scrape_recipes()
        
        if recipes:
            scraper.save_results()
            print(f"\n🎉 SUCCESS!")
            print(f"✓ Scraped {len(recipes)} authentic Sri Lankan recipes")
            print(f"❌ {len(scraper.failed_urls)} URLs failed")
            print(f"📁 Results saved in: {scraper.output_dir}/")
            
            # Show sample data
            if recipes:
                sample = recipes[0]
                print(f"\n📝 Sample recipe: {sample['title']}")
                print(f"   Ingredients: {len(sample.get('ingredients', []))} found")
                print(f"   Instructions: {len(sample.get('instructions', []))} found")
                print(f"   Servings: {sample.get('servings', 'Not found')}")
                
                if sample.get('ingredients'):
                    print(f"   First ingredient: {sample['ingredients'][0]}")
                if sample.get('instructions'):
                    print(f"   First instruction: {sample['instructions'][0][:60]}...")
        else:
            print("❌ No recipes found")
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        if scraper.scraped_recipes:
            scraper.save_results()