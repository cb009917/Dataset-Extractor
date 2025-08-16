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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleCookpadScraper:
    def __init__(self, max_recipes=100, delay_range=(2, 4)):
        """
        Simple Cookpad scraper using only requests (no Selenium required)
        Limited to initially loaded recipes but faster setup
        """
        self.max_recipes = max_recipes
        self.delay_range = delay_range
        self.base_url = "https://cookpad.com"
        self.scraped_recipes = []
        self.scraped_urls = set()
        self.failed_urls = []
        self.rejected_recipes = []
        
        # Create output directory
        self.output_dir = f"cookpad_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sri Lankan authenticity filters
        self.sri_lankan_indicators = {
            'positive': [
                'sri lankan', 'sri lanka', 'curry leaves', 'pandan leaf', 'coconut milk', 
                'kaha bath', 'hoppers', 'kottu', 'string hoppers', 'pol sambol',
                'gotukola', 'murunga', 'drumstick', 'goraka', 'tamarind', 'rampe',
                'maldive fish', 'coconut sambol', 'kiribath', 'pittu', 'ambul thiyal'
            ],
            'negative': [
                'masala dosa', 'sambar', 'idli', 'vada', 'uttapam', 'rasam',
                'biryani', 'tandoor', 'naan', 'garam masala', 'paneer', 'chole'
            ]
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_recipe_urls(self) -> List[str]:
        """Get recipe URLs from multiple search pages"""
        all_urls = []
        
        # Try different search terms for better coverage
        search_terms = [
            'sri lankan',
            'kaha bath',
            'hoppers',
            'kottu',
            'pol sambol',
            'string hoppers'
        ]
        
        for search_term in search_terms:
            try:
                search_url = f"https://cookpad.com/eng/search/{search_term.replace(' ', '%20')}"
                logger.info(f"Searching for: {search_term}")
                
                response = self.session.get(search_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find recipe links
                recipe_links = soup.select('a[href*="/recipes/"]')
                
                for link in recipe_links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        if self.is_valid_recipe_url(full_url):
                            all_urls.append(full_url)
                
                # Add delay between searches
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error searching for '{search_term}': {e}")
                continue
        
        # Remove duplicates
        unique_urls = list(dict.fromkeys(all_urls))
        logger.info(f"Found {len(unique_urls)} unique recipe URLs")
        return unique_urls
    
    def is_valid_recipe_url(self, url: str) -> bool:
        """Check if URL is valid"""
        return (
            'cookpad.com' in url and
            '/recipes/' in url and
            url not in self.scraped_urls
        )
    
    def is_sri_lankan_recipe(self, title: str, ingredients: List[str], instructions: List[str]) -> bool:
        """Check if recipe is authentically Sri Lankan"""
        try:
            all_text = (title + " " + " ".join(ingredients) + " " + " ".join(instructions)).lower()
            
            # Check for exclusion terms
            for negative_term in self.sri_lankan_indicators['negative']:
                if negative_term.lower() in all_text:
                    return False
            
            # Check for positive indicators
            score = 0
            for positive_term in self.sri_lankan_indicators['positive']:
                if positive_term.lower() in all_text:
                    score += 2
            
            # Special bonus for "sri lankan" in title
            if 'sri lankan' in title.lower():
                score += 5
            
            return score >= 3
            
        except Exception as e:
            logger.debug(f"Error checking authenticity: {e}")
            return False
    
    def scrape_recipe(self, recipe_url: str) -> Optional[Dict]:
        """Scrape individual recipe"""
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
                return None
            
            # Extract ingredients using multiple strategies
            ingredients = []
            
            # Strategy 1: Look for ingredient lists
            ingredient_lists = soup.select('ul li, ol li')
            potential_ingredients = []
            
            for item in ingredient_lists:
                text = item.get_text().strip()
                # Look for measurement patterns
                if re.search(r'\d+.*(?:cup|tsp|tbsp|gram|ml|clove|leaf)', text, re.IGNORECASE):
                    potential_ingredients.append(text)
                elif re.search(r'^\d+[\s\-\/]*\w+', text):
                    potential_ingredients.append(text)
            
            # Strategy 2: Text parsing for ingredients section
            if not potential_ingredients:
                all_text = soup.get_text()
                lines = all_text.split('\n')
                
                in_ingredients = False
                for line in lines:
                    line = line.strip()
                    if 'ingredients' in line.lower():
                        in_ingredients = True
                        continue
                    if in_ingredients and ('cooking' in line.lower() or 'instructions' in line.lower() or 'method' in line.lower()):
                        break
                    if in_ingredients and line:
                        if re.search(r'\d+.*(?:cup|tsp|tbsp|gram|ml)', line, re.IGNORECASE):
                            potential_ingredients.append(line)
            
            recipe_data['ingredients'] = potential_ingredients[:15]  # Limit to 15 ingredients
            
            # Extract instructions
            instructions = []
            
            # Strategy 1: Look for ordered lists (most common for instructions)
            ol_items = soup.select('ol li')
            if ol_items:
                for item in ol_items:
                    instruction = item.get_text().strip()
                    instruction = re.sub(r'^\d+\.?\s*', '', instruction)  # Remove numbers
                    if len(instruction) > 20:  # Must be substantial
                        instructions.append(instruction)
            
            # Strategy 2: Look for numbered instructions in text
            if not instructions:
                all_text = soup.get_text()
                lines = all_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for numbered steps
                    if re.match(r'^\d+\.?\s+', line) and len(line) > 30:
                        cooking_verbs = ['add', 'mix', 'cook', 'heat', 'stir', 'wash', 'cover']
                        if any(verb in line.lower() for verb in cooking_verbs):
                            clean_instruction = re.sub(r'^\d+\.?\s*', '', line)
                            instructions.append(clean_instruction)
            
            recipe_data['instructions'] = instructions
            
            # Extract cook time
            all_text = soup.get_text().lower()
            time_match = re.search(r'(\d+)\s*(?:mins?|minutes?)', all_text)
            if time_match:
                recipe_data['cook_time'] = f"{time_match.group(1)} mins"
            
            # Extract servings
            serving_match = re.search(r'(\d+(?:-\d+)?)\s*(?:people|servings?)', all_text)
            if serving_match:
                recipe_data['servings'] = serving_match.group(1)
            
            # Add metadata
            recipe_data['url'] = recipe_url
            recipe_data['scraped_at'] = datetime.now().isoformat()
            
            # Check authenticity
            if not self.is_sri_lankan_recipe(
                recipe_data.get('title', ''),
                recipe_data.get('ingredients', []),
                recipe_data.get('instructions', [])
            ):
                self.rejected_recipes.append({
                    'title': recipe_data.get('title'),
                    'url': recipe_url,
                    'reason': 'Not authentically Sri Lankan'
                })
                return None
            
            # Validate we have essential data
            if recipe_data.get('title') and (recipe_data.get('ingredients') or recipe_data.get('instructions')):
                self.scraped_urls.add(recipe_url)
                return recipe_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {recipe_url}: {e}")
            self.failed_urls.append((recipe_url, str(e)))
            return None
    
    def scrape_recipes(self) -> List[Dict]:
        """Main scraping method"""
        logger.info(f"Starting simple scrape for {self.max_recipes} authentic Sri Lankan recipes")
        
        # Get recipe URLs
        recipe_urls = self.get_recipe_urls()
        
        if not recipe_urls:
            logger.error("No recipe URLs found")
            return []
        
        logger.info(f"Found {len(recipe_urls)} recipe URLs to check")
        
        # Scrape recipes
        authentic_count = 0
        for i, url in enumerate(recipe_urls):
            if authentic_count >= self.max_recipes:
                break
            
            try:
                recipe_data = self.scrape_recipe(url)
                
                if recipe_data:
                    self.scraped_recipes.append(recipe_data)
                    authentic_count += 1
                    logger.info(f"✓ Recipe {authentic_count}/{self.max_recipes}: {recipe_data['title']}")
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1} URLs checked, {authentic_count} authentic recipes found")
                
                # Delay between requests
                time.sleep(random.uniform(*self.delay_range))
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing URL {i+1}: {e}")
                continue
        
        logger.info(f"Completed! Found {len(self.scraped_recipes)} authentic Sri Lankan recipes")
        return self.scraped_recipes
    
    def save_results(self):
        """Save all results to files"""
        # JSON
        json_file = os.path.join(self.output_dir, 'sri_lankan_recipes.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_recipes, f, indent=2, ensure_ascii=False)
        
        # CSV
        if self.scraped_recipes:
            csv_file = os.path.join(self.output_dir, 'sri_lankan_recipes.csv')
            
            # Flatten data for CSV
            flattened = []
            for recipe in self.scraped_recipes:
                flat_recipe = {}
                for key, value in recipe.items():
                    if isinstance(value, list):
                        flat_recipe[key] = ' | '.join(value)
                    else:
                        flat_recipe[key] = value
                flattened.append(flat_recipe)
            
            # Write CSV
            fieldnames = flattened[0].keys() if flattened else []
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened)
        
        # Summary report
        report_file = os.path.join(self.output_dir, 'summary.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SIMPLE SRI LANKAN RECIPE SCRAPER RESULTS\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Authentic recipes found: {len(self.scraped_recipes)}\n")
            f.write(f"Rejected recipes: {len(self.rejected_recipes)}\n")
            f.write(f"Failed URLs: {len(self.failed_urls)}\n\n")
            
            if self.scraped_recipes:
                f.write("SAMPLE RECIPES:\n")
                for i, recipe in enumerate(self.scraped_recipes[:5], 1):
                    f.write(f"{i}. {recipe['title']}\n")
                    f.write(f"   Ingredients: {len(recipe.get('ingredients', []))}\n")
                    f.write(f"   Instructions: {len(recipe.get('instructions', []))}\n")
                    f.write(f"   URL: {recipe['url']}\n\n")
        
        logger.info(f"Results saved to {self.output_dir}/")

# Main execution
if __name__ == "__main__":
    print("🇱🇰 Simple Sri Lankan Recipe Scraper")
    print("=" * 40)
    print("✓ No Selenium required - uses only requests")
    print("✓ Filters out non-Sri Lankan recipes")
    print("✓ Limited to initially loaded recipes (faster setup)")
    print()
    
    max_recipes = int(input("Number of recipes to scrape (default 50): ") or "50")
    
    scraper = SimpleCookpadScraper(max_recipes=max_recipes)
    
    try:
        recipes = scraper.scrape_recipes()
        
        if recipes:
            scraper.save_results()
            print(f"\n🎉 Success!")
            print(f"✓ {len(recipes)} authentic Sri Lankan recipes found")
            print(f"✗ {len(scraper.rejected_recipes)} non-Sri Lankan recipes filtered out")
            print(f"📁 Results saved in: {scraper.output_dir}/")
            
            # Show sample
            if recipes:
                print(f"\n📝 Sample recipe: {recipes[0]['title']}")
                print(f"   Ingredients: {len(recipes[0].get('ingredients', []))}")
                print(f"   Instructions: {len(recipes[0].get('instructions', []))}")
        else:
            print("❌ No recipes found")
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        if scraper.scraped_recipes:
            scraper.save_results()
            print(f"💾 Saved {len(scraper.scraped_recipes)} recipes found so far")