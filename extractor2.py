import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HungryLankanScraper:
    def __init__(self):
        self.base_url = "https://www.hungrylankan.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.recipes = []
        self.recipe_urls = set()
        
    def get_page(self, url, retries=3):
        """Get page content with error handling and retries"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None
    
    def discover_recipe_urls(self):
        """Discover all recipe URLs from various pages"""
        logger.info("Starting recipe URL discovery...")
        
        # URLs to check for recipe links
        discovery_urls = [
            f"{self.base_url}/recipes/",
            f"{self.base_url}/recipe-cuisines/",
            f"{self.base_url}/"
        ]
        
        # Get recipe URLs from main pages
        for url in discovery_urls:
            self._extract_recipe_urls_from_page(url)
        
        # Get recipe URLs from pagination
        self._discover_paginated_recipes()
        
        logger.info(f"Found {len(self.recipe_urls)} unique recipe URLs")
        return list(self.recipe_urls)
    
    def _extract_recipe_urls_from_page(self, url):
        """Extract recipe URLs from a single page"""
        response = self.get_page(url)
        if not response:
            return
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for recipe links in various selectors
        selectors = [
            'a[href*="/recipes/"]',
            '.recipe-card a',
            '.entry-title a',
            'h2 a[href*="/recipes/"]',
            'h3 a[href*="/recipes/"]',
            '.post-title a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and '/recipes/' in href:
                    full_url = urljoin(self.base_url, href)
                    self.recipe_urls.add(full_url)
    
    def _discover_paginated_recipes(self):
        """Discover recipes from paginated pages"""
        page = 1
        while page <= 20:  # Limit to prevent infinite loops
            url = f"{self.base_url}/recipes/page/{page}/"
            response = self.get_page(url)
            
            if not response or response.status_code == 404:
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            initial_count = len(self.recipe_urls)
            self._extract_recipe_urls_from_page(url)
            
            # If no new URLs found, we've reached the end
            if len(self.recipe_urls) == initial_count:
                break
                
            page += 1
            time.sleep(1)  # Be respectful
        
        logger.info(f"Checked {page - 1} paginated pages")
    
    def extract_recipe_data(self, recipe_url):
        """Extract detailed recipe data from a recipe page"""
        response = self.get_page(recipe_url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        recipe_data = {
            'url': recipe_url,
            'title': '',
            'description': '',
            'ingredients': [],
            'instructions': [],
            'prep_time': '',
            'cook_time': '',
            'total_time': '',
            'servings': '',
            'difficulty': '',
            'cuisine': '',
            'course': '',
            'dietary_tags': [],
            'keywords': [],
            'author': '',
            'cooking_method': '',
            'image_url': ''
        }
        
        try:
            # Extract title
            title_selectors = ['h1.entry-title', 'h1', '.recipe-title', '.entry-header h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    recipe_data['title'] = title_elem.get_text().strip()
                    break
            
            # Extract description
            desc_selectors = ['.recipe-summary', '.entry-content p:first-of-type', '.recipe-description']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    recipe_data['description'] = desc_elem.get_text().strip()
                    break
            
            # Extract structured data (JSON-LD)
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    data = json.loads(json_ld.string)
                    if isinstance(data, list):
                        data = data[0]
                    
                    if data.get('@type') == 'Recipe':
                        self._extract_from_json_ld(recipe_data, data)
                except json.JSONDecodeError:
                    pass
            
            # Extract ingredients
            ingredient_selectors = [
                '.recipe-ingredients li',
                '.ingredients li',
                '.wp-block-recipe-card-ingredients li',
                '[class*="ingredient"] li'
            ]
            
            for selector in ingredient_selectors:
                ingredients = soup.select(selector)
                if ingredients:
                    recipe_data['ingredients'] = [ing.get_text().strip() for ing in ingredients]
                    break
            
            # Extract instructions
            instruction_selectors = [
                '.recipe-instructions li',
                '.instructions li',
                '.wp-block-recipe-card-instructions li',
                '[class*="instruction"] li',
                '.recipe-instructions p',
                '.instructions p'
            ]
            
            for selector in instruction_selectors:
                instructions = soup.select(selector)
                if instructions:
                    recipe_data['instructions'] = [inst.get_text().strip() for inst in instructions]
                    break
            
            # Extract recipe metadata
            self._extract_recipe_metadata(soup, recipe_data)
            
            # Extract image
            img_selectors = ['.recipe-image img', '.entry-content img:first-of-type', '.wp-post-image']
            for selector in img_selectors:
                img = soup.select_one(selector)
                if img and img.get('src'):
                    recipe_data['image_url'] = urljoin(recipe_url, img['src'])
                    break
            
        except Exception as e:
            logger.error(f"Error extracting recipe data from {recipe_url}: {e}")
            return None
        
        return recipe_data
    
    def _extract_from_json_ld(self, recipe_data, json_data):
        """Extract data from JSON-LD structured data"""
        recipe_data['title'] = json_data.get('name', recipe_data['title'])
        recipe_data['description'] = json_data.get('description', recipe_data['description'])
        
        # Extract ingredients
        if 'recipeIngredient' in json_data:
            recipe_data['ingredients'] = json_data['recipeIngredient']
        
        # Extract instructions
        if 'recipeInstructions' in json_data:
            instructions = []
            for inst in json_data['recipeInstructions']:
                if isinstance(inst, dict):
                    instructions.append(inst.get('text', ''))
                else:
                    instructions.append(str(inst))
            recipe_data['instructions'] = instructions
        
        # Extract times
        recipe_data['prep_time'] = json_data.get('prepTime', '')
        recipe_data['cook_time'] = json_data.get('cookTime', '')
        recipe_data['total_time'] = json_data.get('totalTime', '')
        
        # Extract servings
        if 'recipeYield' in json_data:
            recipe_data['servings'] = str(json_data['recipeYield'])
        
        # Extract author
        if 'author' in json_data:
            author = json_data['author']
            if isinstance(author, dict):
                recipe_data['author'] = author.get('name', '')
            else:
                recipe_data['author'] = str(author)
        
        # Extract cuisine and category
        recipe_data['cuisine'] = json_data.get('recipeCuisine', '')
        if 'recipeCategory' in json_data:
            recipe_data['course'] = json_data['recipeCategory']
    
    def _extract_recipe_metadata(self, soup, recipe_data):
        """Extract recipe metadata from various elements"""
        # Look for metadata in recipe cards or structured elements
        metadata_map = {
            'prep_time': ['prep time', 'preparation time', 'prep-time'],
            'cook_time': ['cook time', 'cooking time', 'cook-time'],
            'total_time': ['total time', 'total-time'],
            'servings': ['servings', 'serves', 'yield'],
            'difficulty': ['difficulty', 'level'],
            'cuisine': ['cuisine'],
            'course': ['course', 'meal type'],
            'cooking_method': ['cooking method', 'method']
        }
        
        # Extract from meta elements or structured data
        for field, keywords in metadata_map.items():
            for keyword in keywords:
                # Look for elements containing the keyword
                elements = soup.find_all(string=re.compile(keyword, re.I))
                for element in elements:
                    parent = element.parent
                    if parent:
                        # Try to find the value in next sibling or nearby elements
                        value_elem = parent.find_next_sibling() or parent.find_next()
                        if value_elem:
                            value = value_elem.get_text().strip()
                            if value and not recipe_data[field]:
                                recipe_data[field] = value
                                break
        
        # Extract dietary tags and keywords from content
        content_text = soup.get_text().lower()
        dietary_keywords = ['vegan', 'vegetarian', 'gluten-free', 'dairy-free', 'nut-free', 'low-fodmap']
        recipe_data['dietary_tags'] = [tag for tag in dietary_keywords if tag in content_text]
    
    def scrape_all_recipes(self):
        """Main method to scrape all recipes"""
        logger.info("Starting Hungry Lankan recipe scraping...")
        
        # Discover all recipe URLs
        recipe_urls = self.discover_recipe_urls()
        
        if not recipe_urls:
            logger.error("No recipe URLs found!")
            return []
        
        # Extract recipe data
        logger.info(f"Extracting data from {len(recipe_urls)} recipes...")
        
        for i, url in enumerate(recipe_urls, 1):
            logger.info(f"Processing recipe {i}/{len(recipe_urls)}: {url}")
            
            recipe_data = self.extract_recipe_data(url)
            if recipe_data:
                self.recipes.append(recipe_data)
                logger.info(f"Successfully extracted: {recipe_data['title']}")
            else:
                logger.warning(f"Failed to extract recipe from: {url}")
            
            # Be respectful - add delay between requests
            time.sleep(1)
        
        logger.info(f"Successfully scraped {len(self.recipes)} recipes")
        return self.recipes
    
    def save_to_json(self, filename='hungry_lankan_recipes.json'):
        """Save recipes to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.recipes, f, indent=2, ensure_ascii=False)
        logger.info(f"Recipes saved to {filename}")
    
    def save_to_csv(self, filename='hungry_lankan_recipes.csv'):
        """Save recipes to CSV file"""
        if not self.recipes:
            logger.warning("No recipes to save")
            return
        
        fieldnames = list(self.recipes[0].keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for recipe in self.recipes:
                # Convert lists to strings for CSV
                row = recipe.copy()
                for key, value in row.items():
                    if isinstance(value, list):
                        row[key] = ' | '.join(value)
                writer.writerow(row)
        
        logger.info(f"Recipes saved to {filename}")

def main():
    """Main function to run the scraper"""
    scraper = HungryLankanScraper()
    
    try:
        # Scrape all recipes
        recipes = scraper.scrape_all_recipes()
        
        if recipes:
            # Save to both JSON and CSV
            scraper.save_to_json()
            scraper.save_to_csv()
            
            # Print summary
            print(f"\n{'='*50}")
            print(f"SCRAPING COMPLETE!")
            print(f"{'='*50}")
            print(f"Total recipes scraped: {len(recipes)}")
            print(f"Files saved:")
            print(f"  - hungry_lankan_recipes.json")
            print(f"  - hungry_lankan_recipes.csv")
            print(f"{'='*50}")
            
            # Show sample recipes
            if recipes:
                print(f"\nSample recipes found:")
                for i, recipe in enumerate(recipes[:5], 1):
                    print(f"{i}. {recipe['title']}")
        else:
            print("No recipes were successfully scraped.")
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()