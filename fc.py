import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import json
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global thread-safe counters
scraping_stats = {
    'total_processed': 0,
    'successful': 0,
    'failed': 0,
    'lock': Lock()
}

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return BeautifulSoup(res.text, "html.parser")
    except requests.RequestException as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None

def extract_recipe_urls(soup, base_url):
    recipe_urls = []
    # Find article links - look for actual recipe posts
    articles = soup.find_all("article") or soup.find_all("div", class_=re.compile("post|entry"))
    
    for article in articles:
        # Look for title links within articles
        title_link = article.find("h2") or article.find("h1") or article.find("a")
        if title_link:
            link = title_link.find("a") if title_link.name != "a" else title_link
            if link and link.get("href"):
                href = link.get("href")
                # Filter for actual recipe URLs (exclude category pages, etc.)
                if (href and 
                    not any(exclude in href.lower() for exclude in ['/category/', '/tag/', '/page/', '/author/', '.jpg', '.png']) and
                    'foodcnr.com' in href):
                    full_url = urljoin(base_url, href)
                    if full_url not in recipe_urls:
                        recipe_urls.append(full_url)
    
    # Also look for direct recipe links
    all_links = soup.find_all("a", href=True)
    for link in all_links:
        href = link.get("href", "")
        # Look for recipe-specific URLs
        if (href.startswith('https://foodcnr.com/') and 
            not any(exclude in href.lower() for exclude in ['/category/', '/tag/', '/page/', '/author/', 'wp-content', '.jpg', '.png', '.css', '.js']) and
            len(href.split('/')[-1]) > 5):  # Has meaningful slug
            if href not in recipe_urls:
                recipe_urls.append(href)
    
    return recipe_urls

def extract_recipe_details(url):
    soup = get_soup(url)
    if not soup:
        return None
    
    recipe = {
        'url': url,
        'title': '',
        'description': '',
        'cooking_time': '',
        'prep_time': '',
        'total_time': '',
        'difficulty': '',
        'servings': '',
        'ingredients': '',
        'instructions': '',
        'tags': '',
        'image_url': '',
        'author': '',
        'published_date': ''
    }
    
    # Method 1: Try to extract from JSON-LD structured data first
    json_scripts = soup.find_all("script", type="application/ld+json")
    for script in json_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                data = data[0]
            
            # Check for @graph structure first
            recipes_to_check = []
            if data.get("@graph"):
                # Look for Recipe items in @graph
                for item in data["@graph"]:
                    if isinstance(item, dict) and item.get("@type") == "Recipe":
                        recipes_to_check.append(item)
            elif data.get("@type") == "Recipe":
                recipes_to_check.append(data)
            
            # Process each recipe found
            for recipe_data in recipes_to_check:
                # Extract title
                if recipe_data.get("name"):
                    recipe['title'] = recipe_data["name"]
                
                # Extract description
                if recipe_data.get("description"):
                    recipe['description'] = recipe_data["description"]
                
                # Extract author
                author = recipe_data.get("author")
                if author:
                    if isinstance(author, dict):
                        recipe['author'] = author.get("name", "")
                    elif isinstance(author, str):
                        recipe['author'] = author
                
                # Extract published date
                if recipe_data.get("datePublished"):
                    recipe['published_date'] = recipe_data["datePublished"]
                
                # Extract servings/yield
                if recipe_data.get("recipeYield"):
                    yield_val = recipe_data["recipeYield"]
                    if isinstance(yield_val, list):
                        recipe['servings'] = str(yield_val[0])
                    else:
                        recipe['servings'] = str(yield_val)
                
                # Extract cooking times
                if recipe_data.get("prepTime"):
                    recipe['prep_time'] = recipe_data["prepTime"]
                if recipe_data.get("cookTime"):
                    recipe['cooking_time'] = recipe_data["cookTime"]
                if recipe_data.get("totalTime"):
                    recipe['total_time'] = recipe_data["totalTime"]
                
                # Extract ingredients
                ingredients = recipe_data.get("recipeIngredient", [])
                if ingredients:
                    recipe['ingredients'] = "; ".join(ingredients)
                
                # Extract instructions - handle HowToSection structure
                instructions_data = recipe_data.get("recipeInstructions", [])
                if instructions_data:
                    instructions = []
                    for instruction in instructions_data:
                        if isinstance(instruction, dict):
                            if instruction.get("@type") == "HowToSection":
                                # Handle sections with itemListElement
                                section_name = instruction.get("name", "")
                                if "itemListElement" in instruction:
                                    section_steps = []
                                    for step in instruction["itemListElement"]:
                                        if isinstance(step, dict) and step.get("@type") == "HowToStep":
                                            text = step.get("text", "")
                                            if text:
                                                section_steps.append(text)
                                    if section_steps:
                                        if section_name:
                                            instructions.append(f"{section_name}: {' '.join(section_steps)}")
                                        else:
                                            instructions.extend(section_steps)
                            elif instruction.get("@type") == "HowToStep":
                                text = instruction.get("text", "")
                                if text:
                                    instructions.append(text)
                            elif "text" in instruction:
                                instructions.append(instruction["text"])
                        elif isinstance(instruction, str):
                            instructions.append(instruction)
                    if instructions:
                        recipe['instructions'] = "; ".join(instructions)
                
                # Extract image
                image = recipe_data.get("image")
                if image:
                    if isinstance(image, list):
                        recipe['image_url'] = image[0] if image else ""
                    elif isinstance(image, dict):
                        recipe['image_url'] = image.get("url", "")
                    else:
                        recipe['image_url'] = str(image)
                
                # If we found structured data, we can return early
                if recipe['title'] and recipe['ingredients'] and recipe['instructions']:
                    return recipe
        
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    # Method 2: HTML-based extraction as fallback
    
    # Extract title
    if not recipe['title']:
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            recipe['title'] = title_elem.get_text(strip=True)
            # Clean up title
            if " - " in recipe['title']:
                recipe['title'] = recipe['title'].split(" - ")[0].strip()
    
    # Extract meta description
    if not recipe['description']:
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            recipe['description'] = meta_desc.get("content", "")
    
    # Extract author
    if not recipe['author']:
        author_elem = (soup.find("span", class_=re.compile("author", re.I)) or 
                       soup.find("a", rel="author") or
                       soup.find(class_=re.compile("byline", re.I)))
        if author_elem:
            recipe['author'] = author_elem.get_text(strip=True)
    
    # Extract published date
    if not recipe['published_date']:
        date_elem = (soup.find("time") or 
                    soup.find(class_=re.compile("date|published", re.I)) or
                    soup.find("meta", {"property": "article:published_time"}))
        if date_elem:
            if date_elem.name == "meta":
                recipe['published_date'] = date_elem.get("content", "")
            else:
                recipe['published_date'] = date_elem.get_text(strip=True)
    
    # Extract servings from HTML if not found in JSON-LD
    if not recipe['servings']:
        full_text = soup.get_text()
        
        # Multiple patterns to find serving information
        patterns = [
            r'ingredients?\s*\([^)]*(?:for\s*)?(\d+(?:-\d+)?)\s*(?:servings?|people|persons?)\s*\)',
            r'\(makes?\s+(?:about\s+)?(\d+(?:-\d+)?)',
            r'(?:recipe|dish)\s+(?:serves?|yields?|makes?)\s+(\d+(?:-\d+)?)',
            r'serves?\s*:?\s*(\d+(?:-\d+)?)\s*(?:people|persons?)?',
            r'for\s+(\d+(?:-\d+)?)\s+(?:people|persons?)',
            r'(?:yields?|makes?)\s+(\d+(?:-\d+)?)',
            r'servings?\s*:?\s*(\d+(?:-\d+)?)',
            r'portions?\s*:?\s*(\d+(?:-\d+)?)',
            r'recipe\s+for\s+(\d+(?:-\d+)?)',
            r'feeds?\s+(\d+(?:-\d+)?)',
            r'(\d+(?:-\d+)?)\s*servings?',
            r'(\d+(?:-\d+)?)\s*portions?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.I)
            if match:
                recipe['servings'] = match.group(1)
                break
    
    # Extract ingredients from HTML if not found in JSON-LD
    if not recipe['ingredients']:
        # Method 1: Look for ingredient lists in content
        content_div = soup.find("div", class_=re.compile("content|entry|post-content", re.I))
        if content_div:
            ingredient_lists = content_div.find_all("ul")
            for ul in ingredient_lists:
                ingredients = []
                for li in ul.find_all("li"):
                    ingredient = li.get_text(strip=True)
                    if ingredient and len(ingredient) > 2:
                        # Check if this looks like an ingredient
                        if any(word in ingredient.lower() for word in ['tsp', 'tbsp', 'cup', 'gram', 'kg', 'ml', 'liter', 'piece', 'clove', 'inch', 'no', 'nos', 'slice', 'chop']):
                            ingredients.append(ingredient)
                if ingredients and len(ingredients) >= 2:  # Lower threshold
                    recipe['ingredients'] = "; ".join(ingredients)
                    break
        
        # Method 2: Look for ingredients after "Ingredients" heading
        if not recipe['ingredients']:
            ingredients_heading = soup.find(string=re.compile(r'ingredients?', re.I))
            if ingredients_heading:
                parent = ingredients_heading.parent
                while parent and parent.name not in ['div', 'article', 'section']:
                    parent = parent.parent
                
                if parent:
                    # Find the next list or paragraphs after the ingredients heading
                    next_ul = parent.find("ul")
                    if next_ul:
                        ingredients = []
                        for li in next_ul.find_all("li"):
                            ingredient = li.get_text(strip=True)
                            if ingredient and len(ingredient) > 2:
                                ingredients.append(ingredient)
                        if ingredients:
                            recipe['ingredients'] = "; ".join(ingredients)
        
        # Method 3: Parse from text content with patterns
        if not recipe['ingredients']:
            full_text = soup.get_text()
            # Look for ingredient patterns in text
            ingredient_section = re.search(r'ingredients?:?\s*(.*?)(?:method|instructions|directions|preparation)', full_text, re.I | re.DOTALL)
            if ingredient_section:
                ingredient_text = ingredient_section.group(1).strip()
                # Split by line breaks and filter
                potential_ingredients = [line.strip() for line in ingredient_text.split('\n') if line.strip()]
                ingredients = []
                for line in potential_ingredients:
                    if any(word in line.lower() for word in ['cup', 'tsp', 'tbsp', 'gram', 'kg', 'ml', 'piece', 'no', 'nos']):
                        ingredients.append(line)
                if ingredients:
                    recipe['ingredients'] = "; ".join(ingredients)
    
    # Extract instructions from HTML if not found in JSON-LD
    if not recipe['instructions']:
        content_div = soup.find("div", class_=re.compile("content|entry|post-content", re.I))
        if content_div:
            # Method 1: Look for ordered lists
            instruction_ol = content_div.find("ol")
            if instruction_ol:
                instructions = []
                for li in instruction_ol.find_all("li"):
                    instruction = li.get_text(strip=True)
                    if instruction and len(instruction) > 8:  # Lower threshold
                        instructions.append(instruction)
                if instructions:
                    recipe['instructions'] = "; ".join(instructions)
            
            # Method 2: Look for step-by-step paragraphs
            if not recipe['instructions']:
                paragraphs = content_div.find_all("p")
                instructions = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if (re.match(r'^(?:step\s*)?\d+[\.\):\-]\s+', text, re.I) or 
                        text.lower().startswith('step') or
                        re.match(r'^\d+[\.\)]\s+', text)):
                        instructions.append(text)
                if instructions:
                    recipe['instructions'] = "; ".join(instructions)
            
            # Method 3: Look for instructions after method/directions heading
            if not recipe['instructions']:
                method_keywords = ['method', 'instructions', 'directions', 'preparation', 'cooking']
                for keyword in method_keywords:
                    method_heading = content_div.find(string=re.compile(keyword, re.I))
                    if method_heading:
                        parent = method_heading.parent
                        instructions = []
                        
                        # Get all following paragraphs and list items
                        current = parent
                        while current:
                            for sibling in current.find_next_siblings():
                                if sibling.name in ['p', 'div']:
                                    text = sibling.get_text(strip=True)
                                    if len(text) > 15:  # Substantial text
                                        instructions.append(text)
                                elif sibling.name in ['ol', 'ul']:
                                    for li in sibling.find_all("li"):
                                        text = li.get_text(strip=True)
                                        if len(text) > 8:
                                            instructions.append(text)
                            if len(instructions) >= 2:
                                break
                            current = current.parent
                            if not current or current.name in ['body', 'html']:
                                break
                        
                        if instructions:
                            recipe['instructions'] = "; ".join(instructions)
                            break
        
        # Method 4: Parse from full text as last resort
        if not recipe['instructions']:
            full_text = soup.get_text()
            # Look for method/instructions section in text
            method_section = re.search(r'(?:method|instructions|directions):?\s*(.*?)(?:notes?|tips?|$)', full_text, re.I | re.DOTALL)
            if method_section:
                method_text = method_section.group(1).strip()
                # Split into potential steps
                potential_steps = [step.strip() for step in re.split(r'\n+|(?=step\s*\d+)', method_text, flags=re.I) if step.strip()]
                instructions = []
                for step in potential_steps:
                    if len(step) > 20 and any(word in step.lower() for word in ['add', 'cook', 'heat', 'mix', 'stir', 'boil', 'fry']):
                        instructions.append(step)
                if instructions:
                    recipe['instructions'] = "; ".join(instructions)
    
    # Extract tags/categories
    tag_elements = soup.find_all("a", href=re.compile(r'/category/|/tag/'))
    tags = []
    for tag_elem in tag_elements:
        tag = tag_elem.get_text(strip=True)
        if tag and tag not in tags and tag.lower() != 'sri lankan':
            tags.append(tag)
    recipe['tags'] = ", ".join(tags[:5])
    
    # Extract main image if not found in JSON-LD
    if not recipe['image_url']:
        img_elem = (soup.find("img", class_=re.compile("featured|hero|main")) or
                    soup.find("meta", {"property": "og:image"}) or
                    soup.find("img", src=True))
        if img_elem:
            if img_elem.name == "meta":
                recipe['image_url'] = img_elem.get("content", "")
            else:
                img_src = img_elem.get("src", "")
                recipe['image_url'] = urljoin(url, img_src)
    
    return recipe

def get_total_pages(base_url):
    """Get the total number of pages from pagination"""
    soup = get_soup(base_url)
    if not soup:
        return 1
    
    # Look for pagination
    pagination = soup.find("nav", class_=re.compile("pag", re.I)) or soup.find(class_=re.compile("pag", re.I))
    if pagination:
        page_links = pagination.find_all("a")
        page_numbers = []
        for link in page_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                page_numbers.append(int(text))
        if page_numbers:
            return max(page_numbers)
    
    # Fallback: look for "page X of Y" text
    page_text = soup.get_text()
    page_match = re.search(r'page\s+\d+\s+of\s+(\d+)', page_text, re.I)
    if page_match:
        return int(page_match.group(1))
    
    return 34  # Based on previous observations

def extract_recipe_details_threaded(recipe_url, page_num, recipe_index, total_recipes):
    """
    Thread worker function for extracting recipe details
    """
    try:
        recipe_details = extract_recipe_details(recipe_url)
        
        # Thread-safe progress tracking
        with scraping_stats['lock']:
            scraping_stats['total_processed'] += 1
            
            if recipe_details and recipe_details['title']:
                scraping_stats['successful'] += 1
                # Use ASCII-safe output
                title = recipe_details['title'].encode('ascii', 'ignore').decode('ascii')
                ingredients_count = len(recipe_details.get('ingredients', '').split(';')) if recipe_details.get('ingredients') else 0
                instructions_count = len(recipe_details.get('instructions', '').split(';')) if recipe_details.get('instructions') else 0
                logger.info(f"✅ [Page {page_num}] {scraping_stats['total_processed']}/{total_recipes} - {title[:40]}... (I:{ingredients_count}, S:{instructions_count})")
                return recipe_details
            else:
                scraping_stats['failed'] += 1
                logger.warning(f"❌ [Page {page_num}] {scraping_stats['total_processed']}/{total_recipes} - Failed: {recipe_url}")
                return None
                
    except Exception as e:
        with scraping_stats['lock']:
            scraping_stats['total_processed'] += 1
            scraping_stats['failed'] += 1
            logger.error(f"❌ [Page {page_num}] Error processing {recipe_url}: {e}")
        return None

def scrape_recipes_from_page_parallel(recipe_urls, page_num, max_workers=8):
    """
    Scrape recipes from a list of URLs using multi-threading
    """
    recipes = []
    total_urls = len(recipe_urls)
    
    if not recipe_urls:
        return recipes
        
    logger.info(f"🚀 [Page {page_num}] Processing {total_urls} recipes with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all recipe extraction tasks
        future_to_url = {
            executor.submit(extract_recipe_details_threaded, url, page_num, i, total_urls): url 
            for i, url in enumerate(recipe_urls, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            recipe_url = future_to_url[future]
            try:
                recipe_details = future.result()
                if recipe_details:
                    recipes.append(recipe_details)
            except Exception as e:
                logger.error(f"❌ Exception for {recipe_url}: {e}")
    
    return recipes

def scrape_all_recipes(start_page=1, max_workers=None):
    """
    Scrape all recipes using multi-threading for recipe detail extraction
    """
    base_url = "https://foodcnr.com/category/sri-lankan/"
    all_recipes = []
    
    # Set max workers (conservative for web scraping)
    if max_workers is None:
        max_workers = min(8, multiprocessing.cpu_count())
    
    logger.info(f"🍽️ Starting multi-threaded scraping from {base_url}")
    logger.info(f"⚡ Using {max_workers} threads for recipe extraction")
    
    # Reset stats
    scraping_stats.update({'total_processed': 0, 'successful': 0, 'failed': 0})
    
    # Get total pages
    total_pages = get_total_pages(base_url)
    logger.info(f"📖 Found {total_pages} total pages to scrape")
    
    start_time = time.time()
    
    for page in range(start_page, total_pages + 1):
        if page == 1:
            page_url = base_url
        else:
            page_url = f"{base_url}page/{page}/"
        
        logger.info(f"\n📄 [Page {page}/{total_pages}] Crawling: {page_url}")
        soup = get_soup(page_url)
        
        if not soup:
            logger.error(f"❌ Failed to load page {page}")
            continue
        
        # Extract recipe URLs from this page (sequential)
        recipe_urls = extract_recipe_urls(soup, page_url)
        logger.info(f"🔗 [Page {page}] Found {len(recipe_urls)} recipe URLs")
        
        if recipe_urls:
            # Extract details for all recipes on this page (parallel)
            page_recipes = scrape_recipes_from_page_parallel(recipe_urls, page, max_workers)
            all_recipes.extend(page_recipes)
            
            logger.info(f"✅ [Page {page}] Successfully extracted {len(page_recipes)}/{len(recipe_urls)} recipes")
        
        # Brief pause between pages to be respectful
        time.sleep(2)
    
    elapsed_time = time.time() - start_time
    recipes_per_second = len(all_recipes) / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"\n🎉 Scraping completed!")
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   ✅ Successful: {scraping_stats['successful']}")
    logger.info(f"   ❌ Failed: {scraping_stats['failed']}")
    logger.info(f"   📈 Total processed: {scraping_stats['total_processed']}")
    logger.info(f"   ⚡ Speed: {recipes_per_second:.2f} recipes/second")
    logger.info(f"   ⏱️ Total time: {elapsed_time:.1f}s")
    
    return all_recipes

def save_to_csv(recipes, filename):
    if not recipes:
        print("No recipes to save!")
        return
    
    fieldnames = ['title', 'description', 'author', 'published_date', 'prep_time', 'cooking_time', 
                  'total_time', 'difficulty', 'servings', 'ingredients', 'instructions', 'tags', 
                  'url', 'image_url']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for recipe in recipes:
            writer.writerow(recipe)
    
    print(f"\nSaved {len(recipes)} recipes to {filename}")

if __name__ == "__main__":
    logger.info("🇱🇰 Food Corner Sri Lankan Recipe Scraper - Multi-threaded Version")
    logger.info("=" * 70)
    
    # Get configuration from user
    max_workers = min(8, multiprocessing.cpu_count())
    logger.info(f"💻 Available CPU cores: {multiprocessing.cpu_count()}")
    logger.info(f"⚡ Using {max_workers} threads for optimal performance")
    
    try:
        # Scrape all recipes from page 1 with multi-threading
        recipes = scrape_all_recipes(start_page=1, max_workers=max_workers)
        
        if not recipes:
            logger.error("❌ No recipes were scraped!")
            exit(1)
        
        # Save to CSV
        csv_filename = "food_corner_complete_recipes_final.csv"
        save_to_csv(recipes, csv_filename)
        
        logger.info(f"\n🎊 Scraping completed successfully!")
        logger.info(f"📁 Results saved to: {csv_filename}")
        logger.info(f"🍽️ Total recipes found: {len(recipes)}")
        
        # Show data quality statistics
        with_ingredients = sum(1 for r in recipes if r.get('ingredients'))
        with_instructions = sum(1 for r in recipes if r.get('instructions'))
        with_servings = sum(1 for r in recipes if r.get('servings'))
        
        logger.info(f"\n📊 Data Quality Statistics:")
        logger.info(f"   🥕 Recipes with ingredients: {with_ingredients}/{len(recipes)} ({with_ingredients/len(recipes)*100:.1f}%)")
        logger.info(f"   📝 Recipes with instructions: {with_instructions}/{len(recipes)} ({with_instructions/len(recipes)*100:.1f}%)")
        logger.info(f"   👥 Recipes with servings: {with_servings}/{len(recipes)} ({with_servings/len(recipes)*100:.1f}%)")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Scraping interrupted by user")
        logger.info(f"📊 Partial results: {scraping_stats['successful']} successful, {scraping_stats['failed']} failed")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise