import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import json
from urllib.parse import urljoin, urlparse

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return BeautifulSoup(res.text, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
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
            
            # Check if it's a Recipe schema
            if data.get("@type") == "Recipe":
                # Extract title
                if data.get("name"):
                    recipe['title'] = data["name"]
                
                # Extract description
                if data.get("description"):
                    recipe['description'] = data["description"]
                
                # Extract author
                author = data.get("author")
                if author:
                    if isinstance(author, dict):
                        recipe['author'] = author.get("name", "")
                    elif isinstance(author, str):
                        recipe['author'] = author
                
                # Extract published date
                if data.get("datePublished"):
                    recipe['published_date'] = data["datePublished"]
                
                # Extract servings/yield
                if data.get("recipeYield"):
                    yield_val = data["recipeYield"]
                    if isinstance(yield_val, list):
                        recipe['servings'] = str(yield_val[0])
                    else:
                        recipe['servings'] = str(yield_val)
                
                # Extract cooking times
                if data.get("prepTime"):
                    recipe['prep_time'] = data["prepTime"]
                if data.get("cookTime"):
                    recipe['cooking_time'] = data["cookTime"]
                if data.get("totalTime"):
                    recipe['total_time'] = data["totalTime"]
                
                # Extract ingredients
                ingredients = data.get("recipeIngredient", [])
                if ingredients:
                    recipe['ingredients'] = "; ".join(ingredients)
                
                # Extract instructions
                instructions_data = data.get("recipeInstructions", [])
                if instructions_data:
                    instructions = []
                    for instruction in instructions_data:
                        if isinstance(instruction, dict):
                            if instruction.get("@type") == "HowToStep":
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
                image = data.get("image")
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
            r'(?:yields?|makes?)\s+(\d+(?:-\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.I)
            if match:
                recipe['servings'] = match.group(1)
                break
    
    # Extract ingredients from HTML if not found in JSON-LD
    if not recipe['ingredients']:
        content_div = soup.find("div", class_=re.compile("content|entry|post-content", re.I))
        if content_div:
            # Look for ingredient lists
            ingredient_lists = content_div.find_all("ul")
            for ul in ingredient_lists:
                ingredients = []
                for li in ul.find_all("li"):
                    ingredient = li.get_text(strip=True)
                    if ingredient and len(ingredient) > 2:
                        # Check if this looks like an ingredient
                        if any(word in ingredient.lower() for word in ['tsp', 'tbsp', 'cup', 'gram', 'kg', 'ml', 'liter', 'piece', 'clove', 'inch', 'no', 'nos']):
                            ingredients.append(ingredient)
                if ingredients and len(ingredients) >= 3:
                    recipe['ingredients'] = "; ".join(ingredients)
                    break
    
    # Extract instructions from HTML if not found in JSON-LD
    if not recipe['instructions']:
        content_div = soup.find("div", class_=re.compile("content|entry|post-content", re.I))
        if content_div:
            # Look for ordered lists first
            instruction_ol = content_div.find("ol")
            if instruction_ol:
                instructions = []
                for li in instruction_ol.find_all("li"):
                    instruction = li.get_text(strip=True)
                    if instruction and len(instruction) > 10:
                        instructions.append(instruction)
                if instructions:
                    recipe['instructions'] = "; ".join(instructions)
            
            # Look for step-by-step paragraphs
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

def scrape_all_recipes():
    base_url = "https://foodcnr.com/category/sri-lankan/"
    all_recipes = []
    
    print(f"Starting to scrape recipes from {base_url}")
    
    # Get total pages
    total_pages = get_total_pages(base_url)
    print(f"Found {total_pages} total pages to scrape")
    
    for page in range(1, total_pages + 1):  # Scrape ALL pages
        if page == 1:
            page_url = base_url
        else:
            page_url = f"{base_url}page/{page}/"
        
        print(f"\nScraping page {page}/{total_pages}: {page_url}")
        soup = get_soup(page_url)
        
        if not soup:
            print(f"Failed to load page {page}")
            continue
        
        # Extract recipe URLs from this page
        recipe_urls = extract_recipe_urls(soup, page_url)
        print(f"Found {len(recipe_urls)} recipe URLs on page {page}")
        
        # Extract details for each recipe
        for i, recipe_url in enumerate(recipe_urls, 1):
            print(f"  Extracting recipe {i}/{len(recipe_urls)}: {recipe_url}")
            recipe_details = extract_recipe_details(recipe_url)
            
            if recipe_details and recipe_details['title']:
                all_recipes.append(recipe_details)
                # Use ASCII-safe output
                title = recipe_details['title'].encode('ascii', 'ignore').decode('ascii')
                ingredients_count = len(recipe_details.get('ingredients', '').split(';')) if recipe_details.get('ingredients') else 0
                instructions_count = len(recipe_details.get('instructions', '').split(';')) if recipe_details.get('instructions') else 0
                print(f"    [OK] {title} (Ingredients: {ingredients_count}, Instructions: {instructions_count})")
            else:
                print(f"    [ERROR] Failed to extract recipe details")
            
            # Be polite to the server
            time.sleep(1)
        
        # Wait between pages
        time.sleep(2)
    
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
    print("Food Corner Sri Lankan Recipe Scraper - Complete Version")
    print("=" * 60)
    
    # Scrape all recipes
    recipes = scrape_all_recipes()
    
    # Save to CSV
    csv_filename = "food_corner_complete_recipes.csv"
    save_to_csv(recipes, csv_filename)
    
    print(f"\nScraping completed! Found {len(recipes)} recipes.")
    print(f"Results saved to: {csv_filename}")
    
    # Show statistics
    with_ingredients = sum(1 for r in recipes if r.get('ingredients'))
    with_instructions = sum(1 for r in recipes if r.get('instructions'))
    with_servings = sum(1 for r in recipes if r.get('servings'))
    
    print(f"\nData Quality Statistics:")
    print(f"Recipes with ingredients: {with_ingredients}/{len(recipes)} ({with_ingredients/len(recipes)*100:.1f}%)")
    print(f"Recipes with instructions: {with_instructions}/{len(recipes)} ({with_instructions/len(recipes)*100:.1f}%)")
    print(f"Recipes with servings: {with_servings}/{len(recipes)} ({with_servings/len(recipes)*100:.1f}%)")