import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin, urlparse
from difflib import SequenceMatcher

BASE_URL = "https://pulse.lk/pulse-recipes/"
OUTPUT_FILE = "pulse_recipes.csv"
FIELDS = ["title", "ingredients", "instructions", "url", "author", "description"]

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"Request failed for {url}: {e}")
    return None

def extract_recipe_urls(soup, base_url):
    """Extract recipe URLs from the listing page"""
    recipe_urls = []
    
    # Look for recipe links - they should be within the pulse-recipes section
    recipe_links = soup.find_all("a", href=True)
    
    for link in recipe_links:
        href = link.get("href", "")
        # Filter for actual recipe URLs (not filter pages or comments)
        if ("pulse-recipes" in href and 
            href != base_url and
            "filter_by=" not in href and
            not href.endswith("#respond") and
            href.count("/") >= 4):  # Should have at least pulse.lk/pulse-recipes/recipe-name/
            full_url = urljoin(base_url, href)
            # Remove fragment if present
            if "#" in full_url:
                full_url = full_url.split("#")[0]
            if full_url not in recipe_urls:
                recipe_urls.append(full_url)
    
    return recipe_urls

def scrape_recipe(url):
    """Extract recipe details from individual recipe page"""
    soup = get_soup(url)
    if not soup:
        return None
    
    recipe = {
        'url': url,
        'title': '',
        'ingredients': '',
        'instructions': '',
        'author': '',
        'description': ''
    }
    
    # Extract title
    title_elem = soup.find("h1") or soup.find("title")
    if title_elem:
        recipe['title'] = title_elem.get_text(strip=True)
        # Clean up title
        if " | " in recipe['title']:
            recipe['title'] = recipe['title'].split(" | ")[0].strip()
    
    # Extract author
    author_elem = soup.find("span", class_="author") or soup.find("div", class_="author")
    if author_elem:
        recipe['author'] = author_elem.get_text(strip=True)
    
    # Extract meta description
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc:
        recipe['description'] = meta_desc.get("content", "")
    
    # Extract ingredients and instructions from content
    content = soup.find("div", class_=re.compile("content|entry-content|post-content", re.I))
    if content:
        content_text = content.get_text()
        
        # Extract ingredients - look for "Ingredients" section
        ingredients_match = re.search(r'\*\*Ingredients\*\*(.*?)(?:\*\*Method|\*\*Instructions|\*\*Directions|$)', content_text, re.DOTALL | re.I)
        if ingredients_match:
            ingredients_text = ingredients_match.group(1).strip()
            # Split ingredients into list
            ingredients_lines = [line.strip() for line in ingredients_text.split('\n') if line.strip()]
            # Filter out non-ingredient lines
            ingredients = []
            for line in ingredients_lines:
                # Skip lines that are too short or look like section headers
                if len(line) > 3 and not re.match(r'^\*\*', line):
                    # Clean up bullet points and formatting
                    line = re.sub(r'^[-•*]\s*', '', line)
                    if line:
                        ingredients.append(line)
            recipe['ingredients'] = "; ".join(ingredients)
        
        # Extract instructions - look for "Method" or "Instructions" section
        method_patterns = [
            r'\*\*Method:\*\*(.*?)(?:\*\*[A-Z]|$)',
            r'\*\*Instructions:\*\*(.*?)(?:\*\*[A-Z]|$)',
            r'\*\*Directions:\*\*(.*?)(?:\*\*[A-Z]|$)',
            r'Method:(.*?)(?:\*\*[A-Z]|$)'
        ]
        
        for pattern in method_patterns:
            instructions_match = re.search(pattern, content_text, re.DOTALL | re.I)
            if instructions_match:
                instructions_text = instructions_match.group(1).strip()
                # Split into steps
                instructions_lines = [line.strip() for line in instructions_text.split('\n') if line.strip()]
                # Filter and clean instructions
                instructions = []
                for line in instructions_lines:
                    if len(line) > 10 and not re.match(r'^\*\*', line):
                        # Remove numbering if present
                        line = re.sub(r'^\d+\.\s*', '', line)
                        line = re.sub(r'^[-•*]\s*', '', line)
                        if line:
                            instructions.append(line)
                recipe['instructions'] = "; ".join(instructions)
                break
        
        # Fallback: if no structured ingredients/instructions found, try to extract from paragraphs
        if not recipe['ingredients'] or not recipe['instructions']:
            paragraphs = content.find_all('p')
            all_text = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20:
                    all_text.append(text)
            
            # Try to identify ingredients and instructions from paragraphs
            for i, text in enumerate(all_text):
                if 'ingredients' in text.lower() and not recipe['ingredients']:
                    # Next paragraph might be ingredients
                    if i + 1 < len(all_text):
                        recipe['ingredients'] = all_text[i + 1]
                elif any(word in text.lower() for word in ['method', 'instructions', 'steps']) and not recipe['instructions']:
                    # Next paragraphs might be instructions
                    if i + 1 < len(all_text):
                        recipe['instructions'] = "; ".join(all_text[i + 1:i + 4])  # Take next 3 paragraphs
    
    return recipe

def load_more_recipes(base_url, max_loads=10):
    """Load more recipes using AJAX if available"""
    print("Checking for additional recipe pages...")
    
    # Try to find pagination or load more mechanism
    soup = get_soup(base_url)
    if not soup:
        return []
    
    all_recipe_urls = set()
    
    # Get initial recipe URLs
    initial_urls = extract_recipe_urls(soup, base_url)
    all_recipe_urls.update(initial_urls)
    print(f"Found {len(initial_urls)} recipe URLs on main page")
    
    # For now, we'll just scrape what's available on the main page
    # In the future, this could be enhanced to handle AJAX pagination
    
    return list(all_recipe_urls)

def normalize_title(title):
    """Normalize title for duplicate detection"""
    if not title:
        return ""
    
    normalized = re.sub(r'\s+', ' ', title.lower().strip())
    
    # Remove common variations
    variations = [
        r'\brecipe\b',
        r'\beasy\b',
        r'\bquick\b',
        r'\bhomemade\b',
        r'\btraditional\b',
        r'\bauthentic\b'
    ]
    
    for pattern in variations:
        normalized = re.sub(pattern, '', normalized)
    
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def is_duplicate_recipe(recipe1, recipe2, threshold=0.85):
    """Check if two recipes are duplicates"""
    title1 = normalize_title(recipe1.get('title', ''))
    title2 = normalize_title(recipe2.get('title', ''))
    
    if not title1 or not title2:
        return False
    
    similarity = SequenceMatcher(None, title1, title2).ratio()
    return similarity >= threshold

def remove_duplicates(recipes):
    """Remove duplicate recipes"""
    unique_recipes = []
    duplicates_count = 0
    
    print("Checking for duplicates...")
    
    for recipe in recipes:
        is_duplicate = False
        for unique_recipe in unique_recipes:
            if is_duplicate_recipe(recipe, unique_recipe):
                is_duplicate = True
                duplicates_count += 1
                print(f"Found duplicate: '{recipe.get('title', 'No title')}'")
                break
        
        if not is_duplicate:
            unique_recipes.append(recipe)
    
    print(f"Removed {duplicates_count} duplicates")
    return unique_recipes

def main():
    print("Pulse.lk Recipe Scraper")
    print("=" * 30)
    
    # Get all recipe URLs
    recipe_urls = load_more_recipes(BASE_URL)
    
    if not recipe_urls:
        print("No recipe URLs found!")
        return
    
    print(f"\nFound {len(recipe_urls)} total recipe URLs")
    print("Starting recipe extraction...")
    
    # Extract recipe details
    all_recipes = []
    
    for i, url in enumerate(recipe_urls, 1):
        print(f"Extracting recipe {i}/{len(recipe_urls)}: {url}")
        
        recipe_data = scrape_recipe(url)
        if recipe_data and recipe_data['title']:
            all_recipes.append(recipe_data)
            title = recipe_data['title'].encode('ascii', 'ignore').decode('ascii')
            print(f"  [OK] Extracted: {title}")
        else:
            print(f"  [ERROR] Failed to extract recipe")
        
        # Be polite to the server
        time.sleep(1)
    
    print(f"\nTotal recipes scraped: {len(all_recipes)}")
    
    # Remove duplicates
    unique_recipes = remove_duplicates(all_recipes)
    
    # Save to CSV
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for recipe in unique_recipes:
            writer.writerow(recipe)
    
    print(f"Scraping completed: {len(unique_recipes)} unique recipes saved to {OUTPUT_FILE}")
    print(f"Duplicates removed: {len(all_recipes) - len(unique_recipes)}")

if __name__ == "__main__":
    main()