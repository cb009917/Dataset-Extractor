import requests
from bs4 import BeautifulSoup
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from difflib import SequenceMatcher

BASE_URL = "https://topsrilankanrecipe.com/category/{}/page/{}/"
CATEGORIES = ["vegetarian", "non-vegetarian", "starch", "dessert", "snack", "keto", "vegan", "healthy"]
OUTPUT_FILE = "sri_lankan_recipes.csv"

FIELDS = ["title", "ingredients", "instructions", "category", "serves"]

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"Request failed for {url}: {e}")
    return None

def scrape_recipe(url, category):
    """Extract recipe info from EasyRecipe printable card."""
    soup = get_soup(url)
    if not soup:
        return None

    # Title
    title_tag = soup.find("h1", class_="entry-title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # EasyRecipe container (id starts with easyrecipe-)
    recipe_card = soup.find("div", id=re.compile(r"^easyrecipe-\d+"))
    if not recipe_card:
        return None

    # Servings
    serves_tag = recipe_card.select_one(".ERSServes")
    serves = serves_tag.get_text(strip=True) if serves_tag else ""

    # Ingredients
    ingredients_list = [li.get_text(" ", strip=True) for li in recipe_card.select(".ERSIngredients li")]
    ingredients = " | ".join(ingredients_list)

    # Instructions
    instructions_list = [step.get_text(" ", strip=True) for step in recipe_card.select(".ERSInstructions li, .ERSInstructions p")]
    instructions = " ".join(instructions_list)

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions,
        "category": category,
        "serves": serves
    }

def scrape_category(category):
    """Scrape all recipes in a category by visiting each recipe page."""
    page = 1
    results = []

    while True:
        url = BASE_URL.format(category, page)
        soup = get_soup(url)
        if not soup:
            break

        recipe_links = [a["href"] for a in soup.select("h2.entry-title a")]
        if not recipe_links:
            break

        print(f"Scraping category '{category}', page {page}, found {len(recipe_links)} recipes...")

        # Multithreading for speed
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(scrape_recipe, link, category) for link in recipe_links]
            for future in as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        page += 1

    return results

def normalize_title(title):
    """Normalize title for comparison by removing common variations."""
    if not title:
        return ""
    
    # Convert to lowercase and remove extra whitespace
    normalized = re.sub(r'\s+', ' ', title.lower().strip())
    
    # Remove common variations
    variations = [
        r'\bsri\s+lankan?\b',
        r'\btraditional\b',
        r'\bauthentic\b', 
        r'\brecipe\b',
        r'\beasy\b',
        r'\bquick\b',
        r'\bhomemade\b',
        r'\bstyle\b',
        r'\bway\b'
    ]
    
    for pattern in variations:
        normalized = re.sub(pattern, '', normalized)
    
    # Remove extra spaces and punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def calculate_similarity(str1, str2):
    """Calculate similarity ratio between two strings."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def is_duplicate_recipe(recipe1, recipe2, title_threshold=0.85, ingredients_threshold=0.9):
    """Check if two recipes are duplicates based on title and ingredients similarity."""
    
    # Normalize titles for comparison
    title1 = normalize_title(recipe1.get('title', ''))
    title2 = normalize_title(recipe2.get('title', ''))
    
    # Skip if either title is empty
    if not title1 or not title2:
        return False
    
    # Calculate title similarity
    title_similarity = calculate_similarity(title1, title2)
    
    # If titles are very similar, it's likely a duplicate
    if title_similarity >= title_threshold:
        return True
    
    # Also check ingredients similarity for cases with different titles but same recipe
    ingredients1 = recipe1.get('ingredients', '').lower()
    ingredients2 = recipe2.get('ingredients', '').lower()
    
    if ingredients1 and ingredients2:
        ingredients_similarity = calculate_similarity(ingredients1, ingredients2)
        if ingredients_similarity >= ingredients_threshold:
            return True
    
    return False

def remove_duplicates(recipes):
    """Remove duplicate recipes from the list."""
    unique_recipes = []
    duplicates_count = 0
    
    print("Checking for duplicate recipes...")
    
    for i, recipe in enumerate(recipes):
        is_duplicate = False
        
        # Compare with all previously added unique recipes
        for unique_recipe in unique_recipes:
            if is_duplicate_recipe(recipe, unique_recipe):
                is_duplicate = True
                duplicates_count += 1
                print(f"Found duplicate: '{recipe.get('title', 'No title')}' (similar to '{unique_recipe.get('title', 'No title')}')")
                break
        
        if not is_duplicate:
            unique_recipes.append(recipe)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(recipes)} recipes...")
    
    print(f"Removed {duplicates_count} duplicate recipes")
    print(f"Unique recipes: {len(unique_recipes)}")
    
    return unique_recipes

def main():
    all_recipes = []

    for category in CATEGORIES:
        all_recipes.extend(scrape_category(category))
        time.sleep(1)  # small pause between categories

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
