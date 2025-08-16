import requests
from bs4 import BeautifulSoup
import csv
import time
import re
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
    # Find recipe cards/links - specifically look for actual recipe URLs
    recipe_links = soup.find_all("a", href=True)
    
    for link in recipe_links:
        href = link.get("href", "")
        
        # Filter for actual individual recipe URLs only
        if (href.startswith('/recipes/') or 
            (href.startswith('https://www.hungrylankan.com/recipes/') or 
             href.startswith('https://hungrylankan.com/recipes/'))):
            full_url = urljoin(base_url, href)
            if full_url not in recipe_urls:
                recipe_urls.append(full_url)
    
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
        'difficulty': '',
        'servings': '',
        'ingredients': '',
        'instructions': '',
        'tags': '',
        'image_url': ''
    }
    
    # Extract title
    title_elem = soup.find("h1") or soup.find("title")
    if title_elem:
        recipe['title'] = title_elem.get_text(strip=True)
    
    # Extract description
    desc_elem = soup.find("meta", {"name": "description"}) or soup.find("p", class_=re.compile("description|intro"))
    if desc_elem:
        if desc_elem.name == "meta":
            recipe['description'] = desc_elem.get("content", "")
        else:
            recipe['description'] = desc_elem.get_text(strip=True)
    
    # Extract cooking time
    time_elem = soup.find(string=re.compile(r"\d+\s*(min|hour|hr)", re.I))
    if time_elem:
        recipe['cooking_time'] = time_elem.strip()
    
    # Extract difficulty
    difficulty_keywords = ["beginner", "easy", "medium", "hard", "advanced", "intermediate"]
    for keyword in difficulty_keywords:
        if soup.find(string=re.compile(keyword, re.I)):
            recipe['difficulty'] = keyword.capitalize()
            break
    
    # Extract servings - look for dr-yields span (most accurate)
    yields_elem = soup.find("span", class_=["dr-sim-metaa", "dr-yields"])
    if yields_elem:
        # Extract number from text like "Servings:20"
        serving_text = yields_elem.get_text(strip=True)
        serving_match = re.search(r'(\d+)', serving_text)
        if serving_match:
            recipe['servings'] = serving_match.group(1)
    
    # Fallback: look for dr-scale-ingredients input
    if not recipe['servings']:
        scale_input = soup.find("input", class_="dr-scale-ingredients")
        if scale_input and scale_input.get('value'):
            recipe['servings'] = scale_input.get('value')
    
    # Fallback 1: look for print recipe link with servings parameter
    if not recipe['servings']:
        print_link = soup.find("a", href=re.compile(r"recipe_servings=\d+"))
        if print_link:
            href = print_link.get("href", "")
            servings_match = re.search(r"recipe_servings=(\d+)", href)
            if servings_match:
                recipe['servings'] = servings_match.group(1)
    
    # Fallback 2: look for servings in text
    if not recipe['servings']:
        serving_elem = soup.find(string=re.compile(r"serves?\s*\d+|servings?\s*\d+|\d+\s*portions?", re.I))
        if serving_elem:
            recipe['servings'] = serving_elem.strip()
    
    # Extract ingredients
    ingredients_section = soup.find("div", class_=re.compile("ingredients|recipe-ingredients")) or \
                         soup.find("section", class_=re.compile("ingredients")) or \
                         soup.find("ul", class_=re.compile("ingredients"))
    
    if ingredients_section:
        ingredients = []
        for li in ingredients_section.find_all("li"):
            ingredient = li.get_text(strip=True)
            if ingredient:
                ingredients.append(ingredient)
        recipe['ingredients'] = "; ".join(ingredients)
    
    # Extract instructions
    instructions_section = soup.find("div", class_=re.compile("instructions|method|directions")) or \
                          soup.find("section", class_=re.compile("instructions|method")) or \
                          soup.find("ol", class_=re.compile("instructions|method|steps"))
    
    if instructions_section:
        instructions = []
        for item in instructions_section.find_all(["li", "p"]):
            instruction = item.get_text(strip=True)
            if instruction and len(instruction) > 10:
                instructions.append(instruction)
        recipe['instructions'] = "; ".join(instructions)
    
    # Extract tags
    tags = []
    for tag_elem in soup.find_all(["span", "a"], class_=re.compile("tag|category|label")):
        tag = tag_elem.get_text(strip=True)
        if tag and tag not in tags:
            tags.append(tag)
    recipe['tags'] = ", ".join(tags)
    
    # Extract main image
    img_elem = soup.find("img", src=True)
    if img_elem:
        img_src = img_elem.get("src", "")
        recipe['image_url'] = urljoin(url, img_src)
    
    return recipe

def scrape_all_recipes():
    base_url = "https://www.hungrylankan.com/recipe-cuisine/srilankan/"
    all_recipes = []
    page = 1
    max_pages = 11  # Based on the pagination info from the website
    
    print(f"Starting to scrape recipes from {base_url}")
    
    while page <= max_pages:
        if page == 1:
            page_url = base_url
        else:
            page_url = f"{base_url}page/{page}/"
        
        print(f"\nScraping page {page}: {page_url}")
        soup = get_soup(page_url)
        
        if not soup:
            print(f"Failed to load page {page}")
            page += 1
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
                print(f"    [OK] Extracted: {title}")
            else:
                print(f"    [ERROR] Failed to extract recipe details")
            
            # Be polite to the server
            time.sleep(1)
        
        page += 1
        time.sleep(2)  # Wait between pages
    
    return all_recipes

def save_to_csv(recipes, filename):
    if not recipes:
        print("No recipes to save!")
        return
    
    fieldnames = ['title', 'description', 'cooking_time', 'difficulty', 'servings', 
                  'ingredients', 'instructions', 'tags', 'url', 'image_url']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for recipe in recipes:
            writer.writerow(recipe)
    
    print(f"\nSaved {len(recipes)} recipes to {filename}")

if __name__ == "__main__":
    print("Hungry Lankan Recipe Scraper")
    print("=" * 40)
    
    # Scrape all recipes
    recipes = scrape_all_recipes()
    
    # Save to CSV
    csv_filename = "hungry_lankan_recipes_updated.csv"
    save_to_csv(recipes, csv_filename)
    
    print(f"\nScraping completed! Found {len(recipes)} recipes.")
    print(f"Results saved to: {csv_filename}")