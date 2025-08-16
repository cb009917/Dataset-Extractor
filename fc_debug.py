import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return BeautifulSoup(res.text, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def debug_page_structure(url):
    """Debug the page structure to find where recipe data is stored"""
    print(f"\nDebugging: {url}")
    soup = get_soup(url)
    if not soup:
        return None
    
    # Check all script tags
    all_scripts = soup.find_all("script")
    print(f"Found {len(all_scripts)} script tags total")
    
    # Check JSON-LD scripts specifically
    json_scripts = soup.find_all("script", type="application/ld+json")
    print(f"Found {len(json_scripts)} JSON-LD scripts")
    
    for i, script in enumerate(json_scripts):
        try:
            data = json.loads(script.string)
            print(f"\nJSON-LD {i+1}:")
            print(f"  Raw type: {type(data)}")
            if isinstance(data, list):
                print(f"  List length: {len(data)}")
                for j, item in enumerate(data):
                    print(f"    Item {j+1} type: {item.get('@type') if isinstance(item, dict) else type(item)}")
                    if isinstance(item, dict):
                        print(f"    Item {j+1} keys: {list(item.keys())[:10]}")
            elif isinstance(data, dict):
                print(f"  Dict @type: {data.get('@type')}")
                print(f"  Dict keys: {list(data.keys())[:10]}")
                
                # Check @graph structure
                if '@graph' in data:
                    graph = data['@graph']
                    print(f"  Found @graph with {len(graph) if isinstance(graph, list) else 1} items")
                    if isinstance(graph, list):
                        for k, item in enumerate(graph):
                            if isinstance(item, dict):
                                print(f"    Graph item {k+1} @type: {item.get('@type')}")
                                if item.get('@type') == 'Recipe':
                                    print(f"    FOUND RECIPE in graph item {k+1}!")
                                    print(f"    Recipe keys: {list(item.keys())}")
                                    
                                    # Check ingredients
                                    if 'ingredients' in item:
                                        ingredients = item['ingredients']
                                        print(f"    Found ingredients: {len(ingredients) if isinstance(ingredients, list) else 1} items")
                                        if isinstance(ingredients, list) and len(ingredients) > 0:
                                            print(f"    Sample ingredient: {ingredients[0]}")
                                    
                                    # Check recipeInstructions
                                    if 'recipeInstructions' in item:
                                        instructions = item['recipeInstructions']
                                        print(f"    Found recipeInstructions: {len(instructions) if isinstance(instructions, list) else 1} items")
                                        if isinstance(instructions, list) and len(instructions) > 0:
                                            print(f"    Sample instruction: {instructions[0]}")
                
                # Look for ingredients specifically
                if 'ingredients' in data:
                    ingredients = data['ingredients']
                    print(f"  Found 'ingredients' key with {len(ingredients) if isinstance(ingredients, list) else 1} items")
                    if isinstance(ingredients, list) and len(ingredients) > 0:
                        print(f"  Sample ingredient: {ingredients[0]}")
                
                # Look for recipe instructions
                if 'recipeInstructions' in data:
                    instructions = data['recipeInstructions'] 
                    print(f"  Found 'recipeInstructions' key with {len(instructions) if isinstance(instructions, list) else 1} items")
                    if isinstance(instructions, list) and len(instructions) > 0:
                        print(f"  Sample instruction: {instructions[0]}")
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"JSON parsing error for script {i+1}: {e}")
    
    # Check for other script tags that might contain recipe data
    other_scripts = soup.find_all("script", type=lambda x: x != "application/ld+json")
    print(f"\nFound {len(other_scripts)} other script tags")
    
    for i, script in enumerate(other_scripts[:5]):  # Check first 5
        if script.string:
            script_content = script.string.strip()
            if any(word in script_content.lower() for word in ['recipe', 'ingredient', 'instruction']):
                print(f"Script {i+1} might contain recipe data (length: {len(script_content)})")
                # Look for JSON-like structures
                if 'ingredients' in script_content.lower():
                    print(f"  Contains 'ingredients'")
                if 'instructions' in script_content.lower():
                    print(f"  Contains 'instructions'")
    
    # Check for recipe card elements
    recipe_cards = soup.find_all("div", class_=re.compile("recipe", re.I))
    print(f"\nFound {len(recipe_cards)} elements with 'recipe' in class")
    
    # Check for structured data in regular elements
    ingredients_sections = soup.find_all(["div", "section", "ul"], class_=re.compile("ingredient", re.I))
    print(f"Found {len(ingredients_sections)} elements with 'ingredient' in class")
    
    instructions_sections = soup.find_all(["div", "section", "ol"], class_=re.compile("instruction|method|step", re.I))
    print(f"Found {len(instructions_sections)} elements with instruction-related classes")

if __name__ == "__main__":
    debug_page_structure("https://foodcnr.com/sri-lankan-mushroom-stir-fry/")