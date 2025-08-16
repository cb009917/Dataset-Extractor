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

def test_recipe_extraction(url):
    """Test extraction on a specific recipe URL"""
    print(f"\nTesting extraction for: {url}")
    soup = get_soup(url)
    if not soup:
        return None
    
    recipe = {
        'url': url,
        'title': '',
        'servings': '',
        'ingredients': '',
        'instructions': ''
    }
    
    # Extract title
    title_elem = soup.find("h1")
    if title_elem:
        recipe['title'] = title_elem.get_text(strip=True)
        print(f"Title: {recipe['title']}")
    
    # Method 1: Try JSON-LD first
    json_scripts = soup.find_all("script", type="application/ld+json")
    print(f"Found {len(json_scripts)} JSON-LD scripts")
    
    for i, script in enumerate(json_scripts):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                data = data[0]
            
            print(f"JSON-LD {i+1} type: {data.get('@type')}")
            
            # Check for @graph structure first
            recipes_to_check = []
            if data.get("@graph"):
                print("Found @graph structure")
                # Look for Recipe items in @graph
                for j, item in enumerate(data["@graph"]):
                    if isinstance(item, dict) and item.get("@type") == "Recipe":
                        print(f"Found Recipe in @graph item {j+1}")
                        recipes_to_check.append(item)
            elif data.get("@type") == "Recipe":
                recipes_to_check.append(data)
            
            # Process each recipe found
            for recipe_data in recipes_to_check:
                print("Found Recipe schema!")
                
                # Servings
                if recipe_data.get("recipeYield"):
                    yield_val = recipe_data["recipeYield"]
                    if isinstance(yield_val, list):
                        recipe['servings'] = str(yield_val[0])
                    else:
                        recipe['servings'] = str(yield_val)
                    print(f"Servings: {recipe['servings']}")
                
                # Ingredients
                ingredients = recipe_data.get("recipeIngredient", [])
                print(f"Ingredients count: {len(ingredients)}")
                if ingredients:
                    recipe['ingredients'] = "; ".join(ingredients)
                    print(f"Sample ingredients: {ingredients[:3]}")
                
                # Instructions - enhanced parsing
                instructions_data = recipe_data.get("recipeInstructions", [])
                print(f"Instructions data count: {len(instructions_data)}")
                
                if instructions_data:
                    instructions = []
                    for j, instruction in enumerate(instructions_data):
                        print(f"Instruction {j+1} type: {instruction.get('@type') if isinstance(instruction, dict) else type(instruction)}")
                        
                        if isinstance(instruction, dict):
                            if instruction.get("@type") == "HowToSection":
                                print("Found HowToSection")
                                section_name = instruction.get("name", "")
                                print(f"Section name: {section_name}")
                                
                                if "itemListElement" in instruction:
                                    print(f"itemListElement count: {len(instruction['itemListElement'])}")
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
                        print(f"Final instructions count: {len(instructions)}")
                        print(f"Sample instruction: {instructions[0] if instructions else 'None'}")
                
                return recipe
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"JSON parsing error: {e}")
            continue
    
    # Method 2: HTML fallback
    print("Using HTML fallback methods...")
    
    # Try to find serving size in text
    full_text = soup.get_text()
    patterns = [
        r'serves?\s*:?\s*(\d+(?:-\d+)?)\\s*(?:people|persons?)?',
        r'for\s+(\d+(?:-\d+)?)\s+(?:people|persons?)',
        r'(?:yields?|makes?)\s+(\d+(?:-\d+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.I)
        if match:
            recipe['servings'] = match.group(1)
            print(f"Found servings via regex: {recipe['servings']}")
            break
    
    # Try to find ingredients in HTML
    content_div = soup.find("div", class_=re.compile("content|entry|post-content", re.I))
    if content_div:
        ingredient_lists = content_div.find_all("ul")
        print(f"Found {len(ingredient_lists)} unordered lists")
        
        for k, ul in enumerate(ingredient_lists):
            ingredients = []
            for li in ul.find_all("li"):
                ingredient = li.get_text(strip=True)
                if ingredient and len(ingredient) > 2:
                    if any(word in ingredient.lower() for word in ['tsp', 'tbsp', 'cup', 'gram', 'kg', 'ml']):
                        ingredients.append(ingredient)
            
            print(f"List {k+1}: {len(ingredients)} potential ingredients")
            if ingredients and len(ingredients) >= 2:
                recipe['ingredients'] = "; ".join(ingredients)
                break
    
    return recipe

if __name__ == "__main__":
    # Test specific recipe URLs
    test_urls = [
        "https://foodcnr.com/sri-lankan-mushroom-stir-fry/",
        "https://foodcnr.com/black-eyed-beans-stir-fry/",
        "https://foodcnr.com/red-kidney-beans-curry-sri-lankan/"
    ]
    
    for url in test_urls:
        result = test_recipe_extraction(url)
        if result:
            print(f"Final result:")
            print(f"  Title: {result['title']}")
            print(f"  Servings: {result['servings']}")
            print(f"  Ingredients: {len(result['ingredients'].split(';')) if result['ingredients'] else 0}")
            print(f"  Instructions: {len(result['instructions'].split(';')) if result['instructions'] else 0}")
        print("-" * 80)