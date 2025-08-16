import requests
from bs4 import BeautifulSoup
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

BASE_URL = "https://www.hungrylankan.com/recipe-cuisine/{}/page/{}/"
CATEGORIES = ["srilankan"]
OUTPUT_FILE = "hungry_lankan_recipes.csv"

FIELDS = ["title", "ingredients", "instructions", "category", "serves"]

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        print(f"[DEBUG] GET {url} → Status {res.status_code}")
        if res.status_code == 200:
            return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] Request failed for {url}: {e}")
    return None

def scrape_recipe(url, category):
    soup = get_soup(url)
    if not soup:
        print(f"[DEBUG] No soup for {url}")
        return None

    title_tag = soup.find("h1", class_="entry-title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    print(f"[DEBUG] Recipe title: {title}")

    serves_tag = soup.find(string=re.compile(r"Serves", re.IGNORECASE))
    serves = serves_tag.strip() if serves_tag else ""

    ingredients_list = []
    ingredients_section = soup.find("div", class_=re.compile(r"ingredients", re.IGNORECASE))
    if ingredients_section:
        ingredients_list = [li.get_text(" ", strip=True) for li in ingredients_section.find_all("li")]
    else:
        print(f"[DEBUG] No ingredients section found in {url}")

    instructions_list = []
    instructions_section = soup.find("div", class_=re.compile(r"instructions", re.IGNORECASE))
    if instructions_section:
        instructions_list = [step.get_text(" ", strip=True) for step in instructions_section.find_all(["li", "p"])]
    else:
        print(f"[DEBUG] No instructions section found in {url}")

    return {
        "title": title,
        "ingredients": " | ".join(ingredients_list),
        "instructions": " ".join(instructions_list),
        "category": category,
        "serves": serves
    }

def scrape_category(category):
    page = 1
    results = []

    while True:
        if page == 1:
            url = f"https://www.hungrylankan.com/recipe-cuisine/{category}/"
        else:
            url = f"https://www.hungrylankan.com/recipe-cuisine/{category}/page/{page}/"

        soup = get_soup(url)
        if not soup:
            break

        # selector for recipe links
        recipe_links = [a["href"] for a in soup.select("h2.post-title a")]
        print(f"[DEBUG] Found {len(recipe_links)} recipe links on page {page}: {recipe_links}")

        if not recipe_links:
            break

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(scrape_recipe, link, category) for link in recipe_links]
            for future in as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        page += 1

    return results


def main():
    all_recipes = []

    for category in CATEGORIES:
        all_recipes.extend(scrape_category(category))
        time.sleep(1)

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for recipe in all_recipes:
            writer.writerow(recipe)

    print(f"Scraping completed: {len(all_recipes)} recipes saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
