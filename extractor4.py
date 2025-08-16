import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Setup Selenium with Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # comment out if you want to see browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Load the search page
url = "https://cookpad.com/eng/search/sri%20lankan"
driver.get(url)
time.sleep(3)

# Scroll to load more recipes
print("Scrolling to load recipes...")
last_height = driver.execute_script("return document.body.scrollHeight")
for _ in range(5):  # Scroll 5 times (you can increase)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Get recipe links
recipes = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/eng/recipes/"]')
recipe_urls = list(set([r.get_attribute("href") for r in recipes]))

print(f"Found {len(recipe_urls)} recipe links.")

# Prepare CSV
csv_file = open("sri_lankan_recipes.csv", "w", newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(["Title", "Cook Time", "Servings", "Ingredients", "Instructions", "URL"])

# Visit each recipe page
for i, recipe_url in enumerate(recipe_urls):
    try:
        driver.get(recipe_url)
        time.sleep(2)

        title = driver.find_element(By.TAG_NAME, "h1").text.strip()

        try:
            cook_time = driver.find_element(By.CSS_SELECTOR, ".cook_time, .time, [class*=time]").text.strip()
        except NoSuchElementException:
            cook_time = "N/A"

        try:
            servings = driver.find_element(By.CSS_SELECTOR, ".yield, [class*=people]").text.strip()
        except NoSuchElementException:
            servings = "N/A"

        # Ingredients
        ingredients_list = driver.find_elements(By.CSS_SELECTOR, "li[data-ingredient-id]")
        if not ingredients_list:
            ingredients_list = driver.find_elements(By.CSS_SELECTOR, ".ingredients li, [class*=ingredient] li, ul li")
        ingredients = [ing.text.strip() for ing in ingredients_list if ing.text.strip()]
        ingredients_text = ", ".join(ingredients)

        # Instructions
        steps = driver.find_elements(By.CSS_SELECTOR, "ol li[data-step-id]")
        if not steps:
            steps = driver.find_elements(By.CSS_SELECTOR, "ol li, .step, .instruction, [class*=step]")
        instructions = [step.text.strip() for step in steps if step.text.strip()]
        instructions_text = " | ".join(instructions)

        writer.writerow([title, cook_time, servings, ingredients_text, instructions_text, recipe_url])
        print(f"[{i+1}] Saved: {title}")

    except Exception as e:
        print(f"[{i+1}] Failed to process: {recipe_url} | Error: {e}")

driver.quit()
csv_file.close()
print("✅ Done. Recipes saved to sri_lankan_recipes.csv")
