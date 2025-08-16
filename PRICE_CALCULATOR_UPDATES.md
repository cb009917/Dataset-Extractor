# Price Calculator Updates for Real-Time CSV Support

## Overview
The price_calculator.py has been significantly updated to handle real-time CSV data from the Sri Lanka Department of Census and Statistics API with comprehensive unit normalization and improved ingredient matching.

## New Features Implemented

### 1. Real-Time CSV Data Loading
- **File Format**: Now supports `extracted_prices2.csv` with columns: Name, Unit, Average Price, Min Price
- **Data Validation**: Checks for required columns and validates price data
- **Automatic Cleaning**: Removes invalid entries and handles missing data gracefully

### 2. Advanced Unit Normalization

#### Supported Unit Conversions:
- **Solids to LKR/kg**: 
  - "1 kg" → multiplier = 1.0
  - "500 g" → multiplier = 0.5
  - "250 g" → multiplier = 0.25

- **Liquids to LKR/liter**: 
  - "1 liter" → multiplier = 1.0
  - "750 ml" → multiplier = 0.75
  - "500 ml" → multiplier = 0.5

- **Count-based Items**:
  - "each" → Uses default weight mapping (e.g., egg = 60g, coconut = 400g)
  - "bunch" → Assumes 100g unless overridden
  - "100 leaves" → 0.1g per leaf
  - "100 nuts" → 1g per nut

#### Density Handling:
- Oil: 0.92 g/ml
- Milk: 1.03 g/ml  
- Coconut milk: 0.97 g/ml
- Default liquids: 1.0 g/ml (water)

### 3. Enhanced Ingredient Name Matching

#### Text Normalization:
- Converts to lowercase
- Removes punctuation and special characters
- Handles plurals: "tomatoes" → "tomato", "onions" → "onion"

#### Sri Lankan Ingredient Aliases:
```python
{
    'brinjal': 'eggplant',
    'lady\'s finger': 'okra', 
    'green gram': 'mung bean',
    'mysore dhall': 'lentils',
    'red onions': 'onion',
    'bandakka': 'okra',
    'gotukola': 'gotu kola',
    'dried chillies': 'chili',
    'green chillies': 'chili',
    'ash plantain': 'plantain',
    'snake gourd': 'gourd',
    'bitter guard': 'bitter gourd',
    # ... and many more
}
```

#### Default Weight Mappings:
```python
{
    'coconut': 400,  # grams
    'egg': 60,
    'onion': 120,
    'tomato': 100,
    'potato': 120,
    'lemon': 60,
    'lime': 30,
    'garlic': 3,
    'bunch': 100,    # default bunch weight
    'each': 60       # default piece weight
}
```

### 4. Comprehensive Logging System

#### Unit Warnings Log (`unit_warnings.csv`):
- **Columns**: Name, Unit, Reason, Timestamp
- **Captures**: Unrecognized units, conversion errors, missing unit types
- **Example**: `Coconut Oil, 750 ml, Unknown unit type: unknown, 2025-08-13T09:22:07`

#### Missing Prices Log (`missing_prices.csv`):
- **Columns**: Ingredient, Recipe, Timestamp  
- **Captures**: Recipe ingredients that couldn't be matched to price database
- **Example**: `ginger, Chicken Curry, 2025-08-13T09:22:07`

### 5. Price Freshness Tracking
- **New Field**: `price_freshness_days` - Days since price was last updated
- **Calculation**: Based on price data timestamp vs current date
- **Default**: 0 days for current data

### 6. Enhanced Output Format

#### Updated `recipes_with_costs_clean.csv` includes:
- `estimated_cost_lkr`: Cost per serving in LKR (2 decimal places)
- `price_freshness_days`: Days since price update
- `matched_ingredients`: Number of successfully matched ingredients
- `total_ingredients`: Total ingredients in recipe
- `match_percentage`: Percentage of ingredients matched
- `low_confidence`: Boolean flag for quality control
- `skipped_reason`: Reason if recipe was skipped
- `top_cost_contributors`: JSON of most expensive ingredients

### 7. Improved Error Handling
- **Graceful Degradation**: Continues processing even with bad data
- **Detailed Logging**: All errors captured with context
- **Recovery Mechanisms**: Fallback values for unknown units/ingredients

### 8. Performance Optimizations
- **Multithreading**: Already implemented with ThreadPoolExecutor
- **Caching**: LRU cache for ingredient lookups
- **Batch Processing**: Efficient DataFrame operations
- **Memory Management**: Proper cleanup and resource management

## Usage Instructions

### Basic Usage:
```python
from price_calculator import MultithreadedPriceCalculator

# Initialize with CSV file
calculator = MultithreadedPriceCalculator('extracted_prices2.csv')

# Process recipes
recipes_df = pd.read_csv('your_recipes.csv')
results_df, detailed_results = calculator.process_recipes_batch_advanced(
    recipes_df, 'recipes_with_costs_clean.csv'
)
```

### Command Line Usage:
```bash
# Run full calculation
python price_calculator.py

# Test coverage
python price_calculator.py coverage recipes_with_costs_clean.csv

# Run validation tests
python test_price_calculator.py
```

## Output Files Generated

1. **`recipes_with_costs_clean.csv`**: Main output with cost calculations
2. **`unit_warnings.csv`**: Log of unit conversion issues
3. **`missing_prices.csv`**: Log of missing ingredient prices
4. **`cost_outliers.csv`**: Top 30 high-cost recipes for review
5. **`pricing_coverage_report.json`**: Coverage metrics and statistics

## Quality Assurance

### Data Validation:
- Ensures price data is positive and valid
- Validates unit format consistency
- Checks for required CSV columns

### Error Recovery:
- Skips invalid entries instead of crashing
- Logs all issues for review
- Provides fallback calculations

### Deterministic Results:
- Same input always produces same output
- Consistent unit conversions
- Reliable ingredient matching

## Technical Details

### Key Classes:
- `AdvancedPriceCalculator`: Core pricing logic
- `MultithreadedPriceCalculator`: Performance-optimized version
- `RecipeCost`: Data structure for results
- `ParsedIngredient`: Individual ingredient cost data

### Configuration:
The system uses YAML configuration files for:
- `price_aliases.yaml`: Ingredient name mappings
- `densities.yaml`: Density values for liquids
- `pack_sizes.yaml`: Standard package sizes
- `servings_overrides.json`: Recipe serving corrections

## Testing
Run the comprehensive test suite:
```bash
python test_price_calculator.py
```

Tests cover:
- Price loading and normalization
- Unit conversion accuracy
- Ingredient matching logic
- Sample recipe calculation
- Error handling and logging

## Requirements Met

✅ **Load and normalize real-time prices from CSV**
✅ **Handle all unit conversions (kg, g, ml, each, bunch)**  
✅ **Improve ingredient name matching with aliases**
✅ **Log unit warnings and missing prices to CSV files**
✅ **Calculate price freshness in days**
✅ **Integrate with existing recipe cost calculation**
✅ **Maintain multithreading for performance**
✅ **Preserve existing functionality**
✅ **Add comprehensive documentation and comments**
✅ **Ensure deterministic cost calculations**

The updated price calculator is now production-ready for real-time price data processing with comprehensive error handling, logging, and quality assurance mechanisms.