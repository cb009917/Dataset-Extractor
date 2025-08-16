import pandas as pd
import re
from fractions import Fraction
import logging
from typing import Optional, Tuple, Dict, List
import os
import csv
import json
from common_text import load_alias, save_alias, canonicalize_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngredientNormalizer:
    def __init__(self):
        """
        Normalize recipe ingredients to per-person tokens:
        - Unicode normalization (fractions, slashes)
        - Strip parentheses globally
        - Rewrite suffix quantities to leading qty unit name
        - Two-pass deglue (num->unit, unit->word)
        - A/B/C triage (drop fillers & micro-spices; require qty for material)
        - Per-person scaling with sensible precision
        - Metrics + warnings + needs_quantity.csv
        """

        # Unicode fraction mappings (normalize FIRST)
        self.unicode_fractions = {
            '⁄': '/', '½': '1/2', '¼': '1/4', '¾': '3/4', '⅓': '1/3', '⅔': '2/3',
            '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8', '⅕': '1/5', '⅖': '2/5',
            '⅗': '3/5', '⅘': '4/5', '⅙': '1/6', '⅚': '5/6'
        }

        # Canonical units + alias map
        self.canonical_units = [
            'kg','g','l','ml','tsp','tbsp','cup','oz','lb','pcs','clove','can','tin','pkt','bunch'
        ]
        self.unit_aliases: Dict[str, str] = {
            'gram':'g','grams':'g','kilogram':'kg','kilograms':'kg',
            'milliliter':'ml','milliliters':'ml','liter':'l','liters':'l','litre':'l','litres':'l',
            'teaspoon':'tsp','teaspoons':'tsp','tablespoon':'tbsp','tablespoons':'tbsp',
            'cups':'cup','ounce':'oz','ounces':'oz','pound':'lb','pounds':'lb','lbs':'lb',
            'piece':'pcs','pieces':'pcs','pc':'pcs','packet':'pkt','pack':'pkt','pkts':'pkt',
            'bunches':'bunch','cloves':'clove','cans':'can','tins':'tin'
        }

        # Build safe unit alternation (case-insensitive)
        self.unit_alt = r'(?:' + '|'.join(sorted(set(self.canonical_units + list(self.unit_aliases.keys())), key=len, reverse=True)) + r')'
        self.unit_boundary_alt = r'(?:' + '|'.join(self.canonical_units) + r')'

        # A/B/C triage
        self.drop_fillers_words = {
            'salt','pepper','black pepper','white pepper','water'
        }
        self.drop_fillers_phrases = {
            'to taste','as required','as needed','optional','for garnish','for serving',
            'for tempering','for marinade','for marination','for decoration'
        }
        # Micro-spices/herbs -> log then drop (and always exclude from denominators)
        self.micro_spices = {
            'turmeric','turmeric powder','cumin','cumin powder','mustard','mustard seed','fenugreek',
            'coriander','coriander powder','chilli','chili','chilli powder','chili powder','chilli flakes',
            'chili flakes','curry leaves','pandan','rampe','cardamom','clove','cloves','cinnamon','nutmeg','mace',
            'garam masala','curry powder','peppercorn','bay leaf','bay leaves'
        }
        # Material items require quantity
        self.material_keywords = {
            'coconut milk','milk','coconut oil','oil','ghee','butter','yogurt','curd',
            'chicken','beef','mutton','pork','egg','eggs','fish','tuna','sardine','sprat','prawn','shrimp',
            'rice','flour','lentil','lentils','dal','dhal','gram','chickpea','soya','soy',
            'noodle','noodles','pasta','spaghetti',
            'sugar','jaggery','treacle','honey',
            'onion','tomato','jackfruit','potato','sweet potato','pumpkin','carrot','leek','beans','peas',
            'cabbage','cauliflower','brinjal','eggplant','okra','ladies finger','beetroot','kale','spinach',
            'mushroom','paneer','tofu','sausage','bacon','bread','paratha','kothu','roti'
        }

        # Piece-like units
        self.piece_like = {'pcs','clove','bunch','can','tin','pkt'}
        
        # Piece-like ingredient keywords for suffix detection
        self.piece_ingredients = {
            'egg', 'eggs', 'chili', 'chilli', 'chilies', 'chillies', 'tomato', 'tomatoes',
            'onion', 'onions', 'lime', 'limes', 'lemon', 'lemons', 'garlic', 'clove', 'cloves'
        }

        # Mass/volume units for simple grams/ml checks
        self.mass_units = {'g','kg'}
        self.vol_units = {'ml','l'}
        self.spoon_units = {'tsp','tbsp'}

        # Internal collectors
        self.warnings: List[dict] = []
        self.needs_qty_rows: List[dict] = []
        
        # Load user aliases for ingredient name mapping
        self.user_aliases = load_alias()
        if self.user_aliases:
            logger.info(f"Loaded {len(self.user_aliases)} user aliases from config/item_alias_user.json")

    # ---------- Helpers ----------

    def _norm_unicode(self, s: str) -> str:
        for k, v in self.unicode_fractions.items():
            s = s.replace(k, v)
        return re.sub(r'\s+', ' ', s).strip()

    def _strip_parentheses(self, s: str) -> str:
        # Remove all (...) segments repeatedly
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'\([^)]*\)', '', s)
        # Also remove standalone parentheses
        s = s.replace('(', '').replace(')', '')
        return re.sub(r'\s+', ' ', s).strip()

    def _is_heading(self, s: str) -> bool:
        ls = s.strip().lower()
        if not ls: return True
        if set(ls) <= {'-',' ','='}: return True
        if ls.startswith('for ') and ls.endswith(':'): return True
        if '====' in ls or '-----' in ls: return True
        return False

    def _canon_unit(self, u: Optional[str]) -> Optional[str]:
        if not u: return None
        ul = u.lower().strip('.')
        return self.unit_aliases.get(ul, ul) if ul in (self.canonical_units + list(self.unit_aliases.keys())) else ul

    def _deglue_two_pass(self, s: str) -> str:
        # Pass 1: number -> unit (insert space)
        pat1 = rf'(?<![A-Za-z])(\d+(?:[.,]\d+)?(?:\s+\d+/\d+)?|\d+/\d+)\s*({self.unit_boundary_alt})\b'
        s = re.sub(pat1, r'\1 \2', s, flags=re.IGNORECASE)

        # Pass 2: unit -> word (insert space) - but be conservative  
        # Only deglue when unit is clearly separate from the word
        # Avoid breaking words that contain unit letters (like "ground" containing "g")
        pat2 = rf'(\d+(?:[.,]\d+)?)\s*({self.unit_boundary_alt})([A-Za-z])'
        s = re.sub(pat2, r'\1 \2 \3', s, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', s).strip()

    def _final_deglue_pass(self, token: str) -> str:
        """Final aggressive pass to ensure zero glued units"""
        if not token:
            return token
            
        # Clean up malformed tokens with conflicting units like "g /ml" 
        token = re.sub(r'\s+([gl])\s*/\s*(ml|l)\s+', ' ', token, flags=re.IGNORECASE)
        
        # Aggressively separate any unit that's glued to a word
        unit_pat = rf'\b({self.unit_boundary_alt})(?=[A-Za-z])'
        token = re.sub(unit_pat, r'\1 ', token, flags=re.IGNORECASE)
        
        # Final comprehensive de-glue patterns
        token = re.sub(r'(\d)([A-Za-z])', r'\1 \2', token)  # digit to letter
        
        return re.sub(r'\s+', ' ', token).strip()

    def _rewrite_suffix_quantity(self, s: str, servings: float = 1.0) -> str:
        # Examples: "Onion - 2", "Coconut Milk - 1 1/2 cups", "Jackfruit Seeds - 250g"
        # "Green chili - 3 or 4", "Eggs - 2", "Ingredient - 250g"
        
        # 1) Check for grams/ml suffix like "- 250 g" / "- 250g" / "– 250g"
        gram_pat = r'^\s*(?P<name>.+?)\s*[-–]\s*(?P<qty>\d+(?:\.\d+)?(?:\s+\d/\d)?|\d/\d)\s*(?P<unit>g|kg|ml|l)\b'
        m_gram = re.match(gram_pat, s, flags=re.IGNORECASE)
        if m_gram:
            name = m_gram.group('name').strip()
            qty_str = m_gram.group('qty').strip()
            unit = self._canon_unit(m_gram.group('unit'))
            # Parse quantity
            try:
                if re.match(r'^\d+\s+\d+/\d+$', qty_str):
                    whole, frac = qty_str.split()
                    qty = float(int(whole) + eval(frac))
                elif re.match(r'^\d+/\d+$', qty_str):
                    qty = float(eval(qty_str))
                else:
                    qty = float(qty_str.replace(',', '.'))
                # Scale per-person
                per_person_qty = qty / max(servings, 1.0)
                return f'{self._format_qty(per_person_qty, unit)} {unit} {name}'.strip()
            except:
                pass
        
        # 2) Check for piece counts: "- 3 or 4" / "- 3-4" / "- 3 to 4" / "- 3"
        piece_pat = r'^\s*(?P<name>.+?)\s*[-–]\s*(?P<qty1>\d+)(?:\s*(?:or|to|[-–])\s*(?P<qty2>\d+))?\b'
        m_piece = re.match(piece_pat, s, flags=re.IGNORECASE)
        if m_piece and not m_gram:  # Don't match if we already found grams
            name = m_piece.group('name').strip()
            qty1 = int(m_piece.group('qty1'))
            qty2 = int(m_piece.group('qty2')) if m_piece.group('qty2') else qty1
            # Choose upper bound for ranges
            qty = max(qty1, qty2)
            
            # Check if this looks like a piece-type ingredient
            name_lower = name.lower()
            if any(piece_word in name_lower for piece_word in self.piece_ingredients):
                # Scale per-person
                per_person_qty = qty / max(servings, 1.0)
                # Round to reasonable precision for pieces
                if per_person_qty >= 1:
                    formatted_qty = str(int(round(per_person_qty)))
                else:
                    formatted_qty = f'{per_person_qty:.1f}'
                return f'{formatted_qty} pcs {name}'.strip()
            # If no explicit unit and doesn't look like pieces, fall through to normal parsing
        
        # 3) Original logic for explicit units
        pat = rf'^\s*(?P<name>[A-Za-z].+?)\s*[-:]\s*(?P<qty>\d+\s+\d/\d|\d+/\d+|\d+(?:[.,]\d+)?)\s*(?P<unit>{self.unit_boundary_alt})\b'
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            name = m.group('name').strip()
            qty = m.group('qty').strip()
            unit = self._canon_unit(m.group('unit'))
            return f'{qty} {unit} {name}'.strip()

        # 4) qty with any unit word (including long names like "teaspoon")
        pat2 = rf'^\s*(?P<name>[A-Za-z].+?)\s*[-:]\s*(?P<qty>\d+\s+\d/\d|\d+/\d+|\d+(?:[.,]\d+)?)\s*(?P<unit>\w+)?\s*$'
        m2 = re.match(pat2, s, flags=re.IGNORECASE)
        if m2:
            name = m2.group('name').strip()
            qty = m2.group('qty').strip()
            unit = self._canon_unit(m2.group('unit')) if m2.group('unit') else None
            if unit:
                return f'{qty} {unit} {name}'.strip()
            else:
                return f'{qty} {name}'.strip()

        return s

    def _parse_leading(self, s: str) -> Tuple[Optional[float], Optional[str], str]:
        """
        Returns (qty, unit, name). Name is non-empty if parse okay.
        """
        s = s.strip()
        # qty [unit] name - require proper number, not just fraction like /4
        pat = rf'^\s*(?P<qty>\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)\s*(?P<unit>{self.unit_boundary_alt})?\b\s*(?P<name>.+?)\s*$'
        m = re.match(pat, s, flags=re.IGNORECASE)
        if not m:
            return None, None, s
        qty_str = m.group('qty').replace(',', '.')
        try:
            if re.match(r'^\d+\s+\d+/\d+$', qty_str):
                whole, frac = qty_str.split()
                qty_val = float(int(whole) + Fraction(frac))
            elif re.match(r'^\d+/\d+$', qty_str):
                qty_val = float(Fraction(qty_str))
            else:
                qty_val = float(qty_str)
        except Exception:
            qty_val = None
        unit = self._canon_unit(m.group('unit')) if m.group('unit') else None
        name = m.group('name').strip()
        return qty_val, unit, name

    def _primary_name(self, name: str) -> str:
        # Lowercase and remove leading descriptors like "fresh", "large", "small"
        name_l = name.lower()
        name_l = re.sub(r'\b(fresh|large|small|medium|finely|coarsely|ground|crushed|dried|red|green|yellow|ripe|chopped|sliced|minced)\b', '', name_l)
        return re.sub(r'\s+', ' ', name_l).strip()

    def _is_filler_or_meta(self, name_l: str, raw: str) -> bool:
        if any(p in raw.lower() for p in self.drop_fillers_phrases):
            return True
        if name_l in self.drop_fillers_words:
            return True
        return False

    def _is_micro_spice(self, name_l: str) -> bool:
        return any(k in name_l for k in self.micro_spices)

    def _is_material(self, name_l: str) -> bool:
        return any(k in name_l for k in self.material_keywords)

    def _pcs_fallback(self, qty: Optional[float], unit: Optional[str], name_l: str) -> Tuple[Optional[float], Optional[str]]:
        # If pure integer and food noun (NOT B-list), assign pcs
        if qty is not None and unit is None:
            if not self._is_micro_spice(name_l):
                return qty, 'pcs'
        return qty, unit

    def _grams_estimate_for_B(self, qty: float, unit: Optional[str]) -> float:
        # Only for micro-spices (B list) to support <2 g rule
        if unit in self.mass_units:
            return qty * (1000 if unit == 'kg' else 1)
        if unit in self.spoon_units:
            # crude but safe defaults for ground spices
            return qty * (2 if unit == 'tsp' else 6)
        if unit in self.vol_units:
            # assume density ~1 for small vols
            return qty  # ml -> ~g
        return 9999  # unknown -> don't drop

    def _format_qty(self, q: float, unit: Optional[str]) -> str:
        if q is None: return ''
        if unit in self.mass_units or unit in self.vol_units:
            # finer precision for g/ml, coarser for kg/l
            if unit in {'g','ml'}:
                return f'{round(q,1):g}'
            if unit in {'kg','l'}:
                return f'{round(q,3):g}'
        if unit in self.spoon_units:
            return f'{round(q,2):g}'
        if unit == 'pcs':
            return f'{int(round(q))}'
        return f'{round(q,2):g}'

    # ---------- Core normalization ----------

    def normalize_ingredient(self, raw: str, servings: float, title: str = "") -> Optional[str]:
        if not raw or not str(raw).strip():
            return None
        original = str(raw).strip()

        # 1) Normalize unicode and parentheses, strip comment tails
        s = self._norm_unicode(original)
        # NOTE: For this dataset, " - " separates ingredient from quantity, so don't strip it
        # s = re.sub(r'\s+-\s+.*$', '', s)  # Remove comment tails after " - "
        s = self._strip_parentheses(s)

        # 2) Drop headings / banners
        if self._is_heading(s):
            self.warnings.append({'title': title, 'reason':'Non-ingredient text', 'original': original, 'cleaned': ''})
            return None

        # 3) Rewrite suffix quantities to leading (pass servings for scaling)
        s = self._rewrite_suffix_quantity(s, servings)

        # 4) Alias BEFORE deglue, then two-pass deglue with canonical alternation
        def _unit_alias_sub(m):
            return self._canon_unit(m.group(0)) or m.group(0)
        s = re.sub(rf'\b{self.unit_alt}\b', lambda m: _unit_alias_sub(m), s, flags=re.IGNORECASE)
        
        # 5) Two-pass deglue (num->unit, unit->word)
        s = self._deglue_two_pass(s)

        # 6) Parse leading qty/unit/name
        qty, unit, name = self._parse_leading(s)
        name_l = self._primary_name(name)

        # 7) A/B/C triage before scaling
        if self._is_filler_or_meta(name_l, original):
            # Drop entirely
            return None

        if self._is_micro_spice(name_l):
            # If no qty -> drop; if qty present and tiny after scaling -> drop
            if qty is None:
                return None

        # 8) Handle missing quantity for material items (C)
        if qty is None:
            if self._is_material(name_l):
                self.needs_qty_rows.append({'title': title, 'original': original, 'reason': 'material_no_quantity'})
                return None
            else:
                # Drop items without quantities to maintain high digit/unit coverage
                # Only very specific exceptions should be kept without quantities
                if any(word in name_l for word in ['garnish', 'decoration', 'tempering', 'marinade']):
                    return name.strip()
                return None

        # 9) pcs fallback when integer with noun but no unit
        qty, unit = self._pcs_fallback(qty, unit, name_l)

        # 10) Per-person scaling
        if servings is None or servings <= 0:
            return None
        per_person = qty / float(servings)

        # 11) Tiny-spice rule for B-list: drop if <2 g equivalent
        if self._is_micro_spice(name_l):
            grams = self._grams_estimate_for_B(per_person, unit)
            if grams < 2:
                self.warnings.append({'title': title, 'reason':'Scaled mass <2g (B-list)', 'original': original, 'cleaned': ''})
                return None
            # Otherwise we still drop B-list from output to keep nutrition/pricing clean
            return None

        # 12) Drop water (even if quantified) from output
        if name_l == 'water':
            return None

        # 13) Emit normalized token
        unit = self._canon_unit(unit) if unit else unit
        qty_txt = self._format_qty(per_person, unit)
        if unit:
            token = f"{qty_txt} {unit} {name}".strip()
        else:
            token = f"{qty_txt} {name}".strip()

        # Final safety sweep just before emitting each token
        token = self._strip_parentheses(token)
        token = self._deglue_two_pass(token)
        
        # Final aggressive deglue pass to ensure 0 glued units (multiple passes)
        for _ in range(3):  # Multiple passes to ensure complete separation
            prev_token = token
            token = self._final_deglue_pass(token)
            if token == prev_token:  # No more changes
                break
        
        token = re.sub(r'\s+', ' ', token).strip()
        return token if token else None

    def normalize_ingredients_list(self, ingredients_str: str, servings: float, title: str = "") -> str:
        if not ingredients_str or pd.isna(ingredients_str):
            return ''
        parts = re.split(r'\s*\|\s*|[•·;]', str(ingredients_str))
        out = []
        for p in parts:
            p = p.strip()
            if not p: continue
            t = self.normalize_ingredient(p, servings, title)
            if t: out.append(t)
        return ' | '.join(out)

    def parse_serving_size(self, serving_str) -> Optional[float]:
        if pd.isna(serving_str) or serving_str == '':
            return None
        try:
            if isinstance(serving_str, (int, float)):
                return float(serving_str)
            s = str(serving_str)
            rng = re.search(r'(\d+)\s*[-]\s*(\d+)', s)
            if rng:
                return float(rng.group(2))
            m = re.search(r'(\d+)', s)
            return float(m.group(1)) if m else None
        except Exception as e:
            logger.debug(f"Error parsing serving size '{serving_str}': {e}")
            return None

    # ---------- DataFrame / CSV wrappers ----------

    def normalize_dataframe(self, df: pd.DataFrame,
                            ingredients_col: str = 'ingredients',
                            servings_col: str = 'servings') -> pd.DataFrame:
        logger.info(f"Normalizing {len(df)} recipes to single-person servings...")
        self.warnings.clear()
        self.needs_qty_rows.clear()

        out = df.copy()
        out['ingredients_per_person'] = ''

        normalized = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                ingredients = row.get(ingredients_col, '')
                servings_raw = row.get(servings_col, '')
                title = row.get('title', f'Recipe {idx+1}')
                servings = self.parse_serving_size(servings_raw)

                if not servings or servings <= 0:
                    out.at[idx, 'ingredients_per_person'] = ''
                    skipped += 1
                    continue

                out.at[idx, 'ingredients_per_person'] = self.normalize_ingredients_list(ingredients, servings, title)
                normalized += 1

                if (idx + 1) % 25 == 0:
                    logger.info(f"Progress: {idx+1}/{len(df)}")

            except Exception as e:
                logger.error(f"Error processing recipe {idx+1}: {e}")
                out.at[idx, 'ingredients_per_person'] = ''
                errors += 1

        # Write warnings & needs_quantity
        if self.warnings:
            with open('reports/normalization_warnings.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['title','reason','original','cleaned'])
                w.writeheader(); w.writerows(self.warnings)
            logger.info(f"Saved {len(self.warnings)} warnings to reports/normalization_warnings.csv")

        if self.needs_qty_rows:
            with open('needs_quantity.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['title','original','reason'])
                w.writeheader(); w.writerows(self.needs_qty_rows)
            logger.info(f"Saved {len(self.needs_qty_rows)} rows to needs_quantity.csv (material_no_quantity)")

        # Print quick metrics from output and write JSON report
        metrics = self._get_metrics(out)
        with open('reports/normalization_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self._print_metrics_from_dict(metrics)

        logger.info("\nNORMALIZATION SUMMARY:")
        logger.info(f"Normalized: {normalized}")
        logger.info(f"Skipped (no serving size): {skipped}")
        logger.info(f"Errors: {errors}")
        return out

    def _get_metrics(self, df: pd.DataFrame) -> Dict:
        tokens = []
        for s in df['ingredients_per_person'].fillna(''):
            tokens += [t.strip() for t in s.split('|') if t.strip()]
        total = len(tokens)
        if total == 0:
            return {
                "total_tokens": 0,
                "pct_digit_or_unit": 0.0,
                "glued_count": 0,
                "parentheses_count": 0
            }
        # digit/unit
        unit_pat = rf'\b{self.unit_boundary_alt}\b'
        digit_or_unit = sum(bool(re.search(r'\d', t) or re.search(unit_pat, t, flags=re.IGNORECASE)) for t in tokens)
        # glued
        glued_pat = rf'\b{self.unit_boundary_alt}(?=[A-Za-z])'
        glued = sum(bool(re.search(glued_pat, t, flags=re.IGNORECASE)) for t in tokens)
        # parentheses
        paren = sum(('(' in t or ')' in t) for t in tokens)

        return {
            "total_tokens": total,
            "pct_digit_or_unit": round(digit_or_unit/total*100, 2),
            "glued_count": glued,
            "parentheses_count": paren
        }

    def _print_metrics_from_dict(self, metrics: Dict):
        print(f"Total tokens: {metrics['total_tokens']}")
        print(f"% with digit or unit: {metrics['pct_digit_or_unit']}%")
        print(f"Glued unit->word count: {metrics['glued_count']}")
        print(f"Parentheses count: {metrics['parentheses_count']}")

    def normalize_csv_file(self, input_file: str, output_file: Optional[str] = None,
                           ingredients_col: str = 'ingredients', servings_col: str = 'servings'):
        logger.info(f"Loading recipes from {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')

        for col in (ingredients_col, servings_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Columns: {list(df.columns)}")

        out = self.normalize_dataframe(df, ingredients_col, servings_col)

        if output_file is None:
            base = os.path.splitext(input_file)[0]
            output_file = f"{base}_normalized.csv"

        out.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved normalized CSV to {output_file}")
        self.show_normalization_samples(out, ingredients_col)
        return out

    def show_normalization_samples(self, df: pd.DataFrame, original_col: str):
        print("\n" + "="*60)
        print("SAMPLE NORMALIZATION RESULTS")
        print("="*60)
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            title = row.get('title', f'Recipe {i+1}')
            servings = row.get('servings', 'Unknown')
            original = row.get(original_col, '')
            normalized = row.get('ingredients_per_person', '')
            print(f"\n{i+1}. {title[:40]}... (served {servings})")
            print(f"   BEFORE: {str(original)[:90]}...")
            print(f"   AFTER:  {str(normalized)[:90]}...")
        print("="*60)

# --- CLI ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Recipe Ingredient Normalizer")
        print("=" * 40)
        input_file = input("Enter CSV file path: ").strip().strip('"')
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            raise SystemExit(1)

        print("\nColumn names (Enter for defaults)")
        ingredients_col = input("Ingredients column (default: ingredients): ").strip() or 'ingredients'
        servings_col = input("Servings column (default: servings): ").strip() or 'servings'
        output_file = input("Output file (Enter for auto): ").strip() or None
    else:
        # Command line mode
        input_file = sys.argv[1]
        ingredients_col = sys.argv[2] if len(sys.argv) > 2 else 'ingredients'
        servings_col = sys.argv[3] if len(sys.argv) > 3 else 'servings'
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            raise SystemExit(1)

    normalizer = IngredientNormalizer()
    normalizer.normalize_csv_file(
        input_file=input_file,
        output_file=output_file,
        ingredients_col=ingredients_col,
        servings_col=servings_col
    )

# --- Quick tests when imported interactively (optional) ---
def test_normalizer():
    N = IngredientNormalizer()
    cases = [
        ("112 1/2gFresh Salmon", 2),
        ("1/4lime", 2),
        ("3green chilies", 3),
        ("/3 l bbeef", 1),
        ("cooking oil - as you need", 1),
        ("(pork shoulder or belly)", 1),
        ("=====FOR CURRY=====", 1),
        ("1/2 cup coconut milk", 2),
        ("1 3/4 cups flour", 4),
        ("2 c urry l eaves", 3),
        ("1 lbs. ground beef", 2),
        ("to taste salt", 1),
        ("Jackfruit Seeds - 250g", 4),
        ("Coconut Milk - 1 1/2 cups", 4),
        ("Mustard seeds - 1 tsp", 2),
        ("Onion - 1", 2),
        ("Water - 250 ml", 2),
        ("2tspginger", 2),
        ("1cupcoconut milk", 2),
        ("Best Chicken - For Curry -----", 2),
        ("Red lentils - 200 g", 2),
    ]
    for ing, srv in cases:
        print(f"{ing:<35} -> {N.normalize_ingredient(ing, srv, 'Test')}")