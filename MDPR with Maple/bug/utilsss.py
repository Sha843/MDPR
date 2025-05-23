from openai import OpenAI
import json
import os
import ast
from tenacity import retry, stop_after_attempt, wait_random_exponential
import config

try:
    API_KEY = config.DASHSCOPE_API_KEY
    BASE_URL = config.DASHSCOPE_BASE_URL

    if not API_KEY or "YOUR_" in API_KEY:
         print("Error: DashScope API Key (compatibility mode) not properly configured in config.py.")
         client = None
    elif not BASE_URL:
         print("Error: DashScope Base URL not configured in config.py.")
         client = None
    else:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        print(f"OpenAI client initialized, pointing to compatibility mode endpoint: {BASE_URL}")
except AttributeError:
    print("Error: DASHSCOPE_API_KEY or DASHSCOPE_BASE_URL not defined in config.py.")
    client = None
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_llm_description(prompt_text: str, max_tokens: int = config.LLM_MAX_TOKENS, temperature: float = config.LLM_TEMPERATURE) -> str | None:
    if client is None:
        print("Error: OpenAI client not initialized.")
        return None

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            description = response.choices[0].message.content.strip()
            description = description.strip('"').strip("'")
            if ":" in description[:20]: description = description.split(":", 1)[-1].strip()
            return description
        else:
            print("Warning: No valid text content found in API response.")
            print(f"  Response details: {response}")
            return None

    except Exception as e:
        print(f"Exception occurred while calling DashScope API (compatibility mode): {e}")
        raise

def save_json(data: dict | list, filepath: str):
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name, exist_ok=True); print(f"Created directory: {dir_name}")
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {filepath}")
    except Exception as e: print(f"Error saving JSON file to {filepath}: {e}")

def load_json(filepath: str) -> dict | list | None:
    if not os.path.exists(filepath): print(f"Error: File {filepath} does not exist."); return None
    try:
        with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
        print(f"Data successfully loaded from {filepath}"); return data
    except json.JSONDecodeError as e: print(f"Error: File {filepath} is not in valid JSON format. {e}"); return None
    except Exception as e: print(f"Error loading JSON file {filepath}: {e}"); return None

def load_prompt(filepath: str) -> list | dict | None:
    print(f"Attempting to load prompt/category names from file: {filepath}")
    if not os.path.exists(filepath): print(f"  Error: File not found {filepath}"); return None
    try:
        with open(filepath, 'r', encoding='utf-8') as file: content = file.read()
        data = ast.literal_eval(content)
        if isinstance(data, (list, dict)): print(f"  Successfully loaded and parsed {type(data).__name__}, containing {len(data)} items."); return data
        else: print(f"  Error: File content is not a list or dictionary (type: {type(data).__name__})"); return None
    except Exception as e: print(f"  Error loading or parsing file {filepath}: {e}"); return None