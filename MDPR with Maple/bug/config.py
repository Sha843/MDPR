import os

DASHSCOPE_API_KEY = " "
DASHSCOPE_BASE_URL = " "
LLM_MODEL = " "

TOP_N_CONFUSING_PAIRS = 80

OUTPUT_SUBDIR = "cifar100"
ERROR_LOG_FILE = "1_clip_errors.json"
ANALYSIS_RESULTS_NAMED_FILE = os.path.join(OUTPUT_SUBDIR, "1_clip_errors.json")

DATASET = "cifar100"

PROMPT_FILE_TEMPLATE = "../clip/data/prompt_{dataset}.txt"

LLM_MAX_TOKENS = 150
LLM_TEMPERATURE = 0.6

if not DASHSCOPE_API_KEY or "YOUR_" in DASHSCOPE_API_KEY:
    print("Warning: DashScope API key (compatibility mode) in config.py is invalid or not properly set.")

print("--- Configuration Information ---")
print(f"Error log input file: {ERROR_LOG_FILE}")
print(f"All output files will be saved in subdirectory: {OUTPUT_SUBDIR}/")
print(f"LLM model: {LLM_MODEL}")
print(f"API Base URL: {DASHSCOPE_BASE_URL}")
print("---------------")