import os
import sys
import time
import json
import argparse
import re
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(1, parent_dir)

try:
    from utilsss import load_json as utilsss_load_json, save_json, get_llm_description
    import config
    print("Successfully imported required functions and configuration from bug/utilsss.py and config.py.")
except ImportError:
    print("Error: Unable to import from bug/utilsss.py or config.py.")
    exit()

try:
    from clip import clip
    print("Imported clip library")
except ImportError:
    clip = None
    print("Warning: clip library not imported, unable to perform token checks.")

def load_prompt(filepath: str) -> list | dict | None:
    print(f"Attempting to load category names from file: {filepath}")
    path_to_check = filepath
    if not os.path.exists(path_to_check):
        script_dir_local = os.path.dirname(os.path.abspath(__file__))
        project_root_local = os.path.dirname(script_dir_local)
        if filepath.startswith("../"):
            path_to_check = os.path.abspath(os.path.join(script_dir_local, filepath))
        elif filepath.startswith("./clip/data"):
             path_to_check = os.path.join(project_root_local, filepath.lstrip("./"))
        else:
             pass
    if not os.path.exists(path_to_check):
        print(f"  Error: File not found {path_to_check} (original path: {filepath})")
        return None
    try:
        with open(path_to_check, 'r', encoding='utf-8') as file: content = file.read()
        data = ast.literal_eval(content)
        if isinstance(data, (list, dict)): print(f"  Successfully loaded and parsed {type(data).__name__}, containing {len(data)} items."); return data
        else: print(f"  Error: File content is not a list or dictionary (type: {type(data).__name__})"); return None
    except Exception as e: print(f"  Error loading or parsing file {path_to_check}: {e}"); return None

def create_visual_features_prompt(class_name: str, target_word_count: int = 25) -> str:
    return f"""Provide a concise English phrase describing the key visual appearance features of a "{class_name}".
Focus on what it looks like (e.g., shape, color, texture, notable parts).
The phrase should be approximately {target_word_count} words and suitable to complete the sentence: "A {class_name} typically appears as [YOUR PHRASE HERE]."
Output ONLY the descriptive phrase. Do NOT include "A {class_name} typically appears as".

Descriptive phrase for "{class_name}":
"""

def create_functional_use_prompt(class_name: str, target_word_count: int = 20) -> str:
    return f"""Provide a concise English phrase describing the primary function or purpose of a "{class_name}".
Focus on what it is used for.
The phrase should be approximately {target_word_count} words and suitable to complete the sentence: "A {class_name} is used for [YOUR PHRASE HERE]."
Output ONLY the descriptive phrase. Do NOT include "A {class_name} is used for".

Descriptive phrase for "{class_name}":
"""

def create_contextual_scene_prompt(class_name: str, target_word_count: int = 20) -> str:
    return f"""Provide a concise English phrase describing the common environments or contexts where a "{class_name}" is typically found.
Focus on its usual surroundings or scenarios.
The phrase should be approximately {target_word_count} words and suitable to complete the sentence: "A {class_name} is commonly found in [YOUR PHRASE HERE]."
Output ONLY the descriptive phrase. Do NOT include "A {class_name} is commonly found in".

Descriptive phrase for "{class_name}":
"""

def create_differential_comparison_prompt(class_name: str, confusing_class_name: str, target_word_count: int = 30) -> str:
    return f"""Describe the key visual differences of a "{class_name}" when compared to a "{confusing_class_name}".
Focus on features that distinguish a "{class_name}" from a "{confusing_class_name}".
The description should be in English, concise, and approximately {target_word_count} words.
Output ONLY the descriptive phrase itself, suitable for completing the sentence: "Unlike a {confusing_class_name}, a {class_name} has [YOUR PHRASE HERE]."
Output ONLY the descriptive phrase of differences. Do NOT include "Unlike a {confusing_class_name}, a {class_name} has".

Descriptive phrase of differences for "{class_name}" compared to "{confusing_class_name}":
"""

def create_finegrained_attribute_prompt(class_name: str, target_word_count: int = 20) -> str:
    return f"""Provide a concise English phrase describing one or two highly distinctive or fine-grained visual attributes of a "{class_name}" that make it unique or easily identifiable.
Focus on specific, detailed characteristics.
The description should be in English, concise, and approximately {target_word_count} words.
Output ONLY the descriptive phrase itself, suitable for completing the sentence: "A distinctive feature of a {class_name} is [YOUR PHRASE HERE]."
Output ONLY the descriptive phrase of the attribute(s). Do NOT include "A distinctive feature of a {class_name} is".

Descriptive phrase of attribute(s) for "{class_name}":
"""

def clean_llm_output(text: str) -> str:
    if not text: return ""
    text = text.strip().strip('"').strip("'")
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^\s*(Description for|Descriptive phrase for)\s*"[^"]+":\s*', '', text, flags=re.IGNORECASE).strip()
    if text.endswith('.'):
        text = text[:-1].strip()
    return text

def generate_descriptions_for_dimension(
    class_names: list,
    dimension: str,
    output_path: str,
    target_word_count: int,
    analysis_results: dict = None
):
    print(f"\n--- Starting to generate descriptions for dimension '{dimension}' ---")
    dimensional_library = {}
    total_items = len(class_names)
    processed_count, failed_count, fallback_to_visual_count = 0, 0, 0

    confusion_map = {}
    if dimension == "differential_comparison":
        if analysis_results:
            top_confusion_pairs = analysis_results.get("top_confusion_pairs", [])
            for pair_data in top_confusion_pairs:
                if isinstance(pair_data, (list, tuple)) and len(pair_data) == 2 and \
                   isinstance(pair_data[0], (list, tuple)) and len(pair_data[0]) == 2:
                    (true_name, pred_name), count = pair_data
                    if true_name not in confusion_map: confusion_map[true_name] = []
                    confusion_map[true_name].append(pred_name)
            print(f"  Loaded confusion pair information for {len(confusion_map)} categories.")
        else:
            print("  Warning: Requested differential comparison descriptions, but no analysis results (confusion pair information) provided.")

    for class_name in class_names:
        processed_count += 1
        print(f"  Processing category {processed_count}/{total_items}: '{class_name}' (dimension: {dimension})")

        prompt = ""
        final_sentence_template = ""
        confusing_class_for_template = ""
        current_dimension_is_diff_fallback = False

        if dimension == "visual_features":
            prompt = create_visual_features_prompt(class_name, target_word_count)
            final_sentence_template = "A {class_name} typically appears as {description}."
        elif dimension == "functional_use":
            prompt = create_functional_use_prompt(class_name, target_word_count)
            final_sentence_template = "A {class_name} is used for {description}."
        elif dimension == "contextual_scene":
            prompt = create_contextual_scene_prompt(class_name, target_word_count)
            final_sentence_template = "A {class_name} is commonly found in {description}."
        elif dimension == "differential_comparison":
            confusing_classes = confusion_map.get(class_name)
            if confusing_classes:
                confusing_class_for_template = confusing_classes[0]
                print(f"    Comparison object: '{confusing_class_for_template}'")
                prompt = create_differential_comparison_prompt(class_name, confusing_class_for_template, target_word_count)
                final_sentence_template = "Unlike a {confusing_class}, a {class_name} has {description}."
            else:
                print(f"    Warning: No confusion pair information found for category '{class_name}'. Generating [visual features] description as fallback.")
                prompt = create_visual_features_prompt(class_name, target_word_count)
                final_sentence_template = "A {class_name} typically appears as {description}."
                current_dimension_is_diff_fallback = True
                fallback_to_visual_count += 1
        elif dimension == "finegrained_attribute":
            prompt = create_finegrained_attribute_prompt(class_name, target_word_count)
            final_sentence_template = "A distinctive feature of a {class_name} is {description}."
        else:
            print(f"Error: Unknown description dimension '{dimension}'"); return False

        if not prompt:
            print(f"    Warning: No valid prompt generated for category '{class_name}' in dimension '{dimension}', skipping.")
            dimensional_library[class_name] = f"No prompt generated for {class_name} in dimension {dimension}."
            failed_count +=1
            continue

        try:
            print(f"    Calling LLM API (Prompt prefix: {prompt[:60].replace(os.linesep, ' ')}...)")
            max_desc_tokens = target_word_count * 3
            llm_output_phrase = get_llm_description(prompt, max_tokens=max_desc_tokens, temperature=0.5)

            if llm_output_phrase:
                cleaned_phrase = clean_llm_output(llm_output_phrase)
                print(f"      -> LLM output phrase (cleaned): {cleaned_phrase[:100]}...")
                if cleaned_phrase:
                    if dimension == "differential_comparison" and not current_dimension_is_diff_fallback:
                        final_description = final_sentence_template.format(
                            confusing_class=confusing_class_for_template,
                            class_name=class_name,
                            description=cleaned_phrase
                        )
                    else:
                        final_description = final_sentence_template.format(
                            class_name=class_name,
                            description=cleaned_phrase
                        )

                    if final_description:
                        final_description = final_description[0].upper() + final_description[1:]
                        if not final_description.endswith(('.', '!', '?')): final_description += '.'
                    print(f"      -> Final formatted description: {final_description}")
                    dimensional_library[class_name] = final_description
                else:
                    print("      -> Warning: Cleaned phrase is empty."); failed_count += 1
                    dimensional_library[class_name] = f"Description for {class_name} (dimension: {dimension}) could not be generated."
            else:
                print("      -> Warning: LLM failed to return a valid phrase."); failed_count += 1
                dimensional_library[class_name] = f"Description for {class_name} (dimension: {dimension}) could not be generated."
        except Exception as e:
            print(f"      -> Failed (exception: {e})"); failed_count += 1
            dimensional_library[class_name] = f"Description for {class_name} (dimension: {dimension}) could not be generated."
        time.sleep(3)

    print(f"\nDimension '{dimension}' description generation completed.")
    successful_generations = 0
    for desc in dimensional_library.values():
        if not desc.startswith("Description for") and not desc.startswith("No prompt generated") and not desc.startswith("No confusing class found"):
            successful_generations +=1

    print(f"Successfully generated {successful_generations} descriptions.")
    if dimension == "differential_comparison":
        print(f"Of which {fallback_to_visual_count} categories generated visual feature descriptions as fallback due to no confusion pairs.")
    print(f"Failed or empty/placeholder: {total_items - successful_generations}.")
    save_json(dimensional_library, output_path)
    return True

if __name__ == "__main__":
    parser_main = argparse.ArgumentParser(description="Generate Dimensional Descriptions for CLIP")
    parser_main.add_argument(
        '--dimensions',
        nargs='+',
        default=['all', ],
        choices=['visual_features', 'functional_use', 'contextual_scene', 'differential_comparison', 'finegrained_attribute', 'all'],
        help="List of description dimensions to generate, or 'all' to generate all dimensions."
    )
    parser_main.add_argument('--dataset_name', type=str, default=config.DATASET, help="Dataset name (used to load category names)")
    parser_main.add_argument('--target_words', type=int, default=25, help="Target word count for each description")
    parser_main.add_argument('--output_dir', type=str, default=' ', help="Subdirectory to save generated files")
    parser_main.add_argument('--analysis_file', type=str, default=config.ANALYSIS_RESULTS_NAMED_FILE, help="Analysis results file path (required for differential comparison)")

    script_args = parser_main.parse_args()

    print("=============================================")
    print("=== Starting script: Generate multi-dimensional descriptions (formatted output v2) ===")
    print("=============================================")

    try:
        prompt_file_path_template = config.PROMPT_FILE_TEMPLATE
        prompt_file_path = prompt_file_path_template.format(dataset=script_args.dataset_name)
    except AttributeError:
        print("Error: PROMPT_FILE_TEMPLATE not defined in config.py.")
        prompt_file_path = f"../clip/data/prompt_{script_args.dataset_name}.txt"
        print(f"Warning: Using fallback prompt file path: {prompt_file_path}")
    except KeyError:
        print(f"Error: Unable to format PROMPT_FILE_TEMPLATE, possibly incorrect dataset='{script_args.dataset_name}'.")
        exit()

    class_names_list = load_prompt(prompt_file_path)
    if not class_names_list:
        print(f"Error: Unable to load category names from {prompt_file_path}."); exit()

    analysis_data = None
    if 'differential_comparison' in script_args.dimensions or 'all' in script_args.dimensions:
        try:
            analysis_file_to_load = script_args.analysis_file
        except AttributeError:
            print("Error: ANALYSIS_RESULTS_NAMED_FILE not defined in config.py.")
            analysis_file_to_load = os.path.join(script_args.output_dir, "1_process_errors.json")
            print(f"Warning: Using fallback analysis file path: {analysis_file_to_load}")

        print(f"Loading analysis results for differential comparison dimension: {analysis_file_to_load}")
        analysis_data = utilsss_load_json(analysis_file_to_load)
        if not analysis_data:
            print(f"Warning: Unable to load analysis results file {analysis_file_to_load}, differential comparison dimension may not generate correctly or will fully fallback.")

    dimensions_to_process = []
    if 'all' in script_args.dimensions:
        dimensions_to_process = ['visual_features', 'functional_use', 'contextual_scene', 'differential_comparison', 'finegrained_attribute']
    else:
        dimensions_to_process = script_args.dimensions

    try:
        output_directory = script_args.output_dir
    except AttributeError:
        print("Error: OUTPUT_SUBDIR not defined in config.py.")
        output_directory = "maple9_analysis_results"
        print(f"Warning: Using fallback output directory: {output_directory}")

    os.makedirs(output_directory, exist_ok=True)

    overall_success = True
    start_run_time = time.time()

    for dim in dimensions_to_process:
        output_filename = f"library_{dim.replace(' ', '_')}.json"
        output_filepath = os.path.join(output_directory, output_filename)
        print(f"\n>>> Processing dimension: {dim} -> Output to: {output_filepath}")

        success_dim = generate_descriptions_for_dimension(
            class_names_list,
            dim,
            output_filepath,
            script_args.target_words,
            analysis_results=analysis_data
        )
        if not success_dim:
            overall_success = False

    end_run_time = time.time()

    if overall_success:
        print(f"\nAll dimensions processed. Total time: {end_run_time - start_run_time:.2f} seconds")
        print("\n=== Script execution completed ===")
    else:
        print("\n=== Script execution partially failed ===")
    print("=============================================")