import json
import os
import sys
from collections import Counter
import config
from utilsss import save_json, load_json, load_prompt

def analyze_errors(error_log_path: str,
                   top_n: int,
                   output_filepath: str,
                   dataset_name: str,
                   prompt_file_template: str) -> bool:
    print("--- Step 1: Analyzing error log ---")
    error_log = load_json(error_log_path)
    if error_log is None: return False
    prompt_file = prompt_file_template.format(dataset=dataset_name)
    name_class = load_prompt(prompt_file)
    if not name_class or not isinstance(name_class, list):
        print(f"Error: Failed to load valid category name list from {prompt_file}.")
        return False
    num_classes_loaded = len(name_class)
    print(f"Successfully loaded {num_classes_loaded} category names.")
    valid_entries = [e for e in error_log if isinstance(e, (list, tuple)) and len(e) == 3]
    if not valid_entries: print("No valid entries in error log."); return False
    confusion_pairs_indices = [(e[1], e[2]) for e in valid_entries if isinstance(e[1], int) and isinstance(e[2], int)]
    if not confusion_pairs_indices: print("Failed to extract valid confusion pair indices."); return False
    pair_counts = Counter(confusion_pairs_indices)
    most_common_pairs_indices = pair_counts.most_common(top_n)
    most_common_pairs_names = []
    print(f"\nMost common {len(most_common_pairs_indices)}/{top_n} confusion pairs (index -> name):")
    for (true_idx, pred_idx), count in most_common_pairs_indices:
        if 0 <= true_idx < num_classes_loaded and 0 <= pred_idx < num_classes_loaded:
            true_name = name_class[true_idx]
            pred_name = name_class[pred_idx]
            most_common_pairs_names.append(((true_name, pred_name), count))
            print(f"  ({true_idx}) {true_name} -> ({pred_idx}) {pred_name} : {count} times")
        else: print(f"  Warning: Skipping confusion pair ({true_idx}, {pred_idx}), index out of range [0, {num_classes_loaded-1}].")
    all_involved_indices = set(idx for e in valid_entries for idx in e[1:] if isinstance(idx, int))
    all_involved_names = sorted([name_class[idx] for idx in all_involved_indices if 0 <= idx < num_classes_loaded])
    if len(all_involved_names) != len([idx for idx in all_involved_indices if 0 <= idx < num_classes_loaded]):
        print("Warning: Some involved category indices could not be converted to names.")
    print(f"\nTotal {len(all_involved_names)} valid category names involved: {all_involved_names[:20]}...")
    analysis_results = {
        "all_involved_categories": all_involved_names,
        "top_confusion_pairs": most_common_pairs_names
    }
    save_json(analysis_results, output_filepath)
    return True

if __name__ == "__main__":
    print("=============================================")
    print("===  Error processing and analysis ===")
    print("=============================================")

    error_file = ' '
    analysis_file = config.ANALYSIS_RESULTS_NAMED_FILE
    dataset = config.DATASET
    prompt_template = config.PROMPT_FILE_TEMPLATE
    top_n = config.TOP_N_CONFUSING_PAIRS

    print(f"Error log input file: {error_file}")
    print(f"Analysis results output file: {analysis_file}")
    print(f"Dataset name: {dataset}")
    print(f"Prompt file template: {prompt_template}")
    print(f"Analyze top N confusion pairs: {top_n}")

    success = analyze_errors(error_file, top_n, analysis_file, dataset, prompt_template)

    if success: print("\n=== execution completed ===")
    else: print("\n=== execution failed ===")
    print("=============================================")