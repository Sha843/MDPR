import json
import os
from collections import OrderedDict

input_files = [
    "library_visual_features.json",
    "library_functional_use.json",
    "library_finegrained_attribute.json",
    "library_differential_comparison.json",
    "library_contextual_scene.json"
]
output_file = "merged_library.json"

separator = "; "

merged_data = OrderedDict()
num_files = len(input_files)

print(f"Starting to merge {num_files} JSON files and clean internal punctuation...")
for index, filename in enumerate(input_files):
    print(f"Processing file {index + 1}/{num_files}: {filename}...")
    try:
        if not os.path.exists(filename):
            print(f"  Warning: File '{filename}' not found. Skipping this file.")
            for cls in merged_data:
                 if len(merged_data[cls]) == index:
                    merged_data[cls].append("")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                current_data = json.load(f)
                if not isinstance(current_data, dict):
                    print(f"  Error: Top-level structure of file '{filename}' is not a JSON object (dictionary). Skipping.")
                    for cls in merged_data:
                        if len(merged_data[cls]) == index:
                            merged_data[cls].append("")
                    continue
            except json.JSONDecodeError as e:
                print(f"  Error: Failed to parse JSON file '{filename}'. Please check file format. Error: {e}")
                for cls in merged_data:
                    if len(merged_data[cls]) == index:
                        merged_data[cls].append("")
                continue
        processed_keys_in_current_file = set()
        for cls, sentence in current_data.items():
            processed_keys_in_current_file.add(cls)
            if not isinstance(sentence, str):
                print(f"  Warning: In file '{filename}', the value for key '{cls}' is not a string. Using empty string instead.")
                cleaned_sentence = ""
            else:
                cleaned_sentence = sentence.replace(';', ',').replace('；', ',')
            if cls not in merged_data:
                merged_data[cls] = [""] * index
                merged_data[cls].append(cleaned_sentence)
            else:
                if len(merged_data[cls]) == index:
                     merged_data[cls].append(cleaned_sentence)
                elif len(merged_data[cls]) == index + 1:
                    if not merged_data[cls][index]:
                         merged_data[cls][index] = cleaned_sentence
                    else:
                         pass
                else:
                     merged_data[cls].extend([""] * (index - len(merged_data[cls])))
                     merged_data[cls].append(cleaned_sentence)
        all_existing_keys = set(merged_data.keys())
        missing_keys_in_current = all_existing_keys - processed_keys_in_current_file
        for cls in missing_keys_in_current:
             if len(merged_data[cls]) == index:
                merged_data[cls].append("")

    except FileNotFoundError:
        print(f"  Error: File '{filename}' not found (repeated check). Skipping.")
        for cls in merged_data:
            if len(merged_data[cls]) == index:
                merged_data[cls].append("")
    except IOError as e:
        print(f"  Error: Unable to read file '{filename}'. Error: {e}")
        for cls in merged_data:
            if len(merged_data[cls]) == index:
                merged_data[cls].append("")
    except Exception as e:
        print(f"  Unknown error occurred while processing file '{filename}': {e}")
        for cls in merged_data:
            if len(merged_data[cls]) == index:
                merged_data[cls].append("")

print("All files processed. Combining sentences...")
final_output = OrderedDict()
for cls, sentence_list in merged_data.items():
    if len(sentence_list) < num_files:
        print(f"  Warning: Sentence list for category '{cls}' is shorter than {num_files}, padding with empty strings. Final length: {len(sentence_list)}")
        sentence_list.extend([""] * (num_files - len(sentence_list)))
    elif len(sentence_list) > num_files:
         print(f"  Warning: Sentence list for category '{cls}' exceeds {num_files} ({len(sentence_list)}), possible logic error. Truncating.")
         sentence_list = sentence_list[:num_files]
    combined_sentence = separator.join(sentence_list)
    final_output[cls] = combined_sentence

print(f"Merging complete. Preparing to write to file '{output_file}'...")

try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"Successfully wrote merged results to '{output_file}'.")
except IOError as e:
    print(f"Error: Unable to write to output file '{output_file}'. Error: {e}")
except Exception as e:
    print(f"Unknown error occurred while writing to output file: {e}")