import json
import torch
import os
import sys
from clip.vanilla_clip import clip as clip

SEMANTIC_PROMPTS_JSON_PATH = ".json"
CLASSNAMES_TXT_PATH = "clip/data/prompt_cifar100.txt"
OUTPUT_MATRIX_PRETRAIN_PATH = ".pt"
CLIP_MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_NUM_PHRASES_IF_MISSING = 10

def load_prompt(filename):
    import ast
    with open(filename, 'r') as file:
        content = file.read()
    return ast.literal_eval(content)
def load_enhanced_prompts(json_path, class_names):

    with open(json_path, 'r', encoding='utf-8') as f:
        semantic_dict = json.load(f)

    prompts = []
    for cls in class_names:
        if cls in semantic_dict:
            prompts.append(semantic_dict[cls])
        else:
            prompts.append(f"A photo of a {cls}")
    return prompts

def load_semantic_prompts(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Semantic prompts JSON file not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("Semantic prompts data should be a dictionary.")
    print(f"Loaded semantic prompts for {len(data)} classes from {filepath}")
    return data


name_class = load_prompt(f'./clip/data/prompt_cifar100.txt')

def parse_and_align_semantic_data(classnames_ordered, semantic_data_raw):
    all_phrases_for_all_classes = []
    num_phrases_per_class = 0
    print("Parsing and aligning semantic data...")

    for i, cls_name in enumerate(classnames_ordered):
        phrases_for_this_class = []
        processed_successfully = False

        if cls_name not in semantic_data_raw:
            print(f"[WARN] Class '{cls_name}' not found in semantic_prompts_data.")
            if DEFAULT_NUM_PHRASES_IF_MISSING is not None and DEFAULT_NUM_PHRASES_IF_MISSING > 0:
                num_to_generate = DEFAULT_NUM_PHRASES_IF_MISSING
                if num_phrases_per_class > 0:
                    num_to_generate = num_phrases_per_class

                phrases_for_this_class = [f"feature {j+1} of {cls_name}" for j in range(num_to_generate)]
                print(f"         Generated {len(phrases_for_this_class)} placeholder phrases for '{cls_name}'.")
                if phrases_for_this_class:
                    processed_successfully = True

                    if num_phrases_per_class == 0:
                        num_phrases_per_class = len(phrases_for_this_class)
        else:

            long_phrase_string = semantic_data_raw[cls_name]

            if not isinstance(long_phrase_string, str) or not long_phrase_string.strip():
                 print(f"[WARN] Semantic data for class '{cls_name}' is not a non-empty string (found: '{long_phrase_string}'). Treating as zero phrases for now.")
                 phrases_for_this_class = []
            else:
                phrases_for_this_class = [phrase.strip() for phrase in long_phrase_string.split(';') if phrase.strip()]
                if not phrases_for_this_class:
                    raise ValueError(
                        f"Class '{cls_name}' existed in JSON but resulted in zero phrases after parsing string '{long_phrase_string}'. "
                        "Check JSON content, ensure it's not just separators or whitespace, and verify the separator is ';'."
                    )
                else:
                     processed_successfully = True
        if processed_successfully:
            current_phrase_count = len(phrases_for_this_class)

            if num_phrases_per_class == 0:
                 if current_phrase_count == 0:
                      raise ValueError(f"Logic error: Processed class '{cls_name}' successfully but ended up with zero phrases before setting num_phrases_per_class.")
                 num_phrases_per_class = current_phrase_count
                 print(f"Determined num_phrases_per_class = {num_phrases_per_class} from class '{cls_name}'.")
            elif current_phrase_count != num_phrases_per_class:
                 original_data_repr = semantic_data_raw.get(cls_name, "N/A (Missing from JSON)")
                 raise ValueError(
                    f"Mismatch in number of phrases for class '{cls_name}'. "
                    f"Expected {num_phrases_per_class}, but got {current_phrase_count}. "
                    f"Data source: '{original_data_repr}'. "
                    "All classes must yield the same number of semantic phrases after parsing (or placeholder generation)."
                 )
        all_phrases_for_all_classes.append(phrases_for_this_class)

    if not all_phrases_for_all_classes:
        raise ValueError("Failed to parse or generate phrases for any class. Check input files.")
    if num_phrases_per_class == 0:
         raise ValueError("Could not determine the number of phrases per class. Ensure JSON has valid data for at least one class or DEFAULT_NUM_PHRASES_IF_MISSING is set.")

    print(f"Successfully parsed/aligned semantic data for {len(all_phrases_for_all_classes)} classes, each expected to have {num_phrases_per_class} phrases.")
    for i, phrases in enumerate(all_phrases_for_all_classes):
        if len(phrases) != num_phrases_per_class:
             raise ValueError(f"Internal inconsistency: Class '{classnames_ordered[i]}' ended up with {len(phrases)} phrases, but expected {num_phrases_per_class}. This might happen if a class was missing and no placeholders were generated.")


    return all_phrases_for_all_classes, num_phrases_per_class
def main():
    print(f"Using device: {DEVICE}")
    classnames_ordered = load_prompt(CLASSNAMES_TXT_PATH)
    num_classes = len(classnames_ordered)
    semantic_data_raw = load_semantic_prompts(SEMANTIC_PROMPTS_JSON_PATH)
    all_phrases_for_all_classes, num_phrases_per_class = parse_and_align_semantic_data(
        classnames_ordered, semantic_data_raw
    )
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()
    print("CLIP model loaded.")
    print("Calculating matrix_pretrain using text similarity...")
    matrix_pretrain = torch.zeros(num_classes, num_phrases_per_class, device=DEVICE, dtype=clip_model.dtype)

    with torch.no_grad():
        for i in range(num_classes):
            class_name_text =classnames_ordered[i]
            tokenized_class_name = clip.tokenize([class_name_text]).to(DEVICE)
            class_embedding = clip_model.encode_text(tokenized_class_name)
            class_embedding_norm = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
            phrases_for_this_class = all_phrases_for_all_classes[i]
            tokenized_phrases = clip.tokenize(phrases_for_this_class).to(DEVICE)
            phrase_embeddings = clip_model.encode_text(tokenized_phrases)
            phrase_embeddings_norm = phrase_embeddings / phrase_embeddings.norm(dim=-1, keepdim=True)
            similarities = (class_embedding_norm @ phrase_embeddings_norm.T).squeeze(0)
            matrix_pretrain[i, :] = similarities
            if (i + 1) % 1 == 0 or (i + 1) == num_classes:
                print(f"  Processed class {i+1}/{num_classes}: {classnames_ordered[i]}")
    print("matrix_pretrain calculation complete.")
    print(f"Shape of matrix_pretrain: {matrix_pretrain.shape}")
    torch.save(matrix_pretrain.cpu(), OUTPUT_MATRIX_PRETRAIN_PATH)
    print(f"matrix_pretrain saved to {OUTPUT_MATRIX_PRETRAIN_PATH}")

if __name__ == "__main__":
    main()