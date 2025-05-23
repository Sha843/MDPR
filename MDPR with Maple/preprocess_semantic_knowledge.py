import torch
import json
import os
import warnings
from clip.vanilla_clip import clip as clip

SEMANTIC_PROMPTS_JSON_PATH = ".json"
CLASSNAMES_TXT_PATH = "clip/data/prompt_cifar100.txt"
OUTPUT_KNOWLEDGE_PT_PATH = ".pt"
CLIP_MODEL_FOR_ENCODING = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_NUM_DESCRIPTIONS_PER_CLASS = 5
MIN_EXPECTED_DESCRIPTIONS_IF_NOT_FIXED = 3

def load_classnames(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Classnames file not found: {filepath}")
    import ast
    with open(filepath, 'r') as file:
        content = file.read()
    classnames = ast.literal_eval(content)
    print(f"Loaded {len(classnames)} classnames from {filepath}")
    return classnames


def main():
    print(f"Using device: {DEVICE} for preprocessing.")
    classnames_ordered = load_classnames(CLASSNAMES_TXT_PATH)
    num_classes = len(classnames_ordered)

    with open(SEMANTIC_PROMPTS_JSON_PATH, 'r', encoding='utf-8') as f:
        semantic_data_raw = json.load(f)

    all_descriptions_for_all_classes = []
    num_descriptions_per_class_to_use = FIXED_NUM_DESCRIPTIONS_PER_CLASS
    print(f"[CONFIG] Targeting {num_descriptions_per_class_to_use} descriptions per class.")

    for i, cls_name in enumerate(classnames_ordered):
        parsed_descriptions = []
        if cls_name not in semantic_data_raw:
            warnings.warn(
                f"Class '{cls_name}' not found in JSON. Using {num_descriptions_per_class_to_use} default placeholders.")
            parsed_descriptions = [f"default_description_{j + 1}_for_{cls_name.replace('_', ' ')}" for j in
                                   range(num_descriptions_per_class_to_use)]
        else:
            long_description_string = semantic_data_raw[cls_name]
            if not isinstance(long_description_string, str):
                warnings.warn(
                    f"Expected a string for class '{cls_name}', got {type(long_description_string)}. Using placeholders.")
                parsed_descriptions = [f"default_description_{j + 1}_for_{cls_name.replace('_', ' ')}" for j in
                                       range(num_descriptions_per_class_to_use)]
            else:
                print(f"[DEBUG] Processing class '{cls_name}'. Raw string length: {len(long_description_string)}")
                descriptions_from_split = [desc.strip() for desc in long_description_string.split(';') if desc.strip()]
                print(f"[DEBUG] Class '{cls_name}', descriptions after split: {len(descriptions_from_split)} found.")

                if not descriptions_from_split:
                    warnings.warn(f"Zero descriptions found for '{cls_name}' after splitting. Using placeholders.")
                    parsed_descriptions = [f"default_description_{j + 1}_for_{cls_name.replace('_', ' ')}" for j in
                                           range(num_descriptions_per_class_to_use)]
                else:
                    if len(descriptions_from_split) > num_descriptions_per_class_to_use:
                        warnings.warn(
                            f"Class '{cls_name}' has {len(descriptions_from_split)} descriptions. Truncating to {num_descriptions_per_class_to_use}.")
                        parsed_descriptions = descriptions_from_split[:num_descriptions_per_class_to_use]
                    elif len(descriptions_from_split) < num_descriptions_per_class_to_use:
                        warnings.warn(
                            f"Class '{cls_name}' has only {len(descriptions_from_split)} descriptions. Padding to {num_descriptions_per_class_to_use}.")
                        padding_needed = num_descriptions_per_class_to_use - len(descriptions_from_split)
                        if descriptions_from_split:
                            padding_strings = [descriptions_from_split[-1]] * padding_needed
                        else:
                            padding_strings = [f"padding_placeholder_{k}_for_{cls_name.replace('_', ' ')}" for k in
                                               range(padding_needed)]
                        parsed_descriptions = descriptions_from_split + padding_strings
                    else:
                        parsed_descriptions = descriptions_from_split

        if len(parsed_descriptions) != num_descriptions_per_class_to_use:
            raise ValueError(
                f"Final description list for '{cls_name}' does not have target length {num_descriptions_per_class_to_use}. Processed: {parsed_descriptions}")

        all_descriptions_for_all_classes.append(parsed_descriptions)
    num_phrases_per_class_final = num_descriptions_per_class_to_use
    print(
        f"[INFO] All classes processed. num_phrases_per_class (descriptions per class) is set to {num_phrases_per_class_final}.")

    flat_list_of_all_items_to_encode = [item for sublist in all_descriptions_for_all_classes for item in sublist]

    if not flat_list_of_all_items_to_encode:
        raise ValueError("No items to encode after processing all classes.")

    print(f"Loading STANDARD CLIP model: {CLIP_MODEL_FOR_ENCODING} for semantic description encoding...")
    std_clip_model, _ = clip.load(CLIP_MODEL_FOR_ENCODING, device=DEVICE)
    std_clip_model.eval()
    dtype = std_clip_model.dtype

    with torch.no_grad():
        tokenized_items = clip.tokenize(flat_list_of_all_items_to_encode).to(DEVICE)
        encoded_all_items = std_clip_model.encode_text(tokenized_items).type(dtype)

    embed_dim = encoded_all_items.shape[-1]
    encoded_prompt_sem_tensor = encoded_all_items.view(num_classes, num_phrases_per_class_final, embed_dim)
    prompt_sem_cls_features_tensor = torch.mean(encoded_prompt_sem_tensor, dim=1)

    knowledge_to_save = {
        "encoded_prompt_sem": encoded_prompt_sem_tensor.cpu(),
        "prompt_sem_cls_features": prompt_sem_cls_features_tensor.cpu(),
        "num_phrases_per_class": num_phrases_per_class_final,
        "classnames_ordered": classnames_ordered
    }
    torch.save(knowledge_to_save, OUTPUT_KNOWLEDGE_PT_PATH)
    print(f"Pre-encoded semantic knowledge saved to: {OUTPUT_KNOWLEDGE_PT_PATH}")
    print(f"  encoded_prompt_sem shape: {encoded_prompt_sem_tensor.shape}")
    print(f"  prompt_sem_cls_features shape: {prompt_sem_cls_features_tensor.shape}")
    print(f"  num_phrases_per_class in saved file: {num_phrases_per_class_final}")


if __name__ == "__main__":
    main()