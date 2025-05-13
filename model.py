import torch
import open_clip
import os
import pandas as pd
from PIL import Image

# Load CLIP model and preprocessor
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Truncate the CLIP model to the first N transformer layers
def truncate_model(model, num_layers=6):
    model.visual.transformer.resblocks = model.visual.transformer.resblocks[:num_layers]
    return model

# Load image and preprocess with adjustable resolution
def load_image(image_path, resolution=224):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution))  # Resize to the specified resolution
    return preprocess(image).unsqueeze(0).to(device)

# Tokenize text with inbuilt tokenizer
def encode_text(sentence):
    tokens = tokenizer(sentence)
    return tokens.to(device)

# Encode image and text
@torch.no_grad()
def compute_similarity(image_tensor, text_tensor, model):
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tensor)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()
    return similarity

# Decision logic
def classify(similarity_score, threshold=0.75):
    return similarity_score > threshold

# Full trial evaluation function
def evaluate_trial(row, model, resolution=224, threshold=0.75):
    base_path = "stimuli/"
    img_path_1 = os.path.join(base_path, row["image_path_1"])
    img_path_2 = os.path.join(base_path, row["image_path_2"])
    
    if not os.path.exists(img_path_1):
        raise FileNotFoundError(f"Original image not found: {img_path_1}")
    if not os.path.exists(img_path_2):
        raise FileNotFoundError(f"Modified image not found: {img_path_2}")
    
    sentence_1 = row["sentence_1"]
    sentence_2 = row["sentence_2"]
    gt_1 = row["is_true"]
    gt_2 = row["is_change"]

    img_tensor_1 = load_image(img_path_1, resolution=resolution)
    img_tensor_2 = load_image(img_path_2, resolution=resolution)

    text_tensor_1 = encode_text(sentence_1)
    text_tensor_2 = encode_text(sentence_2)

    sim_1 = compute_similarity(img_tensor_1, text_tensor_1, model)
    sim_2 = compute_similarity(img_tensor_2, text_tensor_2, model)

    pred_1 = classify(sim_1, threshold)
    pred_2 = classify(sim_2, threshold)

    return {
        "id": row["trial_id"],
        "quantifier": row["quantifier"],
        "truth_pred": pred_1,
        "truth_gt": bool(gt_1),
        "change_pred": pred_2,
        "change_gt": bool(gt_2),
        "sim_1": sim_1,
        "sim_2": sim_2
    }

# Load metadata
df = pd.read_csv("stimuli/metadata.csv")
df = df.rename(columns={"original_image": "image_path_1", "modified_image": "image_path_2"})

# Evaluate trials under different conditions
conditions = [
    {"name": "baseline", "model": model, "resolution": 224},
    {"name": "truncated", "model": truncate_model(model), "resolution": 224},
    {"name": "low_res", "model": model, "resolution": 112},
    {"name": "truncated_low_res", "model": truncate_model(model), "resolution": 112},
]

for condition in conditions:
    results = []
    for _, row in df.iterrows():
        try:
            trial_result = evaluate_trial(row, model=condition["model"], resolution=condition["resolution"])
            results.append(trial_result)
        except Exception as e:
            print(f"Error on trial {row['trial_id']} in {condition['name']} condition: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"clip_predictions_{condition['name']}.csv", index=False)