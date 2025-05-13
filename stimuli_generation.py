import numpy as np
from PIL import Image, ImageDraw
import os
import csv
import colorspacious as cs
import random

# === CONFIGURATION === #
IMG_SIZE = 224
CIRCLE_RADIUS = 25
N_CIRCLES = 3
BG_COLOR = (128, 128, 128)
HUE_SD = 17
OUTPUT_DIR = "stimuli"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color category ranges (from Bae et al. 2015, approximate center values)
COLOR_CATEGORIES = {
    "green": (120, 150),   # hue range for green
    "blue": (210, 240),    # hue range for blue
    "orange": (30, 60),    # hue range for orange
}


def hue_to_rgb_cielab(hue_deg):
    """Convert a hue angle (0–360) on the CIELAB wheel to RGB."""
    L = 70
    C = 40
    a = C * np.cos(np.deg2rad(hue_deg))
    b = C * np.sin(np.deg2rad(hue_deg))
    lab = np.array([L, a, b])
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return tuple(rgb)


def sample_valid_hues(color_category):
    """Sample three hues from a target category."""
    start, end = COLOR_CATEGORIES[color_category]
    return np.random.uniform(start, end, size=3) % 360


def sample_invalid_hue(color_category):
    """Sample one hue outside the given category."""
    while True:
        hue = np.random.uniform(0, 360)
        start, end = COLOR_CATEGORIES[color_category]
        if not (start <= hue <= end):
            return hue


def apply_hue_change(hue):
    """Apply Gaussian-distributed change to hue (mod 360)."""
    new_hue = np.random.normal(loc=hue, scale=HUE_SD) % 360
    return new_hue


def generate_positions():
    """Generate non-overlapping circle positions."""
    margin = 2 * CIRCLE_RADIUS
    positions = []
    while len(positions) < N_CIRCLES:
        x = np.random.randint(margin, IMG_SIZE - margin)
        y = np.random.randint(margin, IMG_SIZE - margin)
        if all(np.hypot(x - px, y - py) > 2 * CIRCLE_RADIUS for px, py in positions):
            positions.append((x, y))
    return positions


def generate_trial(trial_id, quantifier, color_label, is_true, is_change):
    """Generate a full stimulus trial: image pair + metadata entry."""
    # Color hues for original image
    if is_true:
        hues = sample_valid_hues(color_label)
    else:
        hues = sample_valid_hues(color_label)
        hues[0] = sample_invalid_hue(color_label)

    colors = [hue_to_rgb_cielab(h) for h in hues]
    positions = generate_positions()

    # Original image
    original = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(original)
    for pos, color in zip(positions, colors):
        draw.ellipse([pos[0] - CIRCLE_RADIUS, pos[1] - CIRCLE_RADIUS,
                      pos[0] + CIRCLE_RADIUS, pos[1] + CIRCLE_RADIUS], fill=color)

    # Modified image
    if is_change:
        idx = np.random.randint(0, N_CIRCLES)
        changed_hue = apply_hue_change(hues[idx])
        changed_color = hue_to_rgb_cielab(changed_hue)
        mod_colors = colors.copy()
        mod_colors[idx] = changed_color
    else:
        mod_colors = colors.copy()

    modified = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(modified)
    for pos, color in zip(positions, mod_colors):
        draw.ellipse([pos[0] - CIRCLE_RADIUS, pos[1] - CIRCLE_RADIUS,
                      pos[0] + CIRCLE_RADIUS, pos[1] + CIRCLE_RADIUS], fill=color)

    # File paths
    base_path = f"trial_{trial_id}_original.png"
    mod_path = f"trial_{trial_id}_{'changed' if is_change else 'same'}.png"
    original.save(os.path.join(OUTPUT_DIR, base_path))
    modified.save(os.path.join(OUTPUT_DIR, mod_path))

    return {
        "trial_id": trial_id,
        "quantifier": quantifier,
        "color_label": color_label,
        "is_true": int(is_true),
        "is_change": int(is_change),
        "original_image": base_path,
        "modified_image": mod_path,
        "sentence_1": f"{quantifier.capitalize()} circle is {color_label}.",
        "sentence_2": "One circle changed its color."
    }


def generate_dataset(n_trials=2000):
    quantifiers = ["each", "every"]
    colors = list(COLOR_CATEGORIES.keys())
    metadata = []

    for i in range(n_trials):
        quant = random.choice(quantifiers)
        color = random.choice(colors)
        is_true = bool(i % 2)
        is_change = bool((i // 2) % 2)
        metadata.append(generate_trial(i, quant, color, is_true, is_change))

    # Write metadata CSV
    keys = metadata[0].keys()
    with open(METADATA_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated {n_trials} trials with metadata saved to {METADATA_FILE}")


# === Run it === #
if __name__ == "__main__":
    generate_dataset(n_trials=2000)

df = pd.read_csv("metadata.csv")
df = df.rename(columns={
    "original_image": "image_path_1",
    "modified_image": "image_path_2"
})