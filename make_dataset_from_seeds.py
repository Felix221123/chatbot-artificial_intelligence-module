import os, random, uuid, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SEEDS_DIR = Path("seeds")
OUT_DIR = Path("finance_dataset")

CLASSES = ["bitcoin", "ethereum", "solana", "xrp", "litecoin"]


IMG_SIZE = (128, 128)

# Targets per split (per class)
TARGETS = {
    "train": 400,
    "val": 3,
    "test": 3
}

# How many REAL images to reserve for val/test (per class)
# (These will be copied from seeds without augmentation)
REAL_VAL = 3
REAL_TEST = 3

random.seed(42)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            (OUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

def make_background(size):
    """Random background: solid colour, gradient-ish, or noise."""
    w, h = size
    mode = random.choice(["solid", "noise", "light_grid"])
    if mode == "solid":
        colour = tuple(np.random.randint(0, 256, size=3).tolist())
        return Image.new("RGB", size, colour)

    if mode == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.2)))

    # light_grid
    bg = Image.new("RGB", size, (245, 245, 245))
    arr = np.array(bg)
    # add faint grid lines
    for x in range(0, w, random.choice([8, 12, 16])):
        arr[:, x:x+1] = np.clip(arr[:, x:x+1] - random.randint(5, 15), 0, 255)
    for y in range(0, h, random.choice([8, 12, 16])):
        arr[y:y+1, :] = np.clip(arr[y:y+1, :] - random.randint(5, 15), 0, 255)
    return Image.fromarray(arr, "RGB")

def augment_image(img: Image.Image, is_logo: bool):
    """Applies random transforms. Logos often have alpha & benefit from backgrounds."""
    # Convert to RGBA if it has alpha; otherwise RGB
    img = img.convert("RGBA") if img.mode != "RGBA" else img

    # Random scale
    scale = random.uniform(0.45, 0.95) if is_logo else random.uniform(0.70, 1.00)
    new_w = max(16, int(img.width * scale))
    new_h = max(16, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Random rotation
    angle = random.uniform(-25, 25) if is_logo else random.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    # Enhancements
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0)))

    # Optional slight posterize (makes some variety)
    if random.random() < 0.15:
        img = ImageOps.posterize(img.convert("RGB"), bits=random.choice([4, 5, 6])).convert("RGBA")

    return img

def composite_logo_on_bg(logo_rgba: Image.Image, bg_rgb: Image.Image):
    """Paste RGBA logo onto RGB background with random position."""
    bg = bg_rgb.copy()
    # random position
    max_x = max(0, bg.width - logo_rgba.width)
    max_y = max(0, bg.height - logo_rgba.height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    bg.paste(logo_rgba, (x, y), logo_rgba)
    return bg

def augment_and_save(seed_path: Path, out_path: Path, is_logo: bool):
    img = Image.open(seed_path)

    if is_logo:
        bg = make_background(IMG_SIZE)
        aug = augment_image(img, is_logo=True)
        composed = composite_logo_on_bg(aug, bg)
        final = composed.resize(IMG_SIZE, Image.Resampling.LANCZOS).convert("RGB")
    else:
        # For charts: keep structure, mild transforms, no random backgrounds
        img = img.convert("RGB")
        aug = augment_image(img, is_logo=False).convert("RGB")
        # random crop/pad to keep consistent size
        aug = ImageOps.fit(aug, IMG_SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        final = aug

    final.save(out_path, format="JPEG", quality=92)

def build_class(cname: str):
    seed_folder = SEEDS_DIR / cname
    seeds = list_images(seed_folder)
    if len(seeds) < (REAL_VAL + REAL_TEST + 1):
        print(f"[WARN] {cname}: only {len(seeds)} seed images. Try to collect more for better generalisation.")

    random.shuffle(seeds)

    # Reserve REAL images for val/test (no augmentation)
    val_real = seeds[:REAL_VAL]
    test_real = seeds[REAL_VAL:REAL_VAL + REAL_TEST]
    train_seeds = seeds[REAL_VAL + REAL_TEST:] if len(seeds) > (REAL_VAL + REAL_TEST) else seeds

    # Copy real val/test
    for p in val_real:
        dst = OUT_DIR / "val" / cname / f"{uuid.uuid4().hex}.jpg"
        Image.open(p).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS).save(dst, "JPEG", quality=92)

    for p in test_real:
        dst = OUT_DIR / "test" / cname / f"{uuid.uuid4().hex}.jpg"
        Image.open(p).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS).save(dst, "JPEG", quality=92)

    # For training: generate until target reached
    is_logo = cname in {"bitcoin", "ethereum", "solana", "xrp", "litecoin"}
    target_train = TARGETS["train"]

    out_train_dir = OUT_DIR / "train" / cname
    existing = len(list_images(out_train_dir))
    needed = max(0, target_train - existing)

    if not train_seeds:
        train_seeds = seeds  # fallback

    for i in range(needed):
        seed = random.choice(train_seeds)
        out_path = out_train_dir / f"{uuid.uuid4().hex}.jpg"
        augment_and_save(seed, out_path, is_logo=is_logo)

def main():
    ensure_dirs()
    for c in CLASSES:
        build_class(c)
    print("Done. Generated dataset at:", OUT_DIR)

if __name__ == "__main__":
    main()
