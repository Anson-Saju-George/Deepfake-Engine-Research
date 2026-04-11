import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None

ROOT = Path("datasets")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

# 🔥 tune this (don’t go crazy)
MAX_WORKERS = 16


def collect_images():
    return [p for p in ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def analyze_image(path):

    try:
        img = Image.open(path)
        mode = img.mode

        if mode == "RGB":
            return "ok"

        elif mode == "P":
            return "palette"

        elif mode == "RGBA":
            return "rgba"

        else:
            return f"other:{mode}"

    except Exception:
        return "bad"


def fix_image(path):

    try:
        img = Image.open(path)

        if img.mode == "P":
            img = img.convert("RGBA")

        img = img.convert("RGB")

        img.save(path, quality=95)

        return True

    except Exception:
        return False


def main():

    images = collect_images()

    print(f"\n🔍 Total images found: {len(images)}\n")

    stats = Counter()

    # ===== MULTI-THREAD SCAN =====
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(analyze_image, img): img for img in images}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            result = future.result()
            stats[result] += 1

    print("\n=== DATASET ANALYSIS ===")
    for k, v in stats.items():
        print(f"{k:10} : {v}")

    print("\nSummary:")
    print(f"Needs fixing (non-RGB): {len(images) - stats['ok']}")
    print(f"Corrupted images      : {stats['bad']}")

    # ===== ASK USER =====
    choice = input("\nApply fixes? (y/n): ").strip().lower()

    if choice != "y":
        print("❌ No changes made.")
        return

    print("\n🛠 Fixing images...\n")

    fixed = 0
    deleted = 0

    # ===== MULTI-THREAD FIX =====
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {}

        for img_path in images:

            result = analyze_image(img_path)

            if result == "ok":
                continue

            if result == "bad":
                print(f"\n❌ Corrupted: {img_path}")
                c = input("Delete? (y/n): ").strip().lower()

                if c == "y":
                    try:
                        os.remove(img_path)
                        deleted += 1
                    except:
                        pass
                continue

            futures[executor.submit(fix_image, img_path)] = img_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fixing"):
            if future.result():
                fixed += 1

    print("\n=== FINAL SUMMARY ===")
    print(f"Fixed images : {fixed}")
    print(f"Deleted      : {deleted}")


if __name__ == "__main__":
    main()