import cv2
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random

# =========================
# CONFIG
# =========================
VIDEO_ROOT = Path("datasets/videos/faceforensics++")
OUTPUT_IMAGE_ROOT = Path("datasets/images/faceforensics_frames")
NUM_FRAMES_PER_VIDEO = 5 # Number of frames to extract from each video
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15 # Remaining 0.15 for test

# Ensure deterministic split
random.seed(42)

def extract_frames_from_video(video_path, output_dir, label):
    video_name = video_path.stem
    frames_save_path = output_dir / label / video_name
    frames_save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES_PER_VIDEO, dtype=int)

    saved_frames = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(frames_save_path / frame_filename), frame)
            saved_frames += 1
    cap.release()
    return saved_frames > 0

def process_and_split_dataset():
    print(f"🚀 Starting FaceForensics++ Frame Extraction and Splitting (Target: {NUM_FRAMES_PER_VIDEO} frames/video)")

    # Collect all video paths and group by original identity (e.g., c23/videos/000.mp4 -> 000)
    # This helps in subject-independent splitting
    video_paths_by_identity = {}

    for label_dir in ["fake", "real"]:
        current_label_path = VIDEO_ROOT / label_dir
        if not current_label_path.exists():
            print(f"Warning: {current_label_path} not found. Skipping {label_dir} videos.")
            continue

        for video_path in current_label_path.rglob("*"):
            if video_path.suffix.lower() in VIDEO_EXTS:
                # Extract identity from video name (e.g., 000_001.mp4 -> 000)
                identity = video_path.stem.split('_')[0]
                if identity not in video_paths_by_identity:
                    video_paths_by_identity[identity] = []
                video_paths_by_identity[identity].append((video_path, label_dir))

    identities = list(video_paths_by_identity.keys())
    random.shuffle(identities)

    num_identities = len(identities)
    train_identities_idx = int(num_identities * TRAIN_RATIO)
    val_identities_idx = int(num_identities * (TRAIN_RATIO + VAL_RATIO))

    train_identities = identities[:train_identities_idx]
    val_identities = identities[train_identities_idx:val_identities_idx]
    test_identities = identities[val_identities_idx:]

    print(f"👥 Total identities: {num_identities}")
    print(f"Split: Train {len(train_identities)}, Val {len(val_identities)}, Test {len(test_identities)}")

    splits = {
        "train": train_identities,
        "val": val_identities,
        "test": test_identities
    }

    all_extraction_tasks = []

    for split_name, ids in splits.items():
        for identity in ids:
            for video_path, label in video_paths_by_identity[identity]:
                output_split_dir = OUTPUT_IMAGE_ROOT / split_name
                all_extraction_tasks.append((video_path, output_split_dir, label))

    print(f"📦 Found {len(all_extraction_tasks)} video processing tasks.")

    # Multi-process extraction
    # Using a smaller pool for I/O heavy tasks like video processing
    max_workers = os.cpu_count() // 2 if os.cpu_count() > 1 else 1
    max_workers = max(1, min(max_workers, 8)) # Cap at 8 workers for practicality

    successful_extractions = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in all_extraction_tasks:
            futures.append(executor.submit(extract_frames_from_video, *task))

        for future in tqdm(futures, total=len(futures), desc="Extracting frames"):
            if future.result():
                successful_extractions += 1

    print(f"🏁 Finished. Successfully processed {successful_extractions} videos.")
    print(f"📂 Frames saved to: {OUTPUT_IMAGE_ROOT}")

if __name__ == "__main__":
    process_and_split_dataset()
