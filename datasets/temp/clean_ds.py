import cv2
import os
import time
from pathlib import Path
from tqdm import tqdm


VIDEO_ROOT = Path("videos")

VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

# 🔥 set to >1 later if you want parallel
NUM_WORKERS = 4


def collect_videos():

    videos = []

    for dataset in VIDEO_ROOT.iterdir():

        if not dataset.is_dir():
            continue

        for ext in VIDEO_EXTENSIONS:
            videos.extend(dataset.rglob(ext))

    return videos


def is_video_valid(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return False, "OPEN_FAIL"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Try reading a few frames
    success_reads = 0

    for i in range(3):

        ret, frame = cap.read()

        if ret:
            success_reads += 1
        else:
            break

    cap.release()

    if total_frames <= 0:
        return False, "NO_METADATA"

    if success_reads == 0:
        return False, "NO_READ"

    return True, "OK"


def safe_delete(path, retries=5):

    for _ in range(retries):
        try:
            os.remove(path)
            return True
        except Exception:
            time.sleep(0.5)

    return False


def main():

    print("\n🔍 Scanning videos...\n")

    videos = collect_videos()

    print(f"Total videos found: {len(videos)}\n")

    bad_videos = []

    for vid in tqdm(videos):

        valid, reason = is_video_valid(vid)

        if not valid:
            bad_videos.append((vid, reason))

    print("\n=== SCAN COMPLETE ===")
    print(f"Bad videos found: {len(bad_videos)}\n")

    if not bad_videos:
        print("All videos are valid ✅")
        return

    print("Sample bad videos:")
    for vid, reason in bad_videos[:20]:
        print(f"{reason} -> {vid}")

    choice = input("\nDelete ALL bad videos? (y/n): ").strip().lower()

    if choice != "y":
        print("Skipped deletion")
        return

    print("\n🗑 Deleting bad videos...\n")

    deleted = 0
    failed = 0

    for vid, _ in tqdm(bad_videos):

        if safe_delete(vid):
            deleted += 1
        else:
            failed += 1
            print(f"FAILED DELETE: {vid}")

    print("\n=== DELETE SUMMARY ===")
    print(f"Deleted: {deleted}")
    print(f"Failed : {failed}")


if __name__ == "__main__":
    main()