import cv2
import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


VIDEO_ROOT = Path("videos")
FRAME_ROOT = Path("video-frames")

FRAME_SAMPLE_RATE = 10
MAX_WORKERS = min(14, multiprocessing.cpu_count())

VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv"]


def detect_label(path: Path):

    name = str(path).lower()

    if "real" in name or "youtube-real" in name or "celeb-real" in name:
        return "real"

    if "fake" in name or "synthesis" in name or "deepfake" in name or "ai" in name:
        return "fake"

    return None


def extract_frames(video_path, save_dir):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return "CORRUPT_OPEN"

    frame_idx = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % FRAME_SAMPLE_RATE == 0:

            filename = f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(save_dir / filename), frame)

            saved += 1

        frame_idx += 1

    cap.release()

    if saved == 0:
        return "NO_FRAMES"

    return "OK"


def process_video(video):

    label = detect_label(video.parent)

    if label is None:
        return ("SKIPPED", str(video))

    dataset_name = video.relative_to(VIDEO_ROOT).parts[0]
    video_name = video.stem

    save_dir = FRAME_ROOT / label / dataset_name / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(save_dir.glob("*.jpg"))

    if len(existing_frames) > 5:
        return ("SKIPPED_EXIST", str(video))

    if len(existing_frames) > 0:
        shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    result = extract_frames(video, save_dir)

    return (result, str(video))


def collect_videos():

    videos = []

    for dataset in VIDEO_ROOT.iterdir():

        if not dataset.is_dir():
            continue

        for ext in VIDEO_EXTENSIONS:
            videos.extend(dataset.rglob(ext))

    return videos


# 🔥 robust delete with retries
def safe_delete(file_path, retries=5):

    for i in range(retries):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            time.sleep(0.5)

    print(f"FAILED DELETE: {file_path}")
    return False


def handle_errors(results):

    print("\n=== ERROR HANDLING ===\n")

    corrupt = [v for r, v in results if r in ["CORRUPT_OPEN", "NO_FRAMES"]]

    print(f"Corrupted / failed videos: {len(corrupt)}")

    if not corrupt:
        return

    for vid in corrupt[:20]:
        print(vid)

    choice = input("\nDelete ALL corrupted videos? (y/n): ").strip().lower()

    if choice != "y":
        print("Skipped deletion")
        return

    print("\nDeleting corrupted videos...\n")

    deleted = 0
    failed = 0

    for vid in tqdm(corrupt):

        if safe_delete(vid):
            deleted += 1
        else:
            failed += 1

    print("\nDeletion Summary:")
    print(f"Deleted: {deleted}")
    print(f"Failed : {failed}")


def verify_summary(results):

    total = len(results)
    success = sum(1 for r, _ in results if r == "OK")
    skipped = sum(1 for r, _ in results if "SKIPPED" in r)
    failed = sum(1 for r, _ in results if r not in ["OK"] and "SKIPPED" not in r)

    print("\n=== SUMMARY ===")
    print(f"Total videos     : {total}")
    print(f"Processed        : {success}")
    print(f"Skipped          : {skipped}")
    print(f"Failed           : {failed}")


def main():

    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    videos = collect_videos()

    print("\nTotal videos found:", len(videos))
    print("Using workers:", MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_video, videos), total=len(videos)))

    verify_summary(results)

    handle_errors(results)

    print("\nFrame extraction completed")


if __name__ == "__main__":
    main()