import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

VIDEO_ROOT = Path("videos")
OUTPUT_ROOT = Path("frames")

NUM_FRAMES = 16        # frames per video
MAX_WORKERS = 8        # adjust based on CPU


def extract_frames(video_path: Path):
    try:
        rel_path = video_path.relative_to(VIDEO_ROOT)
        out_dir = OUTPUT_ROOT / rel_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        # skip if already done
        existing = list(out_dir.glob("*.jpg"))
        if len(existing) >= NUM_FRAMES:
            return ("skip", video_path.name)

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            return ("fail", video_path.name)

        # sample evenly
        indices = np.linspace(0, total - 1, NUM_FRAMES).astype(int)

        saved = 0
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            out_path = out_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        cap.release()

        if saved == 0:
            return ("fail", video_path.name)

        return ("success", video_path.name)

    except Exception as e:
        return ("error", f"{video_path.name} | {e}")


def main():
    videos = list(VIDEO_ROOT.rglob("*.mp4"))
    total = len(videos)

    print(f"\n🎬 Found {total} videos\n")

    success, failed, skipped = 0, 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_frames, v) for v in videos]

        with tqdm(total=total, desc="📸 Extracting frames", ncols=100) as pbar:
            for future in as_completed(futures):
                status, _ = future.result()

                if status == "success":
                    success += 1
                elif status == "fail":
                    failed += 1
                else:
                    skipped += 1

                pbar.set_postfix({
                    "✅": success,
                    "❌": failed,
                    "⏭️": skipped
                })

                pbar.update(1)

    print("\n📊 FINAL SUMMARY")
    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")


if __name__ == "__main__":
    main()