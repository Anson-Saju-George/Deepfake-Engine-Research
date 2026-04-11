import os
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# =========================
# CONFIG
# =========================
VIDEO_ROOT = Path("videos")
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

def check_video(video_path):
    """Returns the path if corrupted, otherwise None"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_path
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return video_path
            
        cap.release()
        return None
    except Exception:
        return video_path

def main():
    print(f"🔍 Scanning for corrupted videos in: {VIDEO_ROOT}")
    
    # Gather all videos
    all_videos = []
    for video_path in VIDEO_ROOT.rglob("*"):
        if video_path.suffix.lower() in VIDEO_EXTS:
            all_videos.append(video_path)
            
    print(f"📦 Total videos found: {len(all_videos)}")
    
    # Multi-threaded scan
    corrupted = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = list(tqdm(executor.map(check_video, all_videos), total=len(all_videos)))
        
    corrupted = [v for v in results if v is not None]
    
    if not corrupted:
        print("✅ No corrupted videos found!")
        return

    print(f"\n🚨 Found {len(corrupted)} corrupted videos:")
    for v in corrupted[:10]: # Show first 10
        print(f"  - {v}")
    if len(corrupted) > 10:
        print(f"  ... and {len(corrupted) - 10} more.")

    # Ask for user confirmation
    choice = input(f"\n❓ Do you want to DELETE these {len(corrupted)} videos? (y/n): ").lower()
    
    if choice == 'y':
        print("🗑️ Deleting...")
        for v in corrupted:
            try:
                os.remove(v)
            except Exception as e:
                print(f"❌ Failed to delete {v}: {e}")
        print("🏁 Cleanup complete!")
    else:
        print("✋ Deletion cancelled.")

if __name__ == "__main__":
    main()
