import subprocess
from pathlib import Path
import tempfile
import shutil
import argparse
from datetime import datetime
import sys
import logging
from logging import getLogger

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = getLogger(__name__)

def extract_frames_gpu(input_path: str, output_path: str, fps: float = 1.0):
    # Validate input path
    input_dir = Path(input_path)
    if not input_dir.exists():
        logger.error(f"Input directory '{input_path}' does not exist.")
        return

    # Create output directory with timestamp
    session = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = Path(output_path).joinpath(Path(session))
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        return

    video_files = sorted(input_dir.rglob("*.mp4"))

    if not video_files:
        logger.error(f"No MP4 files found in '{input_path}'.")
        return

    frame_counter = 1
    success_count = 0

    logger.info(f"Found {len(video_files)} video(s). Starting extraction at {fps} fps...")

    for video in video_files:
        logger.info(f"Processing: {video.name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tmp_pattern = tmpdir_path / "tmp_%06d.jpg"

            # Simplified FFmpeg command:
            # -hwaccel cuda: Use GPU for decoding
            # -vf fps={fps}: Extract specified frames per second
            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel", "cuda", 
                "-i", str(video),
                "-vf", f"fps={fps}",
                "-vsync", "0",
                "-q:v", "2", # High quality JPEG
                str(tmp_pattern)
            ]

            # Run ffmpeg and capture output for debugging if needed
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to extract frames from {video.name}. FFmpeg stderr:")
                logger.error(result.stderr)
                continue

            frames = sorted(tmpdir_path.glob("tmp_*.jpg"))
            if not frames:
                logger.warning(f"  Warning: No frames extracted from {video.name}")
                continue

            for frame in frames:
                target = output_dir / f"frame_{frame_counter:06d}.jpg"
                shutil.move(str(frame), target)
                frame_counter += 1
            
            success_count += 1

    logger.info(f"\nCompleted. Extracted {frame_counter - 1} frames from {success_count} videos.")
    logger.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos using GPU acceleration.")
    parser.add_argument("--input", type=str, required=True, help="Folder containing video files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for extracted frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Number of frames to extract per second (default: 1.0)")
    
    # Check if arguments are provided, otherwise print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    extract_frames_gpu(args.input, args.output, args.fps)
