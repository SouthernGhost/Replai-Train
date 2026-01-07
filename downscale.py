import subprocess
import shutil
import sys
import argparse
import json
from pathlib import Path
import logging
from logging import getLogger

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = getLogger(__name__)

def get_video_dimensions(input_path: str, ffprobe_path: str = "ffprobe") -> tuple[int, int]:
    """Get video width and height using ffprobe."""
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        return width, height
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to get video dimensions: {e}")


def downscale_to_720p_hevc(
    input_path: str,
    output_path: str,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
):
    """
    Downscale a video to 720p height using NVIDIA GPU acceleration
    and encode with HEVC (H.265 NVENC).

    Requirements:
    - Windows/Linux
    - NVIDIA GPU with NVDEC/NVENC support
    - FFmpeg build with CUDA, NPP, and NVENC enabled
    """

    # Validate FFmpeg availability
    if not shutil.which(ffmpeg_path):
        raise FileNotFoundError(
            f"FFmpeg not found: {ffmpeg_path}. "
            "Ensure ffmpeg is in PATH or provide full path."
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Calculate dimensions
    try:
        orig_w, orig_h = get_video_dimensions(str(input_path), ffprobe_path)
    except RuntimeError as e:
        logger.warning(f"Warning: Could not determine dimensions ({e}). Defaulting to 1280x720.")
        target_w, target_h = 1280, 720
    else:
        target_h = 720
        aspect_ratio = orig_w / orig_h
        target_w = int(target_h * aspect_ratio)
        # Ensure width is even (required for 4:2:0)
        if target_w % 2 != 0:
            target_w += 1
    
    logger.info(f"Scaling from {orig_w}x{orig_h} to {target_w}x{target_h}")

    cmd = [
        ffmpeg_path,
        "-y",                                  # overwrite output
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", str(input_path),

        # GPU scaling with explicit dimensions using scale_cuda (npp might not be available)
        "-vf", f"scale_cuda={target_w}:{target_h}",

        # HEVC NVENC encoding
        "-c:v", "hevc_nvenc",
        "-preset", "p4",                      # balance quality / speed
        "-rc", "vbr",
        "-cq", "28",                          # quality control (lower = better, 28 is decent for 720p)
        "-profile:v", "main",

        # Audio passthrough
        "-c:a", "copy",

        str(output_path),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Output written to: {output_path}")

    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg execution failed", file=sys.stderr)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to 720p HEVC using NVIDIA Hardware Acceleration")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path to output video file")
    
    args = parser.parse_args()

    downscale_to_720p_hevc(
        input_path=args.input,
        output_path=args.output,
    )
