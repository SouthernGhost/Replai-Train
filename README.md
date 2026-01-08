# Project Scripts Documentation

This document explains how to use the utility scripts provided in this project for downloading datasets, training models, and processing videos.

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (with NVENC support for `downscale.py`)
- Required Python packages:
  ```bash
  pip install ultralytics roboflow requests torch
  ```

---

## 1. Download Dataset (`download_data.py`)

Downloads a dataset from Roboflow using settings defined in a JSON file.

### Usage
```bash
python download_data.py <settings_file> <output_directory>
```

### Arguments
- `<settings_file>`: Path to the JSON file containing Roboflow API credentials and project details.
- `<output_directory>`: Directory where the dataset will be downloaded and extracted.

### Example
```bash
python download_data.py settings/roboflow.json dataset/v3test/
```

### Settings File Format (`settings/roboflow.json`)
```json
{
    "api_key": "YOUR_API_KEY",
    "workspace": "YOUR_WORKSPACE",
    "project": "YOUR_PROJECT",
    "version": 3,
    "format": "yolo11"
}
```

---

## 2. Train Model (`train.py`)

Trains a YOLO11 model using the Ultralytics framework. It loads training configurations from a settings file.

### Usage
```bash
python train.py [--settings <path_to_settings>] [--export <format>]
```

### Arguments
- `--settings`: (Optional) Path to the training settings JSON file. Defaults to `settings.json`.
- `--export`: (Optional) Format to export the model to after training (e.g., `onnx`, `engine`).

### Example
```bash
# Train using default settings.json
python train.py

# Train using a specific settings file
python train.py --settings settings/train.json

# Train and then export to ONNX
python train.py --settings settings/train.json --export onnx
```

### Settings File Format (`settings/train.json`)
```json
{
    "train": {
        "model": "yolo11n.pt",
        "data": "dataset/v3test/data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "project": "runs/train",
        "name": "my_experiment"
    },
    "export": {
        "task": "detect",
        "imgsz": [640, 640],
        "batch": 1,
        "half": false,
        "nms": false,
        "dynamic": false,
        "int8": false,
        "data": "dataset/v3test/data.yaml"
    }
}
```

---

## 3. Downscale Video (`downscale.py`)

Downscales a video to 720p resolution using NVIDIA GPU acceleration (HEVC/H.265).

### Usage
```bash
python downscale.py <input_video> <output_video>
```

### Arguments
- `<input_video>`: Path to the source video file.
- `<output_video>`: Path where the downscaled video will be saved.

### Example
```bash
python downscale.py raw_footage.mp4 processed_720p.mp4
```

### Notes
- This script requires an NVIDIA GPU and an FFmpeg build with `hevc_nvenc` support.
- It attempts to maintain the aspect ratio while setting the height to 720 pixels.

---

## 4. Export Model (`export.py`)

Exports a trained YOLO model to various formats using the Ultralytics framework. It loads export configurations from a settings file.

### Usage
```bash
python export.py --settings <path_to_settings> [--weights <path_to_weights>]
```

### Arguments
- `--settings`: (Required) Path to the settings JSON file containing export configurations.
- `--weights`: (Optional) Path to the model weights file. Defaults to `models/best.pt`.

### Example
```bash
# Export using default weights
python export.py --settings settings/train.json

# Export with specific weights
python export.py --settings settings/train.json --weights runs/train/my_experiment/weights/best.pt
```

### Settings File Format
The export settings should be under the `"export"` key in the JSON file, as shown in the `settings/train.json` example above. If no `"export"` key exists, the entire file is used as export config.

Example export configuration:
```json
{
    "export": {
        "task": "detect",
        "imgsz": [640, 640],
        "batch": 1,
        "half": false,
        "nms": false,
        "dynamic": false,
        "int8": false,
        "data": "dataset/v3test/data.yaml"
    }
}
```
