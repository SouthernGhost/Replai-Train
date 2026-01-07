import json
import os
import argparse
import logging
from datetime import datetime
import torch
import gc
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define default settings
DEFAULT_SETTINGS = {
    "model": "yolo11n.pt",
    "data": "data.yaml",
    "epochs": 100,
    "val": True,
    "plots": True,
    "seed": 42,
    "patience": 10,
    "imgsz": 640,
    "batch": 16,
    "freeze": 10,  # Freezing the first 10 layers (backbone)
    "save": True,
    "cache": "disk",
    "device": "0",
    "pretrained": True,
    "project": "runs/train",
    "name": "yolo11n_finetune",
}

def load_or_create_settings(settings_path):
    """
    Loads settings from the specified path.
    If the file does not exist, creates it with default values.
    """
    if not os.path.exists(settings_path):
        logger.warning(f"'{settings_path}' not found. Creating with default settings...")
        
        # Ensure directory exists if path contains directories
        directory = os.path.dirname(settings_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(settings_path, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS.copy()
    else:
        logger.info(f"Loading settings from '{settings_path}'...")
        try:
            with open(settings_path, "r") as f:
                loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist if partial settings file
                settings = DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
        except json.JSONDecodeError:
            logger.error(f"Error decoding '{settings_path}'. Using default settings.")
            return DEFAULT_SETTINGS.copy()

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 model with custom settings.")
    parser.add_argument("--settings", type=str, default="settings.json", help="Path to the settings JSON file.")
    parser.add_argument("--export", type=str, choices=['onnx', 'engine'], default=None, help="Export format (onnx or engine).")
    args = parser.parse_args()

    # 1. Load Settings
    train_settings = load_or_create_settings(args.settings)["train"]
    export_settings = load_or_create_settings(args.settings)["export"]
    
    # Override settings with command line arguments if provided
    # 2. Initialize Model
    # Use a pretrained yolo11n model
    logger.info(f"Initializing {train_settings['model'].split('.')[0]} model...")
    model = YOLO(train_settings.get("model", "yolo11n.pt"))

    # 3. Train
    # Ensure freeze is set to freeze the backbone (typically 10 layers for YOLOv8/11)
    # We prioritize the settings file, but if 'freeze' was somehow removed or None, we enforce a default or warn.
    # The prompt explicitly asks to freeze the backbone.
    if train_settings.get("freeze") is None:
        logger.warning("'freeze' parameter missing. Defaulting to 10 to freeze backbone.")
        train_settings["freeze"] = 10

    logger.info(f"Starting training with settings: {train_settings}")
    
    # Train the model
    # Note: We pass the settings dictionary as keyword arguments
    model.train(**train_settings)
    logger.info("Training completed.")
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Export Model
    if args.export:
        try:
            logger.info(f"Exporting model to {args.export} format...")
            model.export(format=args.export,
                            task=export_settings['task'],
                            imgsz=export_settings['imgsz'],
                            batch=export_settings['batch'],
                            half=export_settings['half'],
                            nms=export_settings['nms'],
                            dynamic=export_settings['dynamic'],
                            int8=export_settings['int8'],
                            data=export_settings['data']
                        )
        except Exception as e:
            logger.error(f"Failed to export model to {export_settings['export']}: {e}")

    # 5. Save Training Parameters
    # Create a record of this run
    timestamp_iso = datetime.now().isoformat()
    timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_record = {
        "timestamp_iso": timestamp_iso,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "settings_used": train_settings
    }

    # Save in the project folder (current working directory)
    run_filename = f"runs/train/{train_settings['name']}/training_params.json"
    
    try:
        with open(run_filename, "w") as f:
            json.dump(run_record, f, indent=4)
        logger.info(f"Training parameters saved to '{run_filename}'")
    except Exception as e:
        logger.error(f"Failed to save training parameters: {e}")

if __name__ == "__main__":
    main()
