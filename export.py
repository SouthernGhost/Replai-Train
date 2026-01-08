import json
import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Export YOLO model based on settings')
    parser.add_argument('--settings', type=str, required=True, help='Path to settings.json')
    parser.add_argument('--weights', type=str, default='models/best.pt', help='Path to model weights')
    
    args = parser.parse_args()

    if not os.path.exists(args.settings):
        raise FileNotFoundError(f"Settings file not found: {args.settings}")

    # Load settings
    with open(args.settings, 'r') as f:
        settings = json.load(f)

    # Get export configuration
    # Check if 'export' key exists, otherwise use the whole dict
    export_config = settings.get('export', settings)

    # Initialize model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)

    # Run export
    print(f"Exporting model with settings: {export_config}")
    model.export(**export_config)

if __name__ == '__main__':
    main()
