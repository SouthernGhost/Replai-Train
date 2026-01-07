import json
import sys
import os
import logging
from logging import getLogger
from roboflow import Roboflow


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = getLogger(__name__)


def load_settings(settings_path: str) -> dict:
    if not os.path.isfile(settings_path):
        logger.error("Settings file not found: %s", settings_path)
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    logger.info("Loading settings from %s", settings_path)

    with open(settings_path, "r") as f:
        return json.load(f)


def download_dataset(settings: dict, output_dir: str) -> str:
    required_keys = ["api_key", "workspace", "project", "version", "format"]
    missing = [k for k in required_keys if k not in settings]

    if missing:
        logger.error("Missing required settings keys: %s", missing)
        raise ValueError(f"Missing required settings keys: {missing}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Dataset output directory: %s", os.path.abspath(output_dir))

    logger.info(
        "Initializing Roboflow (workspace=%s, project=%s, version=%s, format=%s)",
        settings["workspace"],
        settings["project"],
        settings["version"],
        settings["format"],
    )

    rf = Roboflow(api_key=settings["api_key"])
    project = rf.workspace(settings["workspace"]).project(settings["project"])
    version = project.version(settings["version"])
    
    abs_output_dir = os.path.abspath(output_dir)
    dataset = version.download(
        model_format=settings["format"],
        location=abs_output_dir
    )

    logger.info("Dataset downloaded and extracted successfully")
    logger.debug("Dataset location: %s", dataset.location)

    # Verify files
    if os.path.exists(dataset.location):
        files = os.listdir(dataset.location)
        logger.info("Files in download directory: %s", files)
        if not files:
            logger.warning("Download directory is empty! Attempting manual download...")
            
            # Manual download fallback
            import requests
            import zipfile
            import io

            # Construct API URL
            api_url = f"https://api.roboflow.com/{settings['workspace']}/{settings['project']}/{settings['version']}/{settings['format']}?api_key={settings['api_key']}"
            logger.info("Fetching download URL from: %s", api_url)
            
            resp = requests.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                if 'export' in data and 'link' in data['export']:
                    download_link = data['export']['link']
                    logger.info("Downloading zip from: %s", download_link)
                    
                    zip_resp = requests.get(download_link)
                    if zip_resp.status_code == 200:
                        logger.info("Extracting zip to %s", dataset.location)
                        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                            z.extractall(dataset.location)
                        logger.info("Manual extraction complete.")
                        logger.info("Files after manual extraction: %s", os.listdir(dataset.location))
                    else:
                        logger.error("Failed to download zip: %s", zip_resp.status_code)
                else:
                    logger.error("No download link found in response: %s", data)
            else:
                logger.error("Failed to get export info: %s", resp.status_code)

    else:
        logger.error("Download directory does not exist: %s", dataset.location)

    return dataset.location


def main():
    if len(sys.argv) != 3:
        logger.error(
            "Invalid arguments. "
            "Usage: python download_data.py <path/to/settings_file.json> <output_directory>"
        )
        sys.exit(1)

    settings_path = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        settings = load_settings(settings_path)
        download_dataset(settings, output_dir)
    except Exception:
        logger.exception("Dataset download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
