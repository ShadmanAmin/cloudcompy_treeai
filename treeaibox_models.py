"""Model management for TreeAIBox.

Handles downloading, caching, and path resolution for pre-trained models.
"""

import os
import json
import sys
import requests
from pathlib import Path


MODEL_BASE_URL = "https://github.com/NRCan/TreeAIBox/releases/download/v1.0/"
_PACKAGE_DIR = Path(__file__).parent


def get_model_dir():
    """Return the local directory for storing downloaded model weights."""
    model_dir = Path.home() / ".treeaibox" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def list_available_models():
    """Return the list of available model names from model_zoo.json."""
    zoo_path = _PACKAGE_DIR / "model_zoo.json"
    with open(zoo_path) as f:
        return json.load(f)


def get_config_path(model_name):
    """Resolve the JSON config file path for a given model name.

    Mapping:
      - treefiltering_*, urbanfiltering_*, woodcls_* -> modules/filter/
      - treeisonet_* (stemcls, treeloc, treeoff, crownoff) -> modules/treeisonet/
    """
    name_lower = model_name.lower()

    if name_lower.startswith("treeisonet_"):
        subfolder = "treeisonet"
    elif name_lower.startswith(("treefiltering_", "urbanfiltering_", "woodcls_")):
        subfolder = "filter"
    else:
        subfolder = "filter"

    config_path = _PACKAGE_DIR / "modules" / subfolder / f"{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs can be found in modules/filter/ and modules/treeisonet/"
        )
    return str(config_path)


def get_model_path(model_name, auto_download=True):
    """Return the path to the .pth model file, downloading if necessary.

    Parameters
    ----------
    model_name : str
        Model name from model_zoo.json.
    auto_download : bool
        If True, automatically download the model if not found locally.

    Returns
    -------
    str
        Path to the .pth file.
    """
    model_dir = get_model_dir()
    model_path = model_dir / f"{model_name}.pth"

    if model_path.exists():
        return str(model_path)

    if not auto_download:
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Set auto_download=True or run: download_model('{model_name}')"
        )

    download_model(model_name)
    return str(model_path)


def download_model(model_name, progress_callback=None):
    """Download a model from the GitHub releases.

    Parameters
    ----------
    model_name : str
        Model name from model_zoo.json.
    progress_callback : callable, optional
        Called with (percent_complete,) during download.
    """
    # GitHub reformats brackets in release filenames
    url_name = model_name.replace("(", "_").replace(")", "")
    url = f"{MODEL_BASE_URL}{url_name}.pth"
    model_dir = get_model_dir()
    local_path = model_dir / f"{model_name}.pth"
    temp_path = model_dir / f"{model_name}.pth.temp"

    print(f"Downloading model: {model_name}")
    print(f"  URL: {url}")
    print(f"  Destination: {local_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        percent = int((downloaded / total_size) * 100)
                        progress_callback(percent)

        # Atomic rename
        if local_path.exists():
            local_path.unlink()
        temp_path.rename(local_path)

        print(f"  Download complete: {local_path}")

    except requests.exceptions.RequestException as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model '{model_name}': {e}") from e
