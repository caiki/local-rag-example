import os
import json
import requests
import subprocess

import streamlit as st

from exiftool import ExifToolHelper

import utils.logs as logs

###################################
#
# Save File Upload to Disk
#
###################################


def save_uploaded_file(uploaded_file: bytes, save_dir: str):
    """
    Saves the uploaded file to the specified directory.

    Args:
        uploaded_file (BytesIO): The uploaded file content.
        save_dir (str): The directory where the file will be saved.

    Returns:
        None

    Raises:
        Exception: If there is an error saving the file to disk.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logs.log.info(f"Directory {save_dir} did not exist so creating it")
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            logs.log.info(f"Upload {uploaded_file.name} saved to disk")
    except Exception as e:
        logs.log.error(f"Error saving upload to disk: {e}")


###################################
#
# Extract File Metadata
#
###################################

def get_file_metadata(file_path):
    """
    Extracts various metadata for the specified file.

    Args:
        file_path (str): The path to the file.

    Returns:
        A dictionary containing the extracted metadata.

    Raises:
        Exception: If there is an error extracting the metadata.
    """
    try:
        with ExifToolHelper() as et:
            for d in et.get_metadata(file_path):
                return json.dumps(d, indent=2)
    except Exception:
        pass
