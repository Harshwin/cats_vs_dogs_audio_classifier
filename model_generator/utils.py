import glob
import os
import shutil

import librosa

from utilities.feature_extract import extract_features


def delete_file_if_exists(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

