import os
import logging
import traceback

import numpy as np
import librosa

# --------------------------------------------------
# LOGGER CONFIGURATION
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------
# SINGLE AUDIO MFCC EXTRACTION
# (Used for ONE emotion prediction)
# --------------------------------------------------

def extract_mfcc(
    file_path: str,
    n_mfcc: int = 40,
    duration: float = 3.0,
    offset: float = 0.5
) -> np.ndarray:
    """
    Extract MFCC features from a full audio file
    and return a single mean-pooled MFCC vector.

    Used for:
    - Single emotion prediction
    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio, sample_rate = librosa.load(
            file_path,
            duration=duration,
            offset=offset
        )

        if audio.size == 0:
            raise ValueError("Loaded audio is empty")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc
        )

        # Mean pooling to get fixed-length vector
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        logger.error("Single MFCC extraction failed")
        logger.error(traceback.format_exc())
        raise e


# --------------------------------------------------
# PREPARE INPUT FOR MODEL (LSTM SHAPE)
# --------------------------------------------------

def prepare_input(file_path: str) -> np.ndarray:
    """
    Prepare MFCC features in model-compatible shape.

    Output shape:
    (1, 1, n_mfcc)
    """

    mfcc = extract_mfcc(file_path)
    mfcc = mfcc.reshape(1, 1, mfcc.shape[0])
    return mfcc


# --------------------------------------------------
# AUDIO CHUNKING + MFCC EXTRACTION
# (Used for MULTI emotion prediction)
# --------------------------------------------------

def extract_mfcc_chunks(
    file_path: str,
    chunk_duration: float = 3.0,
    n_mfcc: int = 40
) -> list:
    """
    Split a long audio file into chunks and extract MFCC
    features for each chunk.

    Used for:
    - Multi-emotion (temporal) emotion detection

    Returns:
    - List of MFCC mean vectors (one per chunk)
    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio, sr = librosa.load(file_path)
        chunk_samples = int(chunk_duration * sr)

        mfcc_chunks = []

        for start in range(0, len(audio), chunk_samples):
            end = start + chunk_samples
            chunk = audio[start:end]

            # Skip very small chunks (noise / silence)
            if len(chunk) < chunk_samples // 2:
                continue

            mfcc = librosa.feature.mfcc(
                y=chunk,
                sr=sr,
                n_mfcc=n_mfcc
            )

            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_chunks.append(mfcc_mean)

        return mfcc_chunks

    except Exception as e:
        logger.error("Chunk MFCC extraction failed")
        logger.error(traceback.format_exc())
        raise e
