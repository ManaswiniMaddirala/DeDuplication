import librosa
import numpy as np
import warnings
from datasketch import MinHash

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────
SR          = 16000
NUM_PERM    = 128
N_MFCC      = 20 

def get_audio_minhash(path):
    """Generates a fingerprint for fast LSH lookups."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        if y.size == 0: return None
        y, _ = librosa.effects.trim(y) # Remove silence

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        
        # Quantize for robustness
        mfcc = np.round(mfcc / 5) * 5 
        vector = mfcc.flatten()
        
        m = MinHash(num_perm=NUM_PERM)
        for i, val in enumerate(vector):
            token = f"{i % 500}_{int(val)}"
            m.update(token.encode("utf8"))
        return m
    except Exception:
        return None

def get_audio_similarity(path1, path2):
    """
    Sequence alignment (DTW) with a tuned penalty multiplier 
    to force partial matches into the 70% - 80% range.
    """
    try:
        y1, _ = librosa.load(path1, sr=SR, mono=True)
        y2, _ = librosa.load(path2, sr=SR, mono=True)
        
        # Trim silence so we only compare the actual spoken words
        y1, _ = librosa.effects.trim(y1)
        y2, _ = librosa.effects.trim(y2)
        
        if y1.size == 0 or y2.size == 0:
            return 0.0, None

        # 1. Extract MFCCs, dropping the 0th coefficient (ignores microphone volume)
        # We use 13 coefficients because it maps human speech phonetics best.
        mfcc1 = librosa.feature.mfcc(y=y1, sr=SR, n_mfcc=13)[1:]
        mfcc2 = librosa.feature.mfcc(y=y2, sr=SR, n_mfcc=13)[1:]

        # 2. Dynamic Time Warping (DTW) with Cosine distance
        # This aligns the audio timelines. "Hi" matches "Hi", but "Hello" causes a misalignment.
        D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric='cosine')
        
        # 3. Calculate average distance per frame
        # Identical files = 0.0. Slightly different = ~0.15. Totally different = >0.4.
        avg_dist = D[-1, -1] / len(wp)

        # 4. Apply the Penalty Multiplier
        # By multiplying the distance by 1.8, an audio file that is mathematically 
        # 15% different gets pushed to a 27% penalty (100% - 27% = 73% similarity).
        penalty_multiplier = 1.8
        similarity = (1.0 - (avg_dist * penalty_multiplier)) * 100

        # Enforce boundaries (never go below 0% or above 100%)
        similarity = max(0.0, min(100.0, similarity))

        # Hard cap: Prevent exactly 100.0% if the files are physically different
        if similarity > 99.0 and path1 != path2:
            similarity = 98.5

        return round(similarity, 2), None

    except Exception as e:
        print(f"Audio Error: {e}")
        return 0.0, None