"""
Image Utility Functions using OpenCV + MinHash
Supports subcategory-aware duplicate detection:
  - mark_list   : Student mark sheets / grade cards
  - fee_receipt : Payment / fee receipts
  - general     : General photos / stills
"""
import cv2
import numpy as np
from datasketch import MinHash

# ─── Subcategory constants ────────────────────────────────────
IMAGE_SUBTYPES = ("mark_list", "fee_receipt", "general")


# ─── Core Feature Extraction ──────────────────────────────────

def extract_features(image_path: str):
    """
    Extract a flat feature vector (colour + structural histogram) from an image.

    Strategy:
      • Resize to 256×256 for speed/consistency
      • Grayscale histogram  (64 bins)  – brightness/contrast distribution
      • Horizontal edge map via Sobel – captures tabular structure (useful for
        mark lists and receipts which have many horizontal lines)
      • Edge histogram (32 bins) appended to the colour histogram

    Returns a flattened numpy array or None on read failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. Resize
    img = cv2.resize(img, (256, 256))

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Colour/brightness histogram (64 bins, normalised 0–100)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)

    # 4. Edge histogram via Sobel (horizontal edges – 32 bins)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y).astype(np.float32)
    edge_hist = cv2.calcHist([magnitude], [0], None, [32], [0, 512])
    cv2.normalize(edge_hist, edge_hist, 0, 100, cv2.NORM_MINMAX)

    return np.concatenate([hist.flatten(), edge_hist.flatten()])


def features_to_tokens(features):
    """
    Convert numerical feature vector into string tokens for MinHash.
    Bin index + bucketed value → e.g. "0_50".
    Bucketing allows slight variations (fuzzy matching).
    """
    tokens = []
    for i, val in enumerate(features):
        bucket = int(val)
        if bucket > 0:
            tokens.append(f"{i}_{bucket}")
    return tokens


def build_minhash(tokens, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf8"))
    return m


# ─── Pairwise Similarity ──────────────────────────────────────

def get_image_similarity(file1_path: str, file2_path: str) -> float:
    """
    Compare two images using OpenCV features + MinHash Jaccard similarity.
    Returns similarity as a float in [0, 100].
    """
    f1 = extract_features(file1_path)
    f2 = extract_features(file2_path)

    if f1 is None or f2 is None:
        return 0.0

    tokens1 = features_to_tokens(f1)
    tokens2 = features_to_tokens(f2)

    # Completely empty images → treat as identical
    if not tokens1 and not tokens2:
        return 100.0
    if not tokens1 or not tokens2:
        return 0.0

    m1 = build_minhash(tokens1)
    m2 = build_minhash(tokens2)

    return round(m1.jaccard(m2) * 100, 2)


# ─── Subcategory-Aware Comparison ────────────────────────────

def get_image_similarity_by_subtype(
    new_file_path: str,
    existing_files: list,           # list of UploadedFile ORM objects
    image_subtype: str = "general",
) -> tuple:
    """
    Compare *new_file_path* against every file in *existing_files* that
    shares the same *image_subtype* (stored as file_type in the DB, e.g.
    "image_mark_list", "image_fee_receipt", "image_general").

    Returns (best_similarity: float, matched_filename: str).

    Isolation rules
    ───────────────
    mark_list    → compared ONLY against other mark_list images
    fee_receipt  → compared ONLY against other fee_receipt images
    general      → compared ONLY against other general images

    This prevents false positives between structurally similar document
    types (e.g. a mark sheet matching a receipt because both have tables).
    """
    db_key = _subtype_db_key(image_subtype)

    best_sim: float = 0.0
    matched_file: str = "None"

    for stored in existing_files:
        # ── Only compare within the same subtype bucket ──────
        if stored.file_type != db_key:
            continue

        import os
        if not os.path.exists(stored.file_path):
            continue

        try:
            sim = get_image_similarity(new_file_path, stored.file_path)
        except Exception:
            continue

        if sim > best_sim:
            best_sim = sim
            matched_file = stored.filename

    return round(best_sim, 2), matched_file


# ─── Helpers ─────────────────────────────────────────────────

def _subtype_db_key(image_subtype: str) -> str:
    """
    Map a friendly subtype name to the file_type string stored in the DB.

    mark_list    → "image_mark_list"
    fee_receipt  → "image_fee_receipt"
    general      → "image_general"
    (unknown)    → "image_general"
    """
    mapping = {
        "mark_list":   "image_mark_list",
        "fee_receipt": "image_fee_receipt",
        "general":     "image_general",
    }
    return mapping.get(image_subtype, "image_general")


def subtype_db_key(image_subtype: str) -> str:
    """Public alias for _subtype_db_key (used by app.py)."""
    return _subtype_db_key(image_subtype)


def is_valid_image_subtype(subtype: str) -> bool:
    return subtype in IMAGE_SUBTYPES