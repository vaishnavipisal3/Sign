from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional HEIC image support
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "signature.csv"
SIGNATURE_DIR = BASE_DIR / "Signature"
PHOTO_DIR = BASE_DIR / "Photos" / "Identity sized photo (File responses)"


# --- DATA STRUCTURE ---
@dataclass
class SignatureTemplate:
    name_key: str
    display_name: str
    path: Path
    image: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None


# ----------------------------
#       HELPER FUNCTIONS
# ----------------------------

def normalize_name(raw: str) -> str:
    """Normalize names to lowercase alphanumeric string."""
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def get_name_parts(name: str) -> set:
    """Extract individual name parts (words) from a name string."""
    parts = re.split(r'[^a-zA-Z]+', name.lower())
    return {p for p in parts if len(p) >= 3}


def are_words_similar(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
    if len(w1) >= 4 and len(w2) >= 4:
        if w1 in w2 or w2 in w1:
            return True
    return False


def match_names(name1: str, name2: str) -> int:
    """Count how many name parts match between two names."""
    p1 = get_name_parts(name1)
    p2 = get_name_parts(name2)

    exact = len(p1 & p2)
    fuzzy = 0

    for x in (p1 - p2):
        for y in (p2 - p1):
            if are_words_similar(x, y):
                fuzzy += 1
                break

    return exact + fuzzy


# ----------------------------
#   IMAGE + FEATURE HELPERS
# ----------------------------

def _resize_for_feature_extraction(image: np.ndarray, max_dim: int = 600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


@st.cache_data(ttl=600)
def load_dataset() -> pd.DataFrame:
    """Load CSV and normalize names."""
    if not CSV_PATH.exists():
        st.error(f"CSV file not found at {CSV_PATH}")
        return pd.DataFrame(columns=["Name", "Email Address", "Contact number", "Address"])

    df = pd.read_csv(CSV_PATH).fillna("")
    df["NameKey"] = df["Name"].map(normalize_name)
    return df


def _load_image_as_gray(path: Path) -> Optional[np.ndarray]:
    """Load image using OpenCV or Pillow."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        try:
            with Image.open(path) as pil_img:
                img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        except Exception:
            return None

    return _resize_for_feature_extraction(img)


def _extract_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    orb = cv2.ORB_create(nfeatures=1500)
    return orb.detectAndCompute(image, None)


@st.cache_resource(ttl=600)
def load_signature_templates() -> List[SignatureTemplate]:
    templates: List[SignatureTemplate] = []

    if not SIGNATURE_DIR.exists():
        return templates

    for img_path in sorted(SIGNATURE_DIR.glob("*")):
        if not img_path.is_file():
            continue

        name_part = img_path.stem.split(" - ", 1)[-1].strip()
        normalized = normalize_name(name_part)

        gray = _load_image_as_gray(img_path)
        if gray is None:
            continue

        kp, des = _extract_features(gray)

        templates.append(
            SignatureTemplate(
                name_key=normalized,
                display_name=name_part,
                path=img_path,
                image=gray,
                keypoints=kp,
                descriptors=des,
            )
        )

    return templates


@st.cache_resource(ttl=600)
def load_photo_lookup() -> Dict[str, Path]:
    lookup = {}

    if not PHOTO_DIR.exists():
        return lookup

    for img in PHOTO_DIR.glob("*"):
        if not img.is_file():
            continue

        name_part = img.stem.split(" - ", 1)[-1].strip()
        lookup[normalize_name(name_part)] = img

    return lookup


# ----------------------------
#   SIGNATURE COMPARISON
# ----------------------------

def compute_match_score(upload_des: np.ndarray, template: SignatureTemplate) -> float:
    """Compute signature similarity using ORB + BFMatcher."""
    if upload_des is None or template.descriptors is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(upload_des, template.descriptors, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    denom = max(min(len(upload_des), len(template.descriptors)), 1)
    return len(good) / denom


def preprocess_upload(upload: io.BytesIO) -> Tuple[np.ndarray, List[cv2.KeyPoint], Optional[np.ndarray]]:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Invalid image format.")

    image = _resize_for_feature_extraction(image)
    kp, des = _extract_features(image)

    return image, kp, des


# ----------------------------
#          MAIN APP
# ----------------------------

def main():
    st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")

    st.title("✍️ Signature Verification System")
    st.caption("Verify uploaded signatures against the registered database.")
    st.divider()

    dataset = load_dataset()
    templates = load_signature_templates()
    photo_lookup = load_photo_lookup()

    if not templates:
        st.error("⚠️ No signature templates found inside the Signature folder.")
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload signature image", type=["jpg", "jpeg", "png"])

    if not uploaded_file:
        st.info("Upload a signature image to begin.")
        st.stop()

    with st.spinner("Processing signature..."):
        up_img, up_kp, up_des = preprocess_upload(uploaded_file)

    st.image(up_img, caption="Uploaded Signature", use_container_width=True)
    st.info(f"🔍 Extracted {len(up_kp)} keypoints. Comparing against {len(templates)} templates...")

    with st.spinner("Matching signatures..."):
        scores = [(compute_match_score(up_des, t), t) for t in templates]
        scores.sort(reverse=True, key=lambda x: x[0])

    best_score, best_template = scores[0]
    confidence = best_score * 100

    st.divider()
    st.subheader("🎯 Verification Results")

    if confidence >= 75:
        st.success(f"✅ Verified: {best_template.display_name} ({confidence:.2f}% match)")
    elif confidence >= 45:
        st.warning(f"⚠️ Possible Match: {best_template.display_name} ({confidence:.2f}%)")
    else:
        st.error(f"❌ No reliable match found ({confidence:.2f}%)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Uploaded Signature")
        st.image(up_img, use_container_width=True)
    with col2:
        st.markdown("### Best Database Match")
        st.image(best_template.image, use_container_width=True)
        st.caption(f"File: `{best_template.path.name}`")


if __name__ == "__main__":
    main()
