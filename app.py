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


# --- HELPER FUNCTIONS ---
def normalize_name(raw: str) -> str:
    """Normalize names to lowercase alphanumeric string."""
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def get_name_parts(name: str) -> set:
    """Extract individual name parts (words) from a name string."""
    # Split by common separators and filter out short words
    parts = re.split(r'[^a-zA-Z]+', name.lower())
    return {p for p in parts if len(p) >= 3}  # Only keep words with 3+ characters


def are_words_similar(word1: str, word2: str) -> bool:
    """Check if two words are similar (fuzzy match for typos and variations)."""
    # Exact match
    if word1 == word2:
        return True
    
    # Check if one word is substring of the other (for shortened names like "aishu" in "aishwarya")
    if len(word1) >= 4 and len(word2) >= 4:
        if word1 in word2 or word2 in word1:
            return True
    
    # Check for very similar spelling (Levenshtein-like distance)
    if len(word1) >= 5 and len(word2) >= 5:
        shorter = min(len(word1), len(word2))
        longer = max(len(word1), len(word2))
        
        # Allow length difference of up to 3 characters
        if longer - shorter <= 3:
            # Count matching positions
            matches = sum(c1 == c2 for c1, c2 in zip(word1, word2))
            # Allow 15-20% character differences
            threshold = int(shorter * 0.8)  # 80% similarity required
            if matches >= threshold:
                return True
    
    return False


def match_names(name1: str, name2: str) -> int:
    """Count how many name parts match between two names."""
    parts1 = get_name_parts(name1)
    parts2 = get_name_parts(name2)
    
    # Count exact matches
    exact_matches = len(parts1 & parts2)
    
    # Count fuzzy matches for remaining parts
    fuzzy_matches = 0
    remaining_parts1 = parts1 - parts2
    remaining_parts2 = parts2 - parts1
    
    for p1 in remaining_parts1:
        for p2 in remaining_parts2:
            if are_words_similar(p1, p2):
                fuzzy_matches += 1
                break
    
    return exact_matches + fuzzy_matches


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load CSV and add normalized name key."""
    df = pd.read_csv(CSV_PATH).fillna("")
    df["NameKey"] = df["Name"].map(normalize_name)
    return df


def _load_image_as_gray(path: Path) -> Optional[np.ndarray]:
    """Load image as grayscale and resize for feature extraction."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None and path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(path) as pil_image:
                image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        except Exception:
            return None
    elif image is None:
        return None
    return _resize_for_feature_extraction(image)


def _resize_for_feature_extraction(image: np.ndarray, max_dim: int = 600) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_dim:
        return image
    scale = max_dim / largest_dim
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _extract_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


@st.cache_resource(show_spinner=False)
def load_signature_templates() -> List[SignatureTemplate]:
    """Load all signature templates with features."""
    templates: List[SignatureTemplate] = []
    if not SIGNATURE_DIR.exists():
        return templates
    for image_path in sorted(SIGNATURE_DIR.glob("*")):
        if not image_path.is_file():
            continue
        raw_name = image_path.stem
        if " - " in raw_name:
            _, name_part = raw_name.split(" - ", 1)
        else:
            name_part = raw_name
        name_part = name_part.strip()
        normalized = normalize_name(name_part)
        gray_image = _load_image_as_gray(image_path)
        if gray_image is None:
            continue
        keypoints, descriptors = _extract_features(gray_image)
        templates.append(SignatureTemplate(name_key=normalized,
                                           display_name=name_part,
                                           path=image_path,
                                           image=gray_image,
                                           keypoints=keypoints,
                                           descriptors=descriptors))
    return templates


@st.cache_resource(show_spinner=False)
def load_photo_lookup() -> Dict[str, Path]:
    """Map normalized name keys to photo paths."""
    lookup: Dict[str, Path] = {}
    if not PHOTO_DIR.exists():
        return lookup
    for photo_path in sorted(PHOTO_DIR.glob("*")):
        if not photo_path.is_file():
            continue
        raw_name = photo_path.stem
        # Try to extract name from filename (format: "prefix - Name")
        if " - " in raw_name:
            _, name_part = raw_name.split(" - ", 1)
        elif "-" in raw_name:
            _, name_part = raw_name.split("-", 1)
        else:
            name_part = raw_name
        
        name_part = name_part.strip()
        name_key = normalize_name(name_part)
        lookup[name_key] = photo_path
    return lookup


def compute_match_score(
    uploaded_gray: np.ndarray,
    uploaded_kp: List[cv2.KeyPoint],
    uploaded_des: Optional[np.ndarray],
    template: SignatureTemplate,
) -> float:
    if uploaded_des is None or template.descriptors is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(uploaded_des, template.descriptors, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    denominator = min(len(uploaded_kp), len(template.keypoints)) if template.keypoints else 0
    return float(len(good_matches) / denominator) if denominator else 0.0


def preprocess_upload(upload: io.BytesIO) -> Tuple[np.ndarray, List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Read uploaded file, convert to grayscale, and extract features."""
    file_bytes = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    image = _resize_for_feature_extraction(image)
    keypoints, descriptors = _extract_features(image)
    return image, keypoints, descriptors


def display_user_details(row: pd.Series, photo_lookup: Dict[str, Path], template_image: np.ndarray, score: float) -> None:
    """Display user details in a simple card layout."""
    
    # Simple container with border
    st.markdown("""
        <style>
        .result-card {
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 1.5rem;
            background: white;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    # Name header with score
    score_percentage = score * 100
    st.markdown(f"## 👤 {row['Name']}")
    
    # Score indicator
    if score >= 0.5:
        st.success(f"✅ **Match Confidence: {score_percentage:.1f}%** - Excellent Match")
    elif score >= 0.3:
        st.warning(f"⚠️ **Match Confidence: {score_percentage:.1f}%** - Good Match")
    elif score >= 0.15:
        st.info(f"ℹ️ **Match Confidence: {score_percentage:.1f}%** - Possible Match")
    else:
        st.error(f"❌ **Match Confidence: {score_percentage:.1f}%** - Low Match")
    
    # Layout: Info on left, Images on right
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 📞 Contact Information")
        st.write(f"**Email:** {row['Email Address'] or 'Not provided'}")
        st.write(f"**Phone:** {row['Contact number'] or 'Not provided'}")
        
        st.markdown("#### 📍 Address")
        st.write(row['Address'] or 'Not provided')
    
    with col2:
        # Identity Photo - try multiple matching strategies
        photo_path = None
        
        # Strategy 1: Direct match by normalized name
        photo_path = photo_lookup.get(row["NameKey"])
        
        # Strategy 2: Try to find photo by matching name parts
        if not photo_path or not photo_path.exists():
            best_photo_match = None
            best_photo_score = 0
            
            for photo_key, photo_file in photo_lookup.items():
                # Get original name from photo filename
                photo_stem = photo_file.stem
                if " - " in photo_stem:
                    _, photo_name = photo_stem.split(" - ", 1)
                elif "-" in photo_stem:
                    _, photo_name = photo_stem.split("-", 1)
                else:
                    photo_name = photo_stem
                
                # Match photo name with person's name
                match_score = match_names(photo_name, row["Name"])
                if match_score >= 2 and match_score > best_photo_score:
                    best_photo_score = match_score
                    best_photo_match = photo_file
            
            if best_photo_match:
                photo_path = best_photo_match
        
        # Display photo
        if photo_path and photo_path.exists():
            try:
                st.markdown("**Identity Photo:**")
                with Image.open(photo_path) as img:
                    st.image(img, width=200)
            except Exception as e:
                st.caption(f"⚠️ Photo error: {e}")
        else:
            st.caption("📷 No identity photo found")
        
        # Reference Signature
        st.markdown("**Reference Signature:**")
        st.image(template_image, width=250)
    
    st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---
def main() -> None:
    st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")
    
    # Simple header
    st.title("✍️ Signature Verification System")
    st.markdown("Upload a signature image to match it with registered users.")
    st.divider()

    dataset = load_dataset()
    templates = load_signature_templates()
    photo_lookup = load_photo_lookup()

    if not templates:
        st.error(f"❌ No signature templates found in `{SIGNATURE_DIR}`. Please add signature files.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Choose a signature image to verify", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a signature image to begin verification.")
        return

    try:
        uploaded_image, uploaded_kp, uploaded_des = preprocess_upload(uploaded_file)
    except ValueError as exc:
        st.error(f"❌ Error: {exc}")
        return

    # Display uploaded signature with info
    st.subheader("📝 Uploaded Signature")
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.image(uploaded_image, use_column_width=True)
    
    # Show upload info
    st.info(f"✓ Detected {len(uploaded_kp)} keypoints in uploaded signature. Comparing with {len(templates)} stored signatures...")
    
    st.divider()

    # --- Compute match scores ---
    with st.spinner("🔍 Comparing signatures..."):
        scores: List[Tuple[float, SignatureTemplate]] = []
        for template in templates:
            score = compute_match_score(uploaded_image, uploaded_kp, uploaded_des, template)
            scores.append((score, template))

    if not scores:
        st.warning("⚠️ No signature templates could be compared. Please check your signature files.")
        return

    scores.sort(key=lambda item: item[0], reverse=True)
    
    # Show results header
    st.subheader("🎯 Verification Results")
    
    # Show top matches in an expander
    with st.expander("📊 View All Signature Scores", expanded=False):
        score_table = []
        for idx, (score, template) in enumerate(scores[:15], 1):  # Show top 15
            score_table.append({
                "Rank": idx,
                "Name": template.display_name,
                "Confidence": f"{score * 100:.1f}%"
            })
        st.table(score_table)
    
    # Show available photos for debugging
    with st.expander("📷 Available Identity Photos", expanded=False):
        if photo_lookup:
            photo_info = []
            for photo_key, photo_file in list(photo_lookup.items())[:20]:
                photo_stem = photo_file.stem
                if " - " in photo_stem:
                    _, photo_name = photo_stem.split(" - ", 1)
                elif "-" in photo_stem:
                    _, photo_name = photo_stem.split("-", 1)
                else:
                    photo_name = photo_stem
                photo_info.append({
                    "Display Name": photo_name,
                    "File": photo_file.name
                })
            st.table(photo_info)
        else:
            st.info("No photos found in Photos directory")

    # Find matches and show debug info
    all_matches = []
    
    for score, template in scores[:15]:  # Check top 15 signature matches
        # Try to find matching person in CSV
        matching_person = None
        best_match_count = 0
        
        # Strategy 1: Exact normalized match
        exact_match = dataset[dataset["NameKey"] == template.name_key]
        if not exact_match.empty:
            matching_person = exact_match.iloc[0]
            best_match_count = 99  # High score for exact match
        else:
            # Strategy 2: Name parts matching
            # Compare the display name from template with names in CSV
            for _, person in dataset.iterrows():
                # Count matching name parts between template and CSV name
                match_count = match_names(template.display_name, person["Name"])
                
                # Need at least 2 matching name parts for a valid match
                if match_count >= 2:
                    if match_count > best_match_count:
                        best_match_count = match_count
                        matching_person = person
        
        # Store all valid matches for debugging
        if matching_person is not None:
            all_matches.append({
                "person": matching_person,
                "template": template,
                "sig_score": score,
                "name_match_count": best_match_count
            })
    
    # Show detailed matching results for debugging
    with st.expander("🔍 Detailed Match Analysis (Debug)", expanded=False):
        if all_matches:
            st.write("**Top signature matches with CSV records:**")
            st.caption("🔒 Only matches with ≥80% confidence are verified as authentic")
            debug_table = []
            for idx, match in enumerate(all_matches[:10], 1):
                score_pct = match['sig_score'] * 100
                # Add status indicator
                status = "✅ VERIFIED" if match['sig_score'] >= 0.80 else "⚠️ SUSPICIOUS" if match['sig_score'] >= 0.50 else "❌ REJECTED"
                debug_table.append({
                    "Rank": idx,
                    "Status": status,
                    "CSV Name": match["person"]["Name"],
                    "Signature File": match["template"].display_name,
                    "Confidence": f"{score_pct:.1f}%",
                    "Name Match": "Exact" if match["name_match_count"] == 99 else f"{match['name_match_count']} parts"
                })
            st.table(debug_table)
        else:
            st.warning("No matches found between signatures and CSV records")
    
    # Take the best match (highest signature score)
    best_match = None
    best_match_score = 0
    best_match_template = None
    
    if all_matches:
        best = all_matches[0]  # Already sorted by signature score
        best_match = best["person"]
        best_match_score = best["sig_score"]
        best_match_template = best["template"]
    
    # Fraud detection threshold
    FRAUD_THRESHOLD = 0.80  # 80% minimum confidence required
    
    # Display result
    if best_match is not None:
        confidence_percentage = best_match_score * 100
        
        # Check if confidence meets threshold
        if best_match_score >= FRAUD_THRESHOLD:
            # VERIFIED - High confidence match
            st.success(f"✅ **VERIFIED** - Match Found: **{best_match['Name']}**")
            st.info(f"📊 Matched with stored signature: `{best_match_template.path.name}` | Confidence: **{confidence_percentage:.1f}%**")
            
            # Show side-by-side comparison
            st.markdown("### 🔍 Signature Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Your Uploaded Signature:**")
                st.image(uploaded_image, use_column_width=True)
            with comp_col2:
                st.markdown("**Matched Reference Signature:**")
                st.image(best_match_template.image, use_column_width=True)
            
            st.divider()
            st.markdown("### 👤 User Information")
            display_user_details(best_match, photo_lookup, best_match_template.image, best_match_score)
        else:
            # FRAUD DETECTED - Low confidence
            st.markdown("""
                <div style="background: #dc3545; color: white; padding: 2rem; border-radius: 15px; 
                            text-align: center; margin: 2rem 0; box-shadow: 0 8px 16px rgba(220,53,69,0.4);
                            border: 3px solid #a71d2a;">
                    <h1 style="margin: 0; font-size: 3rem;">🚨 FRAUD DETECTED 🚨</h1>
                    <h2 style="margin: 1rem 0; font-weight: normal;">Signature Verification Failed</h2>
                    <p style="font-size: 1.3rem; margin: 0;">
                        Confidence Score: <strong>{:.1f}%</strong> (Required: 80%)
                    </p>
                </div>
            """.format(confidence_percentage), unsafe_allow_html=True)
            
            st.error("❌ **The uploaded signature does not match the stored signature with sufficient confidence.**")
            
            # Show comparison for investigation
            with st.expander("🔍 View Signature Comparison", expanded=False):
                st.warning(f"Closest match found: **{best_match['Name']}** (Confidence: {confidence_percentage:.1f}%)")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown("**Uploaded Signature:**")
                    st.image(uploaded_image, use_column_width=True)
                with comp_col2:
                    st.markdown("**Best Match in Database:**")
                    st.image(best_match_template.image, use_column_width=True)
                
                st.info("⚠️ This signature may be forged, belong to a different person, or the image quality may be poor.")
    else:
        # No match found at all
        st.markdown("""
            <div style="background: #dc3545; color: white; padding: 2rem; border-radius: 15px; 
                        text-align: center; margin: 2rem 0; box-shadow: 0 8px 16px rgba(220,53,69,0.4);
                        border: 3px solid #a71d2a;">
                <h1 style="margin: 0; font-size: 3rem;">🚨 FRAUD DETECTED 🚨</h1>
                <h2 style="margin: 1rem 0; font-weight: normal;">No Matching Signature Found</h2>
                <p style="font-size: 1.3rem; margin: 0;">
                    This signature is <strong>NOT</strong> in our database
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.error("❌ **The uploaded signature could not be matched with any registered user.**")
        st.warning("⚠️ This person is either not registered or the signature is fraudulent.")
        
        # Show best attempt for investigation
        if scores:
            with st.expander("🔍 View Closest Match (For Investigation)", expanded=False):
                best_score, best_template = scores[0]
                st.info(f"Closest signature: **{best_template.display_name}** (Confidence: {best_score * 100:.1f}%)")
                st.image(best_template.image, caption="Closest matching signature", width=300)


if __name__ == "__main__":
    main()
