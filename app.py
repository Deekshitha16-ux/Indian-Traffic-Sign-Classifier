import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)


# ============================================================
# TRAFFIC SIGN CLASSES
# EXACT ORDER FROM THE TRAINING DATASET
# ============================================================

class_names = [
    "ALL_MOTOR_VEHICLE_PROHIBITED",
    "AXLE_LOAD_LIMIT",
    "BARRIER_AHEAD",
    "BULLOCK_AND_HANDCART_PROHIBITED",
    "BULLOCK_PROHIBITED",
    "CATTLE",
    "COMPULSARY_AHEAD",
    "COMPULSARY_AHEAD_OR_TURN_LEFT",
    "COMPULSARY_AHEAD_OR_TURN_RIGHT",
    "COMPULSARY_CYCLE_TRACK",
    "COMPULSARY_KEEP_LEFT",
    "COMPULSARY_KEEP_RIGHT",
    "COMPULSARY_MINIMUM_SPEED",
    "COMPULSARY_SOUND_HORN",
    "COMPULSARY_TURN_LEFT",
    "COMPULSARY_TURN_LEFT_AHEAD",
    "COMPULSARY_TURN_RIGHT",
    "COMPULSARY_TURN_RIGHT_AHEAD",
    "CROSS_ROAD",
    "CYCLE_CROSSING",
    "CYCLE_PROHIBITED",
    "DANGEROUS_DIP",
    "DIRECTION",
    "FALLING_ROCKS",
    "FERRY",
    "GAP_IN_MEDIAN",
    "GIVE_WAY",
    "GUARDED_LEVEL_CROSSING",
    "HANDCART_PROHIBITED",
    "HEIGHT_LIMIT",
    "HORN_PROHIBITED",
    "HUMP_OR_ROUGH_ROAD",
    "LEFT_HAIR_PIN_BEND",
    "LEFT_HAND_CURVE",
    "LEFT_REVERSE_BEND",
    "LEFT_TURN_PROHIBITED",
    "LENGTH_LIMIT",
    "LOAD_LIMIT",
    "LOOSE_GRAVEL",
    "MEN_AT_WORK",
    "NARROW_BRIDGE",
    "NARROW_ROAD_AHEAD",
    "NO_ENTRY",
    "NO_PARKING",
    "NO_STOPPING_OR_STANDING",
    "OVERTAKING_PROHIBITED",
    "PASS_EITHER_SIDE",
    "PEDESTRIAN_CROSSING",
    "PEDESTRIAN_PROHIBITED",
    "PRIORITY_FOR_ONCOMING_VEHICLES",
    "QUAY_SIDE_OR_RIVER_BANK",
    "RESTRICTION_ENDS",
    "RIGHT_HAIR_PIN_BEND",
    "RIGHT_HAND_CURVE",
    "RIGHT_REVERSE_BEND",
    "RIGHT_TURN_PROHIBITED",
    "ROAD_WIDENS_AHEAD",
    "ROUNDABOUT",
    "SCHOOL_AHEAD",
    "SIDE_ROAD_LEFT",
    "SIDE_ROAD_RIGHT",
    "SLIPPERY_ROAD",
    "SPEED_LIMIT_15",
    "SPEED_LIMIT_20",
    "SPEED_LIMIT_30",
    "SPEED_LIMIT_40",
    "SPEED_LIMIT_5",
    "SPEED_LIMIT_50",
    "SPEED_LIMIT_60",
    "SPEED_LIMIT_70",
    "SPEED_LIMIT_80",
    "STAGGERED_INTERSECTION",
    "STEEP_ASCENT",
    "STEEP_DESCENT",
    "STOP",
    "STRAIGHT_PROHIBITED",
    "TONGA_PROHIBITED",
    "TRAFFIC_SIGNAL",
    "TRUCK_PROHIBITED",
    "TURN_RIGHT",
    "T_INTERSECTION",
    "UNGUARDED_LEVEL_CROSSING",
    "U_TURN_PROHIBITED",
    "WIDTH_LIMIT",
    "Y_INTERSECTION"
]


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "traffic_sign_classification_final.keras"
    )

    return model


model = load_model()


# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    """Normalize image input so it matches the training pipeline more closely."""
    img = image.convert("RGB")

    # Improve contrast and make the sign more prominent.
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = img.filter(ImageFilter.SHARPEN)

    # Keep the sign centered while preserving aspect ratio.
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Resize to the trained input size.
    img = img.resize((96, 96), Image.Resampling.BILINEAR)

    array = np.asarray(img, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


def predict_with_confidence(image: Image.Image):
    """Run a small ensemble of cleaned-up image variants to reduce false matches."""
    variants = []

    base = preprocess_image_for_model(image)
    variants.append(base)

    # Slightly brighten a second version for low-contrast sign images.
    bright = ImageEnhance.Brightness(image.convert("RGB")).enhance(1.1)
    variants.append(preprocess_image_for_model(bright))

    stacked = np.concatenate(variants, axis=0)
    probs = model.predict(stacked, verbose=0)
    avg_probs = probs.mean(axis=0)
    top_3_indices = np.argsort(avg_probs)[-3:][::-1]

    return avg_probs, top_3_indices


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #e8f5e9;
        margin-top: 20px;
        text-align: center;
    }

    .prediction-title {
        font-size: 16px;
        color: #555555;
    }

    .prediction-name {
        font-size: 28px;
        font-weight: bold;
        margin-top: 8px;
    }

    .confidence {
        font-size: 20px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 Indian Traffic Sign Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an Indian traffic sign image and let the trained CNN classify it.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "CNN"
    )

with col2:
    st.metric(
        "Classes",
        "85"
    )

with col3:
    st.metric(
        "Test Accuracy",
        "72.44%"
    )


st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption=uploaded_file.name,
        width=350
    )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    with st.spinner("Analyzing traffic sign..."):
        predictions, top_3_indices = predict_with_confidence(image)

    # --------------------------------------------------------
    # GET TOP 3 PREDICTIONS
    # --------------------------------------------------------

    predicted_index = top_3_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    if confidence < 40:
        st.warning(
            "This image looks low-confidence for the model. Please upload a clearer front-facing sign image for better accuracy."
        )


    # --------------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.03);
            border: 2px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            padding: 22px 18px;
            margin-top: 20px;
            text-align: center;
        ">
            <div style="font-size: 16px; color: #dfe6f1; margin-bottom: 8px;">
                🚦 Predicted Traffic Sign
            </div>
            <div style="font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
                {predicted_class}
            </div>
            <div style="font-size: 20px; color: #7ee2ff;">
                🎯 Confidence: {confidence:.2f}%
            </div>
        </div>
        """.format(predicted_class=predicted_class, confidence=confidence),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    st.subheader("🔎 Top 3 Predictions")

    for rank, index in enumerate(top_3_indices, start=1):

        sign_name = class_names[index]

        probability = predictions[index] * 100

        st.write(
            f"**{rank}. {sign_name}** — {probability:.2f}%"
        )

        st.progress(
            float(predictions[index])
        )


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Model: Custom CNN trained from scratch using TensorFlow/Keras | "
        "85 Indian traffic-sign classes | Test Accuracy: 72.44%"
    )