import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #666666;
    }

    .letter-box {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #dddddd;
        background-color: #f8f9fa;
    }

    .detected-letter {
        font-size: 100px;
        font-weight: bold;
        margin: 0;
    }

    .confidence-text {
        font-size: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🤟 AI Sign Language Translator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time ASL Alphabet Recognition</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    """
    <div style="text-align:center;">
    Show an American Sign Language alphabet gesture
    to the camera and the AI will identify the letter.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("asl_model.pkl")


try:

    model = load_model()

except Exception as e:

    st.error("❌ Could not load the AI model.")

    st.code(str(e))

    st.stop()

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================================================
# NORMALIZE LANDMARKS
# IMPORTANT:
# This must match the method used during training.
# =========================================================

def normalize_landmarks(hand):

    landmarks = hand.landmark

    # Wrist landmark = 0

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    features = []

    for landmark in landmarks:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        features.extend(
            [
                x,
                y,
                z
            ]
        )

    return features

# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "AI Classes",
        "26"
    )

with col2:

    st.metric(
        "Letters",
        "A – Z"
    )

with col3:

    st.metric(
        "Model Accuracy",
        "98.93%"
    )

st.divider()

# =========================================================
# CAMERA
# =========================================================

camera_image = st.camera_input(
    "📷 Show your ASL sign and click Take Photo"
)

# =========================================================
# PROCESS CAMERA IMAGE
# =========================================================

if camera_image is not None:

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    bytes_data = camera_image.getvalue()

    image = cv2.imdecode(
        np.frombuffer(
            bytes_data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "❌ Could not read the camera image."
        )

        st.stop()

    # -----------------------------------------------------
    # MIRROR CAMERA
    #
    # This matches the working desktop translator.
    # -----------------------------------------------------

    image = cv2.flip(
        image,
        1
    )

    # -----------------------------------------------------
    # RGB CONVERSION
    # -----------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # MEDIAPIPE HAND DETECTION
    # -----------------------------------------------------

    with mp_hands.Hands(

        static_image_mode=True,

        max_num_hands=1,

        min_detection_confidence=0.3,

        min_tracking_confidence=0.3

    ) as hands:

        results = hands.process(
            rgb_image
        )

    # =====================================================
    # HAND DETECTED
    # =====================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # -------------------------------------------------
        # DRAW LANDMARKS
        # -------------------------------------------------

        mp_draw.draw_landmarks(

            image,

            hand,

            mp_hands.HAND_CONNECTIONS

        )

        # -------------------------------------------------
        # NORMALIZE LANDMARKS
        # -------------------------------------------------

        features = normalize_landmarks(
            hand
        )

        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        features = np.array(
            features,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        # =================================================
        # AI PREDICTION
        # =================================================

        try:

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )

            confidence = (
                np.max(probabilities) * 100
            )

        except Exception as e:

            st.error(
                "❌ AI prediction failed."
            )

            st.code(str(e))

            st.stop()

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        left, right = st.columns(
            [1.2, 1]
        )

        # -------------------------------------------------
        # LEFT COLUMN
        # -------------------------------------------------

        with left:

            st.subheader(
                "📷 Detected Hand"
            )

            st.image(

                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),

                use_container_width=True

            )

        # -------------------------------------------------
        # RIGHT COLUMN
        # -------------------------------------------------

        with right:

            st.subheader(
                "🤟 Detected Sign"
            )

            # Large letter

            st.markdown(
                f"""
                <div class="letter-box">

                    <p class="detected-letter">
                        {prediction}
                    </p>

                    <p class="confidence-text">
                        Confidence:
                        <b>{confidence:.2f}%</b>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            # Confidence progress bar

            confidence_value = min(
                max(
                    int(confidence),
                    0
                ),
                100
            )

            st.progress(
                confidence_value
            )

            # -------------------------------------------------
            # CONFIDENCE STATUS
            # -------------------------------------------------

            if confidence >= 80:

                st.success(
                    f"✅ Strong prediction: {prediction}"
                )

            elif confidence >= 50:

                st.warning(
                    f"⚠️ Possible prediction: {prediction}"
                )

            else:

                st.error(
                    f"⚠️ Low confidence: {prediction}"
                )

    # =====================================================
    # NO HAND DETECTED
    # =====================================================

    else:

        st.warning(
            """
            ⚠️ No hand detected.

            Please make sure:

            • Your entire hand is visible
            • Your fingers are not cut off
            • There is enough lighting
            • Your hand is not too far from the camera
            • Try moving your hand closer
            """
        )

# =========================================================
# HOW TO USE
# =========================================================

st.divider()

st.subheader(
    "📖 How to Use"
)

st.markdown(
    """
    **Step 1:** Click the camera button.

    **Step 2:** Allow camera permission.

    **Step 3:** Place your complete hand inside the camera.

    **Step 4:** Make an ASL alphabet sign.

    **Step 5:** Click **Take Photo**.

    **Step 6:** The AI will identify the letter.

    **Tip:** Use good lighting and keep your hand clearly
    visible against the background.
    """
)

# =========================================================
# TECHNOLOGY
# =========================================================

st.divider()

st.subheader(
    "⚙️ Technology Used"
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.write("🐍 Python")

with tech2:

    st.write("👁️ MediaPipe")

with tech3:

    st.write("🤖 Random Forest")

with tech4:

    st.write("🌐 Streamlit")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
    <b>AI Sign Language Translator</b><br>
    SIH Prototype 🤟
    </div>
    """,
    unsafe_allow_html=True
)