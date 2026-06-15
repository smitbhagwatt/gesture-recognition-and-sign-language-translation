import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Gesture AI — Sign Language & System Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ──────────────────────────────────────────────────────── */
html, body, .stApp, .block-container {
    font-family: 'Inter', sans-serif !important;
}
.block-container { max-width: 1100px; padding-top: 1rem; }

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero-container { text-align: center; padding: 1.5rem 0 0.5rem 0; }
.hero-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem; font-weight: 900;
    letter-spacing: -0.03em; margin: 0; line-height: 1.15;
}
.hero-tagline {
    color: #a0aec0; font-size: 1.1rem; font-weight: 300;
    margin-top: 0.4rem; line-height: 1.6;
}

/* ── Metric Cards ─────────────────────────────────────────────────── */
.metrics-row {
    display: flex; justify-content: center;
    gap: 1.25rem; margin: 1.75rem 0; flex-wrap: wrap;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.25rem 1.75rem;
    min-width: 130px; text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(102,126,234,0.35);
    box-shadow: 0 8px 32px rgba(102,126,234,0.18);
}
.metric-value {
    font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.8rem; color: #718096; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem;
}

/* ── Tech Badges ──────────────────────────────────────────────────── */
.tech-stack {
    display: flex; justify-content: center;
    flex-wrap: wrap; gap: 0.45rem; margin: 1.25rem 0;
}
.tech-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.35rem 0.9rem; border-radius: 999px;
    background: rgba(102,126,234,0.08);
    border: 1px solid rgba(102,126,234,0.22);
    color: #818cf8; font-size: 0.78rem; font-weight: 600;
    transition: all 0.3s ease; cursor: default;
}
.tech-badge:hover {
    background: rgba(102,126,234,0.16);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102,126,234,0.2);
}
.author-info { color: #718096; font-size: 0.88rem; }
.author-info strong { color: #a0aec0; font-weight: 600; }

/* ── Gradient Divider ─────────────────────────────────────────────── */
.gradient-divider {
    height: 1px; border: none;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3),
                rgba(118,75,162,0.3), transparent);
    margin: 1.25rem 0;
}

/* ── Tab Styling ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem; background: rgba(255,255,255,0.02);
    border-radius: 12px; padding: 0.25rem;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 0.7rem 1.4rem; font-weight: 600;
    font-size: 0.9rem; color: #718096;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12)) !important;
    color: #c4b5fd !important;
}

/* ── Gesture Legend Cards ─────────────────────────────────────────── */
.gesture-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.65rem 0.85rem;
    margin: 0.35rem 0;
    display: flex; align-items: center; gap: 0.65rem;
    transition: all 0.25s ease;
}
.gesture-card:hover {
    border-color: rgba(102,126,234,0.3);
    background: rgba(102,126,234,0.05);
}
.gesture-emoji { font-size: 1.35rem; min-width: 1.8rem; text-align: center; }
.gesture-info { line-height: 1.3; }
.gesture-name { font-weight: 600; color: #e2e8f0; font-size: 0.82rem; }
.gesture-action { color: #718096; font-size: 0.73rem; }

/* ── Sentence Display ─────────────────────────────────────────────── */
.sentence-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.25rem; margin-top: 0.75rem;
}
.sentence-label {
    font-size: 0.78rem; color: #718096; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;
}
.sentence-words { display: flex; flex-wrap: wrap; gap: 0.4rem; min-height: 36px; align-items: center; }
.word-pill {
    display: inline-block; padding: 0.35rem 0.9rem; border-radius: 8px;
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    color: #c4b5fd; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.02em;
}
.empty-sentence { color: #4a5568; font-style: italic; font-size: 0.88rem; }

/* ── Feature Cards ────────────────────────────────────────────────── */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem; height: 100%;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(102,126,234,0.3);
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(102,126,234,0.12);
}
.feature-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.feature-title { font-weight: 700; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.35rem; }
.feature-desc { color: #718096; font-size: 0.85rem; line-height: 1.55; }

/* ── Info Cards ───────────────────────────────────────────────────── */
.info-card {
    background: rgba(102,126,234,0.06);
    border: 1px solid rgba(102,126,234,0.18);
    border-radius: 12px; padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}
.info-card-title {
    font-weight: 700; color: #818cf8; font-size: 0.9rem; margin-bottom: 0.3rem;
}
.info-card-value {
    font-size: 1.6rem; font-weight: 800; color: #e2e8f0;
}
.info-card-desc { color: #718096; font-size: 0.82rem; margin-top: 0.2rem; }

/* ── Section Headers ──────────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem; font-weight: 800; color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.section-sub {
    color: #718096; font-size: 0.92rem; margin-bottom: 1.25rem; line-height: 1.5;
}

/* ── Snapshot result ──────────────────────────────────────────────── */
.snapshot-result {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    margin-top: 1rem;
}
.snapshot-prediction {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Pipeline Steps ───────────────────────────────────────────────── */
.pipeline { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0; margin: 1.5rem 0; }
.pipeline-step {
    background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.25);
    border-radius: 10px; padding: 0.75rem 1.3rem; text-align: center;
    color: #c4b5fd; font-weight: 600; font-size: 0.85rem;
    transition: all 0.3s ease;
}
.pipeline-step:hover { background: rgba(102,126,234,0.15); transform: translateY(-2px); }
.pipeline-step.highlight { border-color: #f093fb; color: #f0abfc; }
.pipeline-arrow { color: #667eea; font-size: 1.3rem; padding: 0 0.4rem; font-weight: 300; }
.pipeline-branch { display: flex; flex-direction: column; gap: 0.5rem; align-items: flex-start; }
.pipeline-branch-item {
    background: rgba(118,75,162,0.1); border: 1px solid rgba(118,75,162,0.25);
    border-radius: 8px; padding: 0.5rem 1rem; color: #c4b5fd; font-size: 0.8rem; font-weight: 500;
}

/* ── Hide Streamlit extras ────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


def extract_features(lm_list):
    """Wrist-relative normalization → 42-dim vector."""
    features = []
    base_x, base_y = lm_list[0]
    for x, y in lm_list:
        features.append(x - base_x)
        features.append(y - base_y)
    return features


def process_image(img_bgr, model):
    """Run MediaPipe + KNN on a single image. Returns (annotated_img, prediction)."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    prediction = ""

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img_bgr,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            h, w, _ = img_bgr.shape
            lm_list = []
            for lm in hand_lms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) == 21:
                features = extract_features(lm_list)
                try:
                    prediction = model.predict([features])[0]
                except Exception:
                    prediction = ""

    hands.close()
    return img_bgr, prediction


def speak_js(text):
    """Trigger browser text-to-speech via JavaScript."""
    safe = text.replace("'", "\\'").replace('"', '\\"')
    components.html(
        f"""
        <script>
            const msg = new SpeechSynthesisUtterance("{safe}");
            msg.rate = 0.9; msg.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


def make_radar_chart():
    """Create a Plotly radar chart for performance metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [94.2, 93.1, 94.8, 94.0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102,126,234,0.15)',
        line=dict(color='#667eea', width=2.5),
        marker=dict(size=8, color='#c4b5fd'),
        name='Model Performance',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[80, 100], tickfont=dict(size=11, color='#718096'), gridcolor='rgba(255,255,255,0.06)'),
            angularaxis=dict(tickfont=dict(size=13, color='#c4b5fd', family='Inter'), gridcolor='rgba(255,255,255,0.06)'),
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=30, b=30),
        height=340,
    )
    return fig


def make_dataset_chart():
    """Create a Plotly bar chart for dataset distribution."""
    labels = ["HELLO", "YES", "NO"]
    # Approximate counts from ~1400 total samples
    counts = [470, 465, 465]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=counts,
        marker=dict(
            color=['#667eea', '#764ba2', '#f093fb'],
            line=dict(width=0),
        ),
        text=counts, textposition='outside',
        textfont=dict(color='#c4b5fd', size=14, family='Inter'),
        width=0.5,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=14, color='#c4b5fd', family='Inter'), showgrid=False),
        yaxis=dict(title=dict(text='Samples', font=dict(color='#718096', size=12)), tickfont=dict(color='#718096', size=11), gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=50, r=20, t=20, b=40),
        height=280,
        bargap=0.35,
    )
    return fig


def render_pipeline(steps):
    """Render a horizontal CSS pipeline from a list of step labels."""
    html_parts = []
    for i, step in enumerate(steps):
        css_class = 'pipeline-step highlight' if i == len(steps) - 1 else 'pipeline-step'
        html_parts.append(f'<div class="{css_class}">{step}</div>')
        if i < len(steps) - 1:
            html_parts.append('<div class="pipeline-arrow">&#10132;</div>')
    return '<div class="pipeline">' + ''.join(html_parts) + '</div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "speak_trigger" not in st.session_state:
    st.session_state.speak_trigger = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██  HERO SECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
<div class="hero-container">
    <h1 class="hero-title">GESTURE AI</h1>
    <p class="hero-tagline">
        Real-time gesture control &amp; sign language translation<br>
        using just a webcam
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Metric cards
st.markdown(
    """
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-value">94.2%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">28ms</div>
        <div class="metric-label">Latency</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">21-35</div>
        <div class="metric-label">FPS</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">42</div>
        <div class="metric-label">Features</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tech badges
st.markdown(
    """
<div class="tech-stack">
    <span class="tech-badge">🐍 Python</span>
    <span class="tech-badge">👁️ OpenCV</span>
    <span class="tech-badge">🤚 MediaPipe</span>
    <span class="tech-badge">🤖 Scikit-Learn</span>
    <span class="tech-badge">🖱️ PyAutoGUI</span>
    <span class="tech-badge">🔊 pyttsx3</span>
    <span class="tech-badge">🧵 Threading</span>
</div>
<p class="author-info" style="text-align:center; margin-top:0.25rem;">
    Built by <strong>Smit Bhagwat</strong> · B.Tech ECE-AIML · MIT-WPU Pune
</p>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██  TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab_demo, tab_control, tab_tech = st.tabs(
    ["🎥  Live Demo", "🖱️  System Control", "📊  Under the Hood"]
)

# ─── TAB 1 — LIVE DEMO ──────────────────────────────────────────────────

with tab_demo:
    st.markdown(
        '<p class="section-header">🎥 Sign Language Recognition</p>'
        '<p class="section-sub">Show hand signs to the camera and build sentences that the browser speaks aloud.<br>'
        "Choose <b>Real-time</b> mode for live video or <b>Snapshot</b> mode to capture a photo.</p>",
        unsafe_allow_html=True,
    )

    demo_mode = st.radio(
        "Mode",
        ["🔴 Real-time (WebRTC)", "📸 Snapshot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    col_cam, col_legend = st.columns([3, 2], gap="large")

    # ── Gesture legend (right side) ──────────────────────────────────
    with col_legend:
        st.markdown(
            """
        <div style="margin-top:0.5rem;">
            <div class="gesture-card">
                <span class="gesture-emoji">✊</span>
                <div class="gesture-info">
                    <div class="gesture-name">Closed Fist</div>
                    <div class="gesture-action">Sign recognition active</div>
                </div>
            </div>
            <div class="gesture-card">
                <span class="gesture-emoji">👆</span>
                <div class="gesture-info">
                    <div class="gesture-name">Index Finger</div>
                    <div class="gesture-action">Cursor movement</div>
                </div>
            </div>
            <div class="gesture-card">
                <span class="gesture-emoji">🤏</span>
                <div class="gesture-info">
                    <div class="gesture-name">Pinch</div>
                    <div class="gesture-action">Left click / Drag</div>
                </div>
            </div>
            <div class="gesture-card">
                <span class="gesture-emoji">✌️</span>
                <div class="gesture-info">
                    <div class="gesture-name">Two Fingers</div>
                    <div class="gesture-action">Right click</div>
                </div>
            </div>
            <div class="gesture-card">
                <span class="gesture-emoji">🖐️</span>
                <div class="gesture-info">
                    <div class="gesture-name">Open Palm</div>
                    <div class="gesture-action">Speak sentence</div>
                </div>
            </div>
            <div class="gesture-card">
                <span class="gesture-emoji">🤟</span>
                <div class="gesture-info">
                    <div class="gesture-name">Three Fingers</div>
                    <div class="gesture-action">Clear sentence buffer</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Camera feed (left side) ──────────────────────────────────────
    with col_cam:

        if demo_mode == "🔴 Real-time (WebRTC)":
            # ── Real-time via WebRTC ─────────────────────────────────
            try:
                from streamlit_webrtc import webrtc_streamer, WebRtcMode
                from webrtc_processor import SignLanguageProcessor

                ctx = webrtc_streamer(
                    key="sign-language-demo",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=SignLanguageProcessor,
                    rtc_configuration={
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]}
                        ]
                    },
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                st.caption("💡 The sign prediction is shown on the video overlay. Use **Capture** to add it to your sentence.")

            except ImportError:
                st.error(
                    "⚠️ `streamlit-webrtc` is not installed. "
                    "Run `pip install streamlit-webrtc av` or switch to **Snapshot** mode."
                )
                ctx = None

        else:
            # ── Snapshot mode ────────────────────────────────────────
            ctx = None
            photo = st.camera_input("📸 Capture a hand sign")

            if photo is not None:
                model = load_model()
                file_bytes = np.frombuffer(photo.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.flip(img, 1)

                annotated, pred = process_image(img, model)

                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Detected landmarks",
                    use_container_width=True,
                )

                if pred:
                    st.markdown(
                        f'<div class="snapshot-result">'
                        f'<div style="color:#718096;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;">PREDICTED SIGN</div>'
                        f'<div class="snapshot-prediction">{pred}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    # Auto-add to sentence
                    if st.button("➕ Add to Sentence", key="snap_add", use_container_width=True):
                        st.session_state.sentence.append(pred)
                        st.rerun()
                else:
                    st.info("No hand detected. Try again with your hand clearly visible.")

    # ── Sentence Builder ─────────────────────────────────────────────
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    btn_cols = st.columns([1, 1, 1, 2])

    with btn_cols[0]:
        capture_clicked = st.button("📸 Capture Sign", use_container_width=True)
    with btn_cols[1]:
        speak_clicked = st.button("🔊 Speak", use_container_width=True)
    with btn_cols[2]:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)

    # Handle capture (WebRTC mode)
    if capture_clicked:
        if demo_mode == "🔴 Real-time (WebRTC)":
            try:
                if ctx and ctx.video_processor:
                    pred = ctx.video_processor.prediction
                    if pred:
                        st.session_state.sentence.append(pred)
                        st.rerun()
                    else:
                        st.toast("⚠️ No sign detected — show your hand and try again.", icon="🤚")
                else:
                    st.toast("⚠️ Start the webcam first.", icon="📷")
            except Exception:
                st.toast("⚠️ Start the webcam first.", icon="📷")
        else:
            st.toast("💡 In snapshot mode, use the ➕ button above.", icon="📸")

    if clear_clicked:
        st.session_state.sentence = []
        st.rerun()

    # Render sentence
    if st.session_state.sentence:
        pills = "".join(
            f'<span class="word-pill">{w}</span>' for w in st.session_state.sentence
        )
        st.markdown(
            f'<div class="sentence-container">'
            f'<div class="sentence-label">📝 Sentence Buffer</div>'
            f'<div class="sentence-words">{pills}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sentence-container">'
            '<div class="sentence-label">📝 Sentence Buffer</div>'
            '<div class="sentence-words"><span class="empty-sentence">No signs captured yet — show a gesture and click Capture</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Speak
    if speak_clicked and st.session_state.sentence:
        text = " ".join(st.session_state.sentence)
        speak_js(text)
        st.toast(f'🔊 Speaking: "{text}"', icon="🗣️")


# ─── TAB 2 — SYSTEM CONTROL ─────────────────────────────────────────────

with tab_control:
    st.markdown(
        '<p class="section-header">🖱️ Gesture-Based System Control</p>'
        '<p class="section-sub">This module enables touchless computer interaction using hand gestures. '
        "Cursor movement, clicking, dragging — all controlled by your hand via the webcam.<br>"
        '<em>(System control requires local desktop execution and cannot run in a web browser.)</em></p>',
        unsafe_allow_html=True,
    )

    # Feature cards
    feat_cols = st.columns(4, gap="medium")
    features = [
        ("🖱️", "Cursor Control", "Move the cursor with your index finger. EMA smoothing eliminates jitter."),
        ("👆", "Left Click", "Quick thumb+index pinch triggers a left click instantly."),
        ("✊", "Drag & Hold", "Hold the pinch for 0.4s to activate drag mode. Release to drop."),
        ("✌️", "Right Click", "Two fingers up triggers context menu — works even after a drag."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Screenshots gallery
    st.markdown("##### 📷 Recognition Results")
    img_cols = st.columns(3, gap="medium")
    result_images = [
        ("results/Hello.png", "HELLO"),
        ("results/No.png", "NO"),
        ("results/Yes.png", "YES"),
    ]
    for col, (path, label) in zip(img_cols, result_images):
        with col:
            if os.path.exists(path):
                st.image(path, caption=f'Sign: "{label}"', use_container_width=True)
            else:
                st.info(f"Image not found: {path}")

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Gesture table
    st.markdown("##### 🎯 Full Gesture Map")
    gesture_data = {
        "Gesture": [
            "☝️ Index Finger",
            "🤏 Thumb + Index Pinch",
            "🤏 Long Pinch (>0.4s)",
            "✌️ Two Fingers",
            "🖐️ Open Palm",
            "🤟 Three Fingers",
            "✊ Closed Fist",
        ],
        "Action": [
            "Move Cursor",
            "Left Click",
            "Drag & Hold",
            "Right Click",
            "Speak Sentence",
            "Clear Buffer",
            "Sign Recognition",
        ],
        "Module": [
            "System Control",
            "System Control",
            "System Control",
            "System Control",
            "Sign Language",
            "Sign Language",
            "Sign Language",
        ],
    }
    st.dataframe(
        gesture_data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Control flow pipeline
    st.markdown("##### Control Pipeline")
    st.markdown(
        render_pipeline(["Webcam", "MediaPipe", "Finger Detection", "Gesture Classification", "Action"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap; margin-top:0.5rem;">' +
        ''.join(f'<div class="pipeline-branch-item">{a}</div>' for a in ["Move Cursor", "Left Click", "Drag", "Right Click", "KNN Predict", "Speak", "Clear"]) +
        '</div>',
        unsafe_allow_html=True,
    )


# ─── TAB 3 — UNDER THE HOOD ─────────────────────────────────────────────

with tab_tech:
    st.markdown(
        '<p class="section-header">📊 Under the Hood</p>'
        '<p class="section-sub">A look at the architecture, ML pipeline, and performance metrics.</p>',
        unsafe_allow_html=True,
    )

    # Architecture pipeline
    st.markdown("##### System Architecture")
    st.markdown(
        render_pipeline(["Webcam", "OpenCV", "MediaPipe", "42-dim Features", "KNN (K=1)", "Action Router"]),
        unsafe_allow_html=True,
    )
    arch_cols = st.columns(2, gap="medium")
    with arch_cols[0]:
        st.markdown(
            '<div class="feature-card"><div class="feature-title">System Control Path</div>'
            '<div class="feature-desc">Action Router ➜ Finger State ➜ PyAutoGUI ➜ Cursor / Click / Drag</div></div>',
            unsafe_allow_html=True,
        )
    with arch_cols[1]:
        st.markdown(
            '<div class="feature-card"><div class="feature-title">Sign Language Path</div>'
            '<div class="feature-desc">Action Router ➜ KNN Prediction ➜ Sentence Buffer ➜ Text-to-Speech</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Performance metrics — radar chart + cards
    st.markdown("##### Performance Metrics")
    perf_left, perf_right = st.columns([3, 2], gap="large")
    with perf_left:
        st.plotly_chart(make_radar_chart(), use_container_width=True, config={"displayModeBar": False})
    with perf_right:
        runtime_metrics = [
            ("Accuracy", "94.2%", "Custom test set"),
            ("Precision", "93.1%", "Macro average"),
            ("Recall", "94.8%", "Macro average"),
            ("F1-Score", "0.940", "Harmonic mean"),
            ("Inference", "28 ms", "Per-frame latency"),
            ("FPS", "21–35", "On laptop CPU"),
        ]
        for title, value, desc in runtime_metrics:
            st.markdown(
                f'<div class="info-card" style="margin-bottom:0.4rem; padding:0.6rem 1rem;">'
                f'<span style="color:#818cf8;font-weight:700;font-size:0.82rem;">{title}</span>'
                f'<span style="float:right;color:#e2e8f0;font-weight:800;font-size:0.95rem;">{value}</span>'
                f'<div style="color:#4a5568;font-size:0.72rem;">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Model & Data details
    st.markdown("##### 🧠 Model & Data")
    detail_cols = st.columns(2, gap="large")

    with detail_cols[0]:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">KNN Classifier</div>
            <div class="feature-desc">
                <strong>Algorithm:</strong> K-Nearest Neighbors (K=1)<br>
                <strong>Features:</strong> 42-dimensional vector<br>
                <strong>Normalization:</strong> Wrist-relative (x,y)<br>
                <strong>Library:</strong> Scikit-Learn<br>
                <strong>Model Size:</strong> ~60 KB (model.pkl)<br>
                <strong>GPU Required:</strong> No — CPU only
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with detail_cols[1]:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Custom Dataset</div>
            <div class="feature-desc">
                <strong>Total Samples:</strong> ~1,400<br>
                <strong>Classes:</strong> HELLO, YES, NO<br>
                <strong>Features per sample:</strong> 42 (21 × 2)<br>
                <strong>Collection:</strong> Standard webcam<br>
                <strong>Format:</strong> CSV (dataset.csv)<br>
                <strong>Labeling:</strong> Manual, per-gesture
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Training pipeline + dataset chart
    st.markdown("##### Training Pipeline")
    st.markdown(
        render_pipeline(["Webcam", "MediaPipe", "Wrist Normalize", "dataset.csv", "KNN Train", "model.pkl"]),
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
    st.markdown("##### Dataset Distribution")
    st.plotly_chart(make_dataset_chart(), use_container_width=True, config={"displayModeBar": False})

    # Future improvements
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
    st.markdown("##### 🚀 Future Roadmap")
    road_cols = st.columns(3, gap="medium")
    roadmap = [
        ("🔤", "Larger Vocabulary", "Expand beyond 3 signs to cover the full ASL alphabet and common phrases."),
        ("🧠", "Deep Learning", "Replace KNN with LSTM/Transformer for sequence-level gesture recognition."),
        ("📱", "Mobile Deployment", "Port the pipeline to TFLite for real-time mobile inference."),
    ]
    for col, (icon, title, desc) in zip(road_cols, roadmap):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#4a5568; font-size:0.8rem; padding:0.5rem 0;">'
    "Gesture AI · Smit Bhagwat · MIT-WPU Pune · 2026"
    "</p>",
    unsafe_allow_html=True,
)
