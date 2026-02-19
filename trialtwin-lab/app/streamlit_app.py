"""TrialTwin Lab — Interactive Dashboard.

Loads pipeline artifacts and provides two views:
  1. Patient Explorer: individual patient digital twin analysis
  2. Trial Dashboard: aggregate trial-level results and efficiency
"""

import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
ARTIFACT_DIR = BASE_DIR / "data" / "outputs"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

st.set_page_config(
    page_title="TrialTwin Lab",
    page_icon="🧬",
    layout="wide",
)


@st.cache_data
def load_artifacts(trial_id: str) -> dict:
    path = ARTIFACT_DIR / trial_id / "artifacts.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    st.title("TrialTwin Lab")
    st.caption("Digital twins for oncology clinical trials")

    # Discover available trials
    trial_dirs = sorted([
        d.name for d in ARTIFACT_DIR.iterdir()
        if d.is_dir() and (d / "artifacts.pkl").exists()
    ])

    if not trial_dirs:
        st.error("No artifacts found. Run `make data && make train` first.")
        return

    trial_id = st.sidebar.selectbox("Trial", trial_dirs)
    artifacts = load_artifacts(trial_id)

    tab1, tab2 = st.tabs(["Patient Explorer", "Trial Dashboard"])

    with tab1:
        from app.patient_explorer import render_patient_explorer
        render_patient_explorer(artifacts)

    with tab2:
        from app.trial_dashboard import render_trial_dashboard
        render_trial_dashboard(artifacts, REPORT_DIR)


if __name__ == "__main__":
    main()
