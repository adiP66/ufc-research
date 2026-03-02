import pandas as pd
import numpy as np
import streamlit as st
import boto3
import sys
import os
import re
import requests
from bs4 import BeautifulSoup
import autogluon.common.utils.utils as ag_utils
import logging

# --- MONKEY PATCH AUTOGLUON ENVIRONMENT ERROR ---
# Fixes AttributeError: 'NoneType' object has no attribute 'lower' in get_package_versions
def safe_get_package_versions():
    import importlib.metadata
    versions = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                versions[name.lower()] = dist.version
        except Exception:
            pass
    return versions

ag_utils.get_package_versions = safe_get_package_versions

from autogluon.tabular import TabularPredictor

# --- PATH SETUP ---
sys.path.append(os.path.abspath(r'd:/Python/ufc-scraper/azure_job_package'))
from production_feature_pipeline_v2 import get_latest_fighter_stats, compute_features_for_matchup

# --- CONFIG ---
st.set_page_config(page_title="UFC AI Predictor", layout='centered')
# BUCKET_NAME = 'mma-info-bucket'  # Disabled S3
FILE_NAME = 'ufc_fights_ml_updated.csv'
# NEW: Point to the Kaggle-verified 2025 model (Unzipped)
MODEL_PATH = r'd:\Python\ufc-scraper\deployed_models\autogluon_wfv_2025'
CACHE_PATH = 'latest_fighter_stats.pkl'
PLACEHOLDER_PHOTO_URL = 'placeholder.jpg'

# --- DATA LOADING ---
@st.cache_resource
def load_resources():
    """
    Loads data and model. Uses persistent disk caching for the feature pipeline 
    to avoid re-calculating history (30s+) on every app restart.
    """
    with st.spinner("Initializing AI Core..."):
        # 1. Load Data
        try:
            raw_df = pd.read_csv(FILE_NAME)
            raw_df['event_date'] = pd.to_datetime(raw_df['event_date'])
        except Exception as e:
            st.error(f"Failed to load data from {FILE_NAME}: {e}")
            st.stop()

        # 2. Run Pipeline (or load cache)
        if os.path.exists(CACHE_PATH):
            # print(f"Loading cached stats from {CACHE_PATH}...")
            latest_stats_df = pd.read_pickle(CACHE_PATH)
        else:
            # print("Computing latest stats (this takes ~30s)...")
            latest_stats_df = get_latest_fighter_stats(raw_df)
            latest_stats_df.to_pickle(CACHE_PATH)

        # 3. Load Model
        try:
            predictor = TabularPredictor.load(MODEL_PATH, require_py_version_match=False)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
            
        return raw_df, latest_stats_df, predictor

# Load globals
RAW_DF, LATEST_STATS, PREDICTOR = load_resources()
ALL_FIGHTERS = sorted(LATEST_STATS.index.unique().tolist())

# --- HELPER FUNCTIONS ---

def find_closest_fighter_name(input_name):
    from thefuzz import process
    if not input_name:
        return None
    result = process.extractOne(input_name, ALL_FIGHTERS)
    return result[0] if result else None

@st.cache_data(show_spinner=False)
def get_fighter_photo(exact_name):
    """Fetches fighter image from UFC.com."""
    name_clean = re.sub(r'[^a-z0-9\s-]', '', exact_name.lower()).split()
    final_name = '-'.join(name_clean)
    url = f"https://ufc.com/athlete/{final_name}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        resp = requests.get(url, timeout=5, headers=headers)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            img = soup.find('img', alt=exact_name)
            if img and 'src' in img.attrs:
                return img['src']
    except Exception:
        pass
    
    return PLACEHOLDER_PHOTO_URL

def predict_matchup(fighter_a, fighter_b):
    """
    Predicts winner between A and B using the latest stats + AutoGluon model.
    Enforces symmetry: P(A wins) = (P(A vs B) + (1 - P(B vs A))) / 2
    """
    try:
        # 1. Feature Vector A vs B
        feat_a = compute_features_for_matchup(fighter_a, fighter_b, LATEST_STATS)
        
        # 2. Feature Vector B vs A
        feat_b = compute_features_for_matchup(fighter_b, fighter_a, LATEST_STATS)
        
        # 3. Predict Both
        # Model class 1 = Winner (usually Fighter A in training context)
        p_a = PREDICTOR.predict_proba(feat_a).iloc[0][1]
        p_b = PREDICTOR.predict_proba(feat_b).iloc[0][1]
        
        # 4. Average
        # If model is perfectly symmetric, p_a == 1 - p_b
        # We enforce it:
        p_final = (p_a + (1.0 - p_b)) / 2.0
        
        return p_final, None

    except Exception as e:
        return None, f"Prediction error: {e}"

def display_tale_of_the_tape(col, name, stats):
    """Displays key stats (Height, Reach, Age, etc)"""
    # Stats dict is just the row from LATEST_STATS
    if stats is None:
        return

    # Helper to safe get
    def val(k, fmt="{:.0f}"):
        v = stats.get(k, np.nan)
        return fmt.format(v) if pd.notnull(v) else "N/A"

    tape_html = f"""
    <div style="font-size: 0.9rem; margin-left: 20px; line-height: 1.6;">
        <strong>Age:</strong> {val('age')}<br>
        <strong>Height:</strong> {val('height')} in<br>
        <strong>Reach:</strong> {val('reach')} in<br>
        <strong>Fights:</strong> {val('total_fights')}<br>
        <strong>Career Win Rate:</strong> {float(stats.get('win_rate',0))*100:.0f}%<br>
    </div>
    """
    col.markdown(tape_html, unsafe_allow_html=True)

# --- UI LAYOUT ---

st.title("UFC AI Predictor 🤖")
st.write("Powered by AutoGluon WeightedEnsemble_L2 (72% Accuracy)")

col1, col2 = st.columns(2)
with col1:
    f1_input = st.selectbox("Fighter 1", ALL_FIGHTERS, index=ALL_FIGHTERS.index('Islam Makhachev') if 'Islam Makhachev' in ALL_FIGHTERS else 0)
with col2:
    f2_input = st.selectbox("Fighter 2", ALL_FIGHTERS, index=ALL_FIGHTERS.index('Charles Oliveira') if 'Charles Oliveira' in ALL_FIGHTERS else 1)

if st.button("Predict Outcome", type="primary"):
    if f1_input == f2_input:
        st.error("Please select two different fighters.")
    else:
        # Predict
        p_win_1, err = predict_matchup(f1_input, f2_input)
        
        if err:
            st.error(err)
        else:
            # Result Logic
            winner = f1_input if p_win_1 > 0.5 else f2_input
            confidence = p_win_1 if p_win_1 > 0.5 else (1 - p_win_1)
            loser = f2_input if winner == f1_input else f1_input
            
            # Photos
            img1 = get_fighter_photo(f1_input)
            img2 = get_fighter_photo(f2_input)
            
            # Stats for display
            stats1 = LATEST_STATS.loc[f1_input]
            stats2 = LATEST_STATS.loc[f2_input]

            st.write("---")
            
            # Columns for fighters area
            c1, cmid, c2 = st.columns([4, 2, 4])
            
            with c1:
                st.image(img1, width=200)
                st.subheader(f1_input)
                if winner == f1_input:
                    st.success(f"**WINNER** ({confidence:.1%})")
                display_tale_of_the_tape(c1, f1_input, stats1)

            with cmid:
                st.markdown("<h1 style='text-align: center; padding-top: 80px;'>VS</h1>", unsafe_allow_html=True)

            with c2:
                st.image(img2, width=200)
                st.subheader(f2_input)
                if winner == f2_input:
                    st.success(f"**WINNER** ({confidence:.1%})")
                display_tale_of_the_tape(c2, f2_input, stats2)
            
            st.write("---")
            
            # Better Probability Bar (Custom CSS)
            p1_pct = int(p_win_1 * 100)
            p2_pct = 100 - p1_pct
            
            st.subheader("Model Confidence")
            
            # Custom HTML Progress Bar
            bar_html = f"""
            <div style="width: 100%; background-color: #f0f2f6; border-radius: 10px; height: 30px; display: flex; overflow: hidden; font-weight: bold; color: white;">
                <div style="width: {p1_pct}%; background-color: #ff4b4b; display: flex; align-items: center; justify-content: center;">
                    {f1_input} ({p1_pct}%)
                </div>
                <div style="width: {p2_pct}%; background-color: #1f77b4; display: flex; align-items: center; justify-content: center;">
                    {f2_input} ({p2_pct}%)
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

            # Strategy Guide (Data-Backed)
            if confidence >= 0.80:
                st.success("🏆 **ELITE TIER:** Historically +25% ROI. (Strong Bet)")
            elif confidence >= 0.75:
                st.info("📈 **SNIPER TIER:** Historically +12% ROI. (Good Bet)")
            elif confidence >= 0.70:
                st.warning("😐 **GRIND TIER:** Historically +3% ROI. (Marginal)")
            else:
                st.error("🛑 **PASS:** Value too low for long-term profit.")

            # --- VISUALIZATIONS ---
            st.write("### 📊 Matchup Analysis")
            
            # Helper to normalize for Radar
            def get_norm(val, col_name):
                # Approximate max values for normalization based on domain knowledge
                max_vals = {
                    'sig_strikes_landed_per_min_dec_avg': 8.0,
                    'takedowns_landed_per_fight_dec_avg': 5.0,
                    'submission_attempts_per_fight_dec_avg': 2.0,
                    'sig_strike_defense_dec_avg': 1.0, # percent
                    'takedown_defense_dec_avg': 1.0,   # percent
                    'win_rate': 1.0,
                    'sos_ewm': 20.0 # Strength of schedule
                }
                mv = max_vals.get(col_name, 10.0)
                v = float(val) if pd.notnull(val) else 0.0
                return min(v / mv, 1.0)

            # 1. Radar Chart Data
            radar_metrics = {
                'Striking Vol': 'sig_strikes_landed_per_min_dec_avg',
                'Grappling': 'takedowns_landed_per_fight_dec_avg',
                'Subs': 'submission_attempts_per_fight_dec_avg',
                'Strike Def': 'sig_strike_defense_dec_avg',
                'TD Def': 'takedown_defense_dec_avg',
                'Win Rate': 'win_rate'
            }
            
            # Prepare data
            r1 = [get_norm(stats1.get(c), c) for c in radar_metrics.values()]
            r2 = [get_norm(stats2.get(c), c) for c in radar_metrics.values()]
            
            # Raw values for tooltip
            raw1 = [stats1.get(c, 0) for c in radar_metrics.values()]
            raw2 = [stats2.get(c, 0) for c in radar_metrics.values()]
            
            cats = list(radar_metrics.keys())
            
            # Close the loop
            r1 += [r1[0]]
            r2 += [r2[0]]
            raw1 += [raw1[0]]
            raw2 += [raw2[0]]
            cats += [cats[0]]

            import plotly.graph_objects as go
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r1, 
                theta=cats, 
                fill='toself', 
                name=f1_input, 
                line_color='#ff4b4b',
                customdata=raw1,
                hovertemplate='%{theta}: <b>%{customdata:.2f}</b><extra></extra>'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=r2, 
                theta=cats, 
                fill='toself', 
                name=f2_input, 
                line_color='#1f77b4',
                customdata=raw2,
                hovertemplate='%{theta}: <b>%{customdata:.2f}</b><extra></extra>'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 1])), # Hide radial numbers (0.2, 0.4...) as they are normalized
                showlegend=True,
                title="Fighter Comparison (Normalized)",
                height=400
            )

            # 2. Key Advantages Data (Bar Chart)
            # We explicitly look at the diffs for high-impact features
            key_diffs = {
                'Reach Adv': (stats1.get('reach',0) - stats2.get('reach',0)),
                'Age Adv (Younger)': (stats2.get('age',30) - stats1.get('age',30)), # Positive means F1 is younger
                'Striking Acc Adv': (stats1.get('sig_strike_accuracy_dec_avg',0) - stats2.get('sig_strike_accuracy_dec_avg',0)),
                'Defense Adv': (stats1.get('sig_strike_defense_dec_avg',0) - stats2.get('sig_strike_defense_dec_avg',0)),
                'Grappling Adv': (stats1.get('takedowns_landed_per_fight_dec_avg',0) - stats2.get('takedowns_landed_per_fight_dec_avg',0)),
                 'SOS (Exp) Adv': (stats1.get('sos_ewm',0) - stats2.get('sos_ewm',0))
            }
            
            adv_labels = list(key_diffs.keys())
            adv_vals = list(key_diffs.values())
            colors = ['#ff4b4b' if v > 0 else '#1f77b4' for v in adv_vals] # Red = F1, Blue = F2 (reverted logic in UI: F1 is typically Red/Left)
            # Actually, let's match the UI bar: F1 is Red (#ff4b4b), F2 is Blue (#1f77b4)
            
            fig_bar = go.Figure(go.Bar(
                x=adv_vals,
                y=adv_labels,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.1f}" for v in adv_vals], # Add + sign
                textposition='auto',
                hovertemplate='%{y}: <b>%{text}</b><extra></extra>'
            ))
            fig_bar.update_layout(
                title=f"Key Advantages (Positive = {f1_input})",
                xaxis_title="Net Difference",
                height=400
            )

            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.plotly_chart(fig_radar, use_container_width=True)
            with c_chart2:
                st.plotly_chart(fig_bar, use_container_width=True)


# Debug Expander
with st.expander("Model & Features Debug Info"):
    st.write(f"Model ID: WeightedEnsemble_L2")
    st.write(f"Training Accuracy: ~72% (Verified on Test Set)")
    st.write("This model uses 120+ features including:", 
             "Decayed Performance Averages, Physical Ratios, Momentum, and Opponent-Adjusted Z-Scores.")
