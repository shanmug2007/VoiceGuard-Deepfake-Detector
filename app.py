import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VoiceGuard Forensics", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ VoiceGuard: Deepfake Audio Detector")
st.markdown("""
**System Status:** Active  
**Detection Mode:** Spectral & Signal Forensics  
**Objective:** Differentiate between Human Voice and AI-Generated/Converted Voice.
""")

# --- THE FORENSIC ENGINE ---
def analyze_audio_forensics(audio_path):
    # Load audio (y = audio time series, sr = sampling rate)
    y, sr = librosa.load(audio_path, sr=None) 
    
    ai_score = 0
    evidence = []
    
    # --- TEST 1: FREQUENCY CUTOFF CHECK ---
    # UPDATED: Lowered threshold to 14kHz to accept standard laptop mics as Human
    stft = np.abs(librosa.stft(y))
    energy = np.sum(stft, axis=1)
    
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]
    threshold_idx = np.searchsorted(cumulative_energy, total_energy * 0.99)
    freqs = librosa.fft_frequencies(sr=sr)
    cutoff_freq = freqs[threshold_idx]
    
   # MODIFIED: Increased penalty from 50 to 60 so low-quality AI triggers RED immediately
    if cutoff_freq < 14000:
        ai_score += 60  # <--- CHANGED THIS FROM 50 TO 60
        evidence.append(f"⚠️ **Hard Frequency Cutoff detected at {int(cutoff_freq)}Hz.** (Likely AI/Low-Quality)")
    else:
        evidence.append(f"✅ **Full Frequency Range ({int(cutoff_freq)}Hz).** (Natural)")
    # --- TEST 2: SILENCE PATTERN ANALYSIS ---
    # UPDATED: Made silence check stricter (needs to be near absolute zero to flag as AI)
    rms = librosa.feature.rms(y=y)[0]
    min_silence = np.min(rms)
    
    if min_silence < 0.0000001: 
        ai_score += 30
        evidence.append("⚠️ **Unnatural 'Digital Silence' detected.** (Lack of room tone)")
    else:
        evidence.append("✅ **Natural Background Noise detected.** (Room tone present)")

    # --- TEST 3: DISPERSION / JITTER ---
    zero_crossings = librosa.zero_crossings(y, pad=False)
    zc_variation = np.var(zero_crossings)
    
    if zc_variation < 0.02:
        ai_score += 20
        evidence.append("⚠️ **Signal is too smooth.** (Lacks organic jitter)")
    
    return min(ai_score, 100), evidence, y, sr

# --- USER INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Upload your audio sample to run forensic analysis.")
    uploaded_file = st.file_uploader("Upload Audio (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save file temporarily
    with open("temp_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Play Audio
    st.audio("temp_file.wav")
    
    if st.button("🔍 ANALYZE AUDIO SOURCE"):
        with st.spinner("Running Spectral Analysis..."):
            score, evidence_list, y, sr = analyze_audio_forensics("temp_file.wav")
            
            # --- SHOW VERDICT ---
            st.divider()
            # If score is high (>50), it's AI. If low, it's Human.
            if score > 50:
                st.error(f"🚨 **VERDICT: AI / SYNTHETIC VOICE** (Confidence: {score}%)")
            else:
                st.success(f"✅ **VERDICT: HUMAN / NATURAL VOICE** (Confidence: {100-score}%)")
            
            # --- DETAILS & GRAPHS ---
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📋 Forensic Evidence")
                for item in evidence_list:
                    st.write(item)
            
            with c2:
                st.subheader("📊 Spectrogram Analysis")
                # Plot Spectrogram
                fig, ax = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set_title("Frequency Heatmap")
                st.pyplot

