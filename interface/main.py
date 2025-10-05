import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import io
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Transformer & preprocessing
from transformer import TimeSeriesTransformer, preprocess_flux_array, SEQ_LEN, DEVICE

# Optional libraries
try:
    import lightkurve as lk
except:
    lk = None

try:
    from astroquery.mast import Observations
except:
    Observations = None

st.set_page_config(page_title="Exoplanet Transit Prediction", layout="wide")
st.title("🔭 Exoplanet Transit Prediction")
st.write("Upload your own FITS/CSV file or fetch Kepler/TESS light curves automatically.")

# ------------------------
# Model loading
# ------------------------
@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state", checkpoint)
    model = TimeSeriesTransformer()
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model("interface/models/synth_model.pt")  # update if needed

# ------------------------
# Helpers
# ------------------------
def safe_name(name: str):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


def read_fits_flux(file):
    """Safely read flux data from FITS file or Streamlit UploadedFile."""
    try:
        if hasattr(file, "read"):
            original_bytes = file.read()
        else:
            with open(file, "rb") as f:
                original_bytes = f.read()

        # --- Attempt 1: Try Lightkurve ---
        if lk is not None:
            try:
                lc = lk.read(io.BytesIO(original_bytes))

                # Ensure flux is numeric and unmasked
                flux = lc.flux
                if hasattr(flux, "value"):
                    flux = flux.value
                if hasattr(flux, "mask"):
                    flux = np.ma.filled(flux, np.nan)
                flux = np.array(flux, dtype=np.float32)

                flux = flux[np.isfinite(flux)]  # remove NaN/inf
                if len(flux) > 10:
                    return flux
            except Exception as e:
                print(f"Lightkurve failed: {e}")

        # --- Attempt 2: Manual FITS parse ---
        with fits.open(io.BytesIO(original_bytes), memmap=False) as hdul:
            for hdu in hdul:
                if not hasattr(hdu, "data") or hdu.data is None:
                    continue
                if hasattr(hdu.data, "dtype") and getattr(hdu.data.dtype, "names", None):
                    names = hdu.data.dtype.names
                    flux_cols = [n for n in names if "flux" in n.lower()]
                    if flux_cols:
                        data = hdu.data[flux_cols[0]]
                        data = np.ma.filled(data, np.nan)  # fill masked
                        flux = np.array(data, dtype=np.float32)
                        flux = flux[np.isfinite(flux)]
                        if len(flux) > 10:
                            return flux
                elif isinstance(hdu.data, np.ndarray):
                    arr = np.ma.filled(hdu.data, np.nan)
                    arr = np.array(arr, dtype=np.float32)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 10:
                        return arr
        return None
    except Exception as e:
        print(f"Error reading FITS: {e}")
        return None


def predict_flux_from_array(arr):
    """Predict directly from flux array (already numeric)."""
    processed = preprocess_flux_array(arr, target_len=SEQ_LEN)
    if processed is None:
        return None, None, None
    x = torch.tensor(processed, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs, processed


# ------------------------
# Streamlit Interface
# ------------------------
tab1, tab2, tab3 = st.tabs(["🌐 Fetch from Mission", "📤 Upload Your Own File", "ℹ️ About / Model Info"])

with tab1:
    target_name = st.text_input("Target name (KIC/TIC/star)", "Kepler-10")
    mission = st.selectbox("Mission", ["Kepler", "TESS"])
    max_files = st.number_input("Max light curves to fetch", min_value=1, max_value=10, value=3)

    if st.button("Download & Predict"):
        def download_lightkurve(target, mission="Kepler", max_files=3, out_dir="downloads"):
            if lk is None:
                st.warning("Lightkurve not installed.")
                return []
            search = lk.search_lightcurvefile(target, mission=mission)
            if len(search) == 0:
                st.warning(f"No {mission} light curves found for {target}.")
                return []
            selection = search[:max_files]
            out_dir = Path(out_dir) / safe_name(target)
            out_dir.mkdir(parents=True, exist_ok=True)
            _ = selection.download_all(download_dir=str(out_dir))
            return sorted(out_dir.rglob("*.fits"))[:max_files]

        fits_files = download_lightkurve(target_name, mission, max_files)
        if not fits_files:
            st.error("No light curves found.")
        else:
            st.success(f"Found {len(fits_files)} FITS files.")
            for i, fp in enumerate(fits_files, 1):
                flux = read_fits_flux(fp)
                if flux is None:
                    st.warning(f"Skipping {fp} (could not read flux).")
                    continue

                pred, probs, processed = predict_flux_from_array(flux)
                if pred is None:
                    st.warning(f"Could not process {fp}.")
                    continue

                prob_no, prob_yes = f"{probs[0]*100:.2f}%", f"{probs[1]*100:.2f}%"
                result_text = "🌍 **Exoplanet transit likely detected!**" if pred == 1 else "✨ **No exoplanet transit detected.**"

                st.markdown(f"### File {i}: `{Path(fp).name}`")
                st.markdown(result_text)
                st.markdown(f"**Probabilities:** No Transit: {prob_no} | Transit: {prob_yes}")

                fig, ax = plt.subplots()
                ax.plot(processed, color="blue")
                ax.set_xlabel("Time (resampled)")
                ax.set_ylabel("Normalized Flux")
                st.pyplot(fig)

                st.divider()
                st.markdown(f"#### 🌌 Exoplanet Summary for `{target_name}`")
                st.markdown(f"""
                - **Target**: {target_name}
                - **Mission**: {mission}
                - **Files analyzed**: {i}
                - **Transit Detection**: {"Yes" if pred == 1 else "No"}
                - **Confidence**: {prob_yes if pred == 1 else prob_no}
                """)

with tab2:
    uploaded_file = st.file_uploader("Upload FITS, CSV, TXT, or NPY file", type=["fits", "csv", "txt", "npy"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".fits"):
                flux = read_fits_flux(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                flux = np.loadtxt(uploaded_file, delimiter=",")
            elif uploaded_file.name.endswith(".txt"):
                flux = np.loadtxt(uploaded_file)
            elif uploaded_file.name.endswith(".npy"):
                flux = np.load(uploaded_file)
            else:
                flux = None

            if flux is None or len(flux) == 0:
                st.error("Could not process uploaded file. Please check format or content.")
            else:
                pred, probs, processed = predict_flux_from_array(flux)
                if pred is None:
                    st.error("Prediction failed. Please check your file's format.")
                else:
                    prob_no, prob_yes = f"{probs[0]*100:.2f}%", f"{probs[1]*100:.2f}%"
                    result_text = "🌍 **Exoplanet transit likely detected!**" if pred == 1 else "✨ **No exoplanet transit detected.**"

                    st.success(result_text)
                    st.markdown(f"**Probabilities:** No Transit: {prob_no} | Transit: {prob_yes}")

                    fig, ax = plt.subplots()
                    ax.plot(processed, color="green")
                    ax.set_xlabel("Time (resampled)")
                    ax.set_ylabel("Normalized Flux")
                    st.pyplot(fig)

                    st.divider()
                    st.markdown(f"#### 🌌 Exoplanet Summary")
                    st.markdown(f"""
                    - **File Name**: {uploaded_file.name}
                    - **Transit Detected**: {"Yes" if pred == 1 else "No"}
                    - **Confidence**: {prob_yes if pred == 1 else prob_no}
                    - **Model Used**: `synth_model.pt`
                    """)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
                    
with tab3:
    st.header("About This Exoplanet Transit Prediction App")
    st.markdown("""
    This application uses a **Transformer-based deep learning model** to detect possible exoplanet transits
    from light curves obtained by **Kepler** and **TESS** missions.

    ### How the Model Works
    - The model is a **TimeSeriesTransformer**, trained on normalized flux arrays.
    - It predicts whether a segment of a star's light curve shows signs of an **exoplanet transit**.
    - Inputs can be **FITS files, CSV, TXT, or NPY arrays** of light curves.
    - The model outputs a **binary classification** (Transit / No Transit) along with confidence probabilities.

    ### Usage Instructions
    1. **Fetch from Mission**: Enter the target star's ID (KIC/TIC) and mission, then download light curves automatically.
    2. **Upload Your Own File**: Drag and drop your FITS/CSV/TXT/NPY file. Ensure it contains flux values over time.
    3. **View Results**: For each light curve, the app plots the flux, shows prediction probabilities, and gives a summary.
    
    ### Model Details
    - **Model file**: `synth_model.pt`
    - **Sequence length used for training**: {SEQ_LEN}
    - **Device used**: {DEVICE}
    
    ### Notes
    - Make sure your uploaded file contains valid numerical flux data.
    - For large FITS files, it might take a few seconds to parse and predict.
    - The model is **pre-trained on synthetic and curated light curves** for demonstration purposes.

    """)

