# ═══════════════════════════════════════════════════════════════
# FINAL TRAIN + INFERENCE PIPELINE WITH WINDOW/TRIAL ACCURACY
#   - Train final LDA on Emognition
#   - Inference on Participant 1 ... Participant N folders
#   - Uses only BVP from Samsung watch
#   - HR/PPI are derived from BVP
#   - MUSE band powers are resampled to true 10 Hz
#   - Reports BOTH window-level and trial-level accuracy
# ═══════════════════════════════════════════════════════════════

import os
import json
import glob
import time
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from scipy.signal import welch, coherence as sp_coherence, find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
# ---- TRAINING DATA (Emognition) ----
PROCESSED_EEG_ROOT = "/kaggle/input/datasets/sasinduabewickrema/emognition-processed/Emognition Processed"
RAW_WATCH_ROOT     = "/kaggle/input/emognition"
RAW_MUSE_ROOT      = "/kaggle/input/emognition"

# ---- INFERENCE DATA (your participant folders) ----
INFER_ROOT = "/kaggle/input/datasets/sasinduabewickrema/participants-data-2/Study Data"

# ---- OUTPUT ----
OUT_TRIAL_CSV  = "/kaggle/working/inference_trial_predictions.csv"
OUT_WINDOW_CSV = "/kaggle/working/inference_window_predictions.csv"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
EEG_SR = 256

# True band power update rate
BAND_SR = 10

# Training JSON band powers are stored on 256 Hz grid with duplicates
TRAIN_BAND_STORED_SR = 256

# Training Samsung watch BVP sampling rate
TRAIN_BVP_SR = 20

WINDOW_SEC = 10
OVERLAP_FRAC = 0.75

EEG_WIN  = WINDOW_SEC * EEG_SR
BAND_WIN = WINDOW_SEC * BAND_SR
TRAIN_BVP_WIN = int(WINDOW_SEC * TRAIN_BVP_SR)

NUM_CLASSES = 4

EMOTION_LABELS = {
    "NEUTRAL": 0,
    "ENTHUSIASM": 1,
    "SADNESS": 2,
    "FEAR": 3,
}
IDX_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}

EEG_CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

BAND_CHANNELS = [
    "Alpha_TP9",  "Alpha_AF7",  "Alpha_AF8",  "Alpha_TP10",
    "Beta_TP9",   "Beta_AF7",   "Beta_AF8",   "Beta_TP10",
    "Delta_TP9",  "Delta_AF7",  "Delta_AF8",  "Delta_TP10",
    "Gamma_TP9",  "Gamma_AF7",  "Gamma_AF8",  "Gamma_TP10",
    "Theta_TP9",  "Theta_AF7",  "Theta_AF8",  "Theta_TP10",
]

# Feature sizes
N_FEAT_EEG  = 156
N_FEAT_MUSE = 62
N_FEAT_BVP  = 7
N_FEAT_HR   = 5
N_FEAT_PPI  = 8
N_FEATURES_RAW = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI

print(f"Raw feature count = {N_FEATURES_RAW}")

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_array(x):
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)

def infer_sampling_rate_from_time_series(t):
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return 1.0 / np.median(dt)

def resample_1d_by_time(arr, orig_sr, target_sr):
    arr = safe_array(np.asarray(arr, dtype=np.float32))
    if len(arr) == 0:
        return arr.astype(np.float32)

    old_t = np.arange(len(arr), dtype=np.float64) / float(orig_sr)
    new_len = int(np.floor(len(arr) * float(target_sr) / float(orig_sr)))
    new_len = max(new_len, 1)
    new_t = np.arange(new_len, dtype=np.float64) / float(target_sr)

    out = np.interp(new_t, old_t, arr)
    return out.astype(np.float32)

def resample_multich_by_time(arr2d, orig_sr, target_sr):
    return np.vstack([
        resample_1d_by_time(arr2d[i], orig_sr, target_sr)
        for i in range(arr2d.shape[0])
    ])

def parse_emotion_from_training_trial_name(trial_name):
    parts = trial_name.split("_")
    if len(parts) < 2:
        return None
    emo = parts[1].upper()
    return emo if emo in EMOTION_LABELS else None

def parse_trial_key_inference(base_name):
    if base_name.endswith("_eeg.csv"):
        return base_name[:-8]
    if base_name.endswith("_ppg_hr_ibi.csv"):
        return base_name[:-15]
    return os.path.splitext(base_name)[0]

def parse_true_label_from_infer_trial_key(trial_key):
    low = trial_key.lower()
    if low.startswith("neutral"):
        return "NEUTRAL"
    if low.startswith("enthusiasm"):
        return "ENTHUSIASM"
    if low.startswith("sad"):
        return "SADNESS"
    if low.startswith("fear"):
        return "FEAR"
    return None

def trial_vote_from_probs(probs, num_classes=NUM_CLASSES):
    mean_prob = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(mean_prob))
    conf = float(mean_prob[pred_idx])
    return pred_idx, conf, mean_prob

# ═══════════════════════════════════════════════════════════════
# BVP -> HR / PPI
# ═══════════════════════════════════════════════════════════════
def bandpass_bvp(sig, sr, low=0.7, high=3.5, order=2):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < max(10, order * 3):
        return sig

    nyq = 0.5 * sr
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.999)

    if low_n >= high_n:
        return sig

    try:
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def derive_hr_ppi_from_bvp(bvp_window, sr):
    sig = safe_array(np.asarray(bvp_window, dtype=np.float64))
    if len(sig) < 8:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    sig_f = bandpass_bvp(sig, sr=sr, low=0.7, high=3.5, order=2)

    sstd = np.std(sig_f)
    if sstd > 1e-10:
        sig_n = (sig_f - np.mean(sig_f)) / sstd
    else:
        sig_n = sig_f - np.mean(sig_f)

    min_dist = max(1, int(0.35 * sr))

    try:
        peaks, _ = find_peaks(sig_n, distance=min_dist, prominence=0.2)
    except Exception:
        peaks = np.array([], dtype=int)

    if len(peaks) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    ibi_sec = np.diff(peaks) / float(sr)
    ibi_sec = ibi_sec[(ibi_sec >= 0.35) & (ibi_sec <= 1.5)]

    if len(ibi_sec) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    ppi_vals_ms = (ibi_sec * 1000.0).astype(np.float32)
    hr_vals_bpm = (60.0 / ibi_sec).astype(np.float32)

    return hr_vals_bpm, ppi_vals_ms

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
FREQ_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
CH_PAIRS = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def _compute_psd(sig, sr, nperseg=256):
    return welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))

def _band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return np.mean(psd[mask]) if mask.any() else 1e-10

def _differential_entropy(sig):
    v = np.var(sig)
    return 0.5 * np.log(2 * np.pi * np.e * v) if v > 1e-12 else 0.0

def _hjorth(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    act = np.var(sig)
    if act < 1e-12:
        return 0.0, 0.0, 0.0
    mob = np.sqrt(np.var(d1) / act)
    v_d1 = np.var(d1)
    if v_d1 < 1e-12:
        return float(act), float(mob), 0.0
    comp = np.sqrt(np.var(d2) / v_d1) / mob if mob > 1e-12 else 0.0
    return float(act), float(mob), float(comp)

def _zcr(sig):
    return float(np.sum(np.abs(np.diff(np.sign(sig))) > 0)) / max(len(sig) - 1, 1)

def spectral_entropy(sig, sr=EEG_SR, nperseg=256):
    _, psd = welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))
    psd_n = psd / (psd.sum() + 1e-12)
    psd_n = psd_n[psd_n > 0]
    return float(-np.sum(psd_n * np.log2(psd_n))) if len(psd_n) > 0 else 0.0

def permutation_entropy(sig, order=3, delay=1):
    n = len(sig)
    if n < (order - 1) * delay + 1:
        return 0.0
    indices = np.arange(n - (order - 1) * delay)
    cols = np.column_stack([sig[indices + d * delay] for d in range(order)])
    perms = np.argsort(cols, axis=1)
    encoded = np.zeros(perms.shape[0], dtype=np.int64)
    for i in range(order):
        encoded = encoded * order + perms[:, i]
    _, counts = np.unique(encoded, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def wavelet_subband_energy(sig, wavelet="db4", level=5):
    try:
        import pywt
        max_lev = pywt.dwt_max_level(len(sig), wavelet)
        coeffs = pywt.wavedec(sig, wavelet, level=min(level, max_lev))
        energies = [float(np.mean(c**2)) for c in coeffs[:5]]
        while len(energies) < 5:
            energies.append(0.0)
        return energies
    except Exception:
        return [0.0] * 5

# 156 EEG features
def extract_eeg_features(eeg_4ch, sr=EEG_SR):
    feats = []
    bp = np.zeros((4, 5), dtype=np.float64)

    for ch in range(4):
        sig = eeg_4ch[ch].astype(np.float64)
        freqs, psd = _compute_psd(sig, sr)

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw = _band_power(freqs, psd, lo, hi)
            bp[ch, bi] = np.log1p(pw)
        feats.extend(bp[ch].tolist())

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw = _band_power(freqs, psd, lo, hi)
            feats.append(0.5 * np.log(2 * np.pi * np.e * pw) if pw > 1e-12 else 0.0)

        feats.extend(_hjorth(sig))
        feats.extend([
            float(np.mean(sig)),
            float(np.std(sig)),
            float(skew(sig)),
            float(kurtosis(sig))
        ])
        feats.append(_zcr(sig))
        feats.append(_differential_entropy(sig))
        feats.append(spectral_entropy(sig, sr))
        feats.append(permutation_entropy(sig))

    for ch in range(4):
        feats.extend(wavelet_subband_energy(eeg_4ch[ch].astype(np.float64)))

    for ch in range(4):
        a, b, t = bp[ch, 2], bp[ch, 3], bp[ch, 1]
        feats.extend([a - b, a - t, t - b])

    for bi in range(5):
        feats.append(bp[2, bi] - bp[1, bi])
    for bi in range(5):
        feats.append(bp[3, bi] - bp[0, bi])

    for (ci, cj) in CH_PAIRS:
        try:
            f_coh, coh = sp_coherence(
                eeg_4ch[ci].astype(np.float64),
                eeg_4ch[cj].astype(np.float64),
                fs=sr,
                nperseg=min(256, len(eeg_4ch[ci]))
            )
            for bi, (lo, hi) in enumerate(FREQ_BANDS):
                mask = (f_coh >= lo) & (f_coh < hi)
                feats.append(float(np.mean(coh[mask])) if mask.any() else 0.0)
        except Exception:
            feats.extend([0.0] * 5)

    return safe_array(np.array(feats, dtype=np.float32))

# 62 band features
def extract_band_features(band_window):
    feats = []

    for ch in range(20):
        col = band_window[ch]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            feats.extend([0.0, 0.0])
        else:
            feats.append(float(np.mean(col)))
            feats.append(float(np.std(col)) + 1e-8)

    def bp(bi, ei):
        col = band_window[bi * 4 + ei]
        col = col[np.isfinite(col)]
        return float(np.mean(col)) if len(col) > 0 else 0.0

    for ei in range(4):
        a, b, t = bp(0, ei), bp(1, ei), bp(4, ei)
        db = b if abs(b) > 1e-6 else 1e-6
        dt = t if abs(t) > 1e-6 else 1e-6
        feats.extend([a / db, a / dt, t / db])

    for bi in range(5):
        feats.append(bp(bi, 2) - bp(bi, 1))
    for bi in range(5):
        feats.append(bp(bi, 3) - bp(bi, 0))

    return safe_array(np.clip(np.array(feats, dtype=np.float32), -1e4, 1e4))

# 7 BVP features
def extract_bvp_features(bvp_window, sr):
    sig = bvp_window.astype(np.float64)
    feats = [
        float(np.mean(sig)),
        float(np.std(sig)),
        float(skew(sig)),
        float(kurtosis(sig))
    ]

    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    mask_hr = (freqs > 0.5) & (freqs < 4.0)

    feats.append(
        float(freqs[mask_hr][np.argmax(fft_mag[mask_hr])])
        if mask_hr.any() and fft_mag[mask_hr].max() > 0 else 0.0
    )
    feats.append(_zcr(sig))
    feats.append(
        float(np.sum(freqs * fft_mag) / fft_mag.sum())
        if len(fft_mag) > 1 and fft_mag.sum() > 0 else 0.0
    )

    return safe_array(np.array(feats, dtype=np.float32))

# 5 HR features
def extract_hr_features(hr_values):
    if len(hr_values) < 2:
        return np.zeros(5, dtype=np.float32)

    hr = hr_values.astype(np.float64)
    return safe_array(np.array([
        float(np.mean(hr)),
        float(np.std(hr)),
        float(np.min(hr)),
        float(np.max(hr)),
        float(np.max(hr) - np.min(hr))
    ], dtype=np.float32))

# 8 PPI features
def extract_ppi_features(ppi_values):
    if len(ppi_values) < 3:
        return np.zeros(8, dtype=np.float32)

    ipi = ppi_values.astype(np.float64)
    ipi_s = ipi / 1000.0 if np.median(ipi) > 10 else ipi.copy()

    feats = [float(np.mean(ipi_s)), float(np.std(ipi_s))]
    sd = np.diff(ipi_s)

    feats.append(float(np.sqrt(np.mean(sd**2))) if len(sd) > 0 else 0.0)
    feats.append(float(np.mean(np.abs(sd) > 0.05)) if len(sd) > 0 else 0.0)
    feats.append(float(np.std(sd)) if len(sd) > 0 else 0.0)

    if len(ipi_s) > 6:
        t_ipi = np.cumsum(ipi_s)
        t_uniform = np.arange(t_ipi[0], t_ipi[-1], 0.25)
        if len(t_uniform) > 8:
            ipi_uniform = np.interp(t_uniform, t_ipi, ipi_s)
            f_ipi, psd_ipi = welch(ipi_uniform, fs=4.0, nperseg=min(len(ipi_uniform), 32))
            lf_m = (f_ipi >= 0.04) & (f_ipi < 0.15)
            hf_m = (f_ipi >= 0.15) & (f_ipi < 0.4)
            lf = float(np.mean(psd_ipi[lf_m])) if lf_m.any() else 0.0
            hf = float(np.mean(psd_ipi[hf_m])) if hf_m.any() else 0.0
            feats.extend([lf, hf, lf / hf if hf > 1e-10 else 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0])

    return safe_array(np.array(feats, dtype=np.float32))

# ═══════════════════════════════════════════════════════════════
# LOAD TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\nLoading training trials from Emognition ...")
t0 = time.time()

train_trials = []
skipped = defaultdict(int)

for subj in sorted(os.listdir(PROCESSED_EEG_ROOT)):
    subj_dir = os.path.join(PROCESSED_EEG_ROOT, subj)
    if not os.path.isdir(subj_dir) or not subj.isdigit():
        continue

    for trial_name in sorted(os.listdir(subj_dir)):
        trial_path = os.path.join(subj_dir, trial_name)
        if not os.path.isdir(trial_path):
            continue

        emotion = parse_emotion_from_training_trial_name(trial_name)
        if emotion is None:
            skipped["bad_trial_name"] += 1
            continue

        label_idx = EMOTION_LABELS[emotion]
        sid = subj

        eeg_json = os.path.join(trial_path, trial_name + ".json")
        if not os.path.isfile(eeg_json):
            skipped["missing_eeg_json"] += 1
            continue

        try:
            with open(eeg_json, "r") as f:
                ed = json.load(f)
            eeg = np.stack([np.array(ed[ch], dtype=np.float32) for ch in EEG_CHANNELS])
            eeg = safe_array(eeg)
        except Exception:
            skipped["bad_eeg_json"] += 1
            continue

        T_eeg = eeg.shape[1]

        muse_json = os.path.join(RAW_MUSE_ROOT, sid, f"{sid}_{emotion}_STIMULUS_MUSE.json")
        if not os.path.isfile(muse_json):
            skipped["missing_muse_json"] += 1
            continue

        try:
            with open(muse_json, "r") as f:
                md = json.load(f)

            band_list = []
            for bch in BAND_CHANNELS:
                raw_band = safe_array(np.array(md[bch], dtype=np.float32))
                true_band = resample_1d_by_time(raw_band, TRAIN_BAND_STORED_SR, BAND_SR)
                band_list.append(true_band)

            min_band = min(len(x) for x in band_list)
            band_arr = np.stack([x[:min_band] for x in band_list], axis=0)
        except Exception:
            skipped["bad_muse_json"] += 1
            continue

        sw_json = os.path.join(RAW_WATCH_ROOT, sid, f"{sid}_{emotion}_STIMULUS_SAMSUNG_WATCH.json")
        if not os.path.isfile(sw_json):
            skipped["missing_watch_json"] += 1
            continue

        try:
            with open(sw_json, "r") as f:
                sw = json.load(f)

            if "BVPProcessed" not in sw:
                skipped["missing_BVPProcessed"] += 1
                continue

            bvp_vals = safe_array(np.array([r[1] for r in sw["BVPProcessed"]], dtype=np.float32))
        except Exception:
            skipped["bad_watch_json"] += 1
            continue

        dur = T_eeg / EEG_SR
        dur = min(dur, band_arr.shape[1] / BAND_SR)
        dur = min(dur, len(bvp_vals) / TRAIN_BVP_SR)

        eeg      = eeg[:, :int(dur * EEG_SR)]
        band_arr = band_arr[:, :int(dur * BAND_SR)]
        bvp_vals = bvp_vals[:int(dur * TRAIN_BVP_SR)]

        train_trials.append({
            "sid": sid,
            "emotion": emotion,
            "label": label_idx,
            "eeg": eeg,
            "band": band_arr,
            "bvp": bvp_vals,
            "trial_key": f"{sid}_{emotion}"
        })

print(f"Training trials loaded: {len(train_trials)} in {time.time()-t0:.1f}s")
if skipped:
    print("Skipped:", dict(skipped))

# ═══════════════════════════════════════════════════════════════
# WINDOW TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\nWindowing training trials ...")
t1 = time.time()

step = WINDOW_SEC * (1 - OVERLAP_FRAC)

all_feat_w = []
all_labels_w = []
all_tkeys_w = []
all_sids_w = []
all_tidx_w = []

for ti, tr in enumerate(train_trials):
    sid = tr["sid"]
    lbl = tr["label"]
    eeg = tr["eeg"]
    band = tr["band"]
    bvp = tr["bvp"]
    tkey = tr["trial_key"]

    dur = eeg.shape[1] / EEG_SR
    dur = min(dur, band.shape[1] / BAND_SR)
    dur = min(dur, len(bvp) / TRAIN_BVP_SR)

    n_wins = int((dur - WINDOW_SEC) / step) + 1
    if n_wins <= 0:
        continue

    for wi in range(n_wins):
        t_start = wi * step

        e_s = int(t_start * EEG_SR);       e_e = e_s + EEG_WIN
        m_s = int(t_start * BAND_SR);      m_e = m_s + BAND_WIN
        b_s = int(t_start * TRAIN_BVP_SR); b_e = b_s + TRAIN_BVP_WIN

        if e_e > eeg.shape[1] or m_e > band.shape[1] or b_e > len(bvp):
            break

        ew = eeg[:, e_s:e_e]
        mw = band[:, m_s:m_e]
        bw = bvp[b_s:b_e]

        hr_win, ppi_win = derive_hr_ppi_from_bvp(bw, sr=TRAIN_BVP_SR)

        f_eeg  = extract_eeg_features(ew)
        f_muse = extract_band_features(mw)
        f_bvp  = extract_bvp_features(bw, sr=TRAIN_BVP_SR)
        f_hr   = extract_hr_features(hr_win)
        f_ppi  = extract_ppi_features(ppi_win)

        feat = np.concatenate([f_eeg, f_muse, f_bvp, f_hr, f_ppi]).astype(np.float32)

        all_feat_w.append(feat)
        all_labels_w.append(lbl)
        all_tkeys_w.append(tkey)
        all_sids_w.append(sid)
        all_tidx_w.append(ti)

allF_raw = np.array(all_feat_w, dtype=np.float32)
allY     = np.array(all_labels_w)
allTK    = np.array(all_tkeys_w)
allSID   = np.array(all_sids_w)
allTI    = np.array(all_tidx_w)

print(f"Training windows: {len(allY)} in {time.time()-t1:.1f}s")

# ═══════════════════════════════════════════════════════════════
# PREPROCESS TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\nApplying training preprocessing ...")

vt = VarianceThreshold(threshold=0.001)
allF_sel = vt.fit_transform(allF_raw)
print(f"VarianceThreshold kept {allF_sel.shape[1]} / {N_FEATURES_RAW}")

allF_n = np.empty_like(allF_sel)
train_sid_stats = {}

for sid in sorted(set(allSID)):
    mask = (allSID == sid)
    data = allF_sel[mask]
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    train_sid_stats[sid] = (mu, sd)
    allF_n[mask] = (data - mu) / sd

allF_n = np.clip(safe_array(allF_n), -10, 10)
N_FEATURES = allF_n.shape[1]
print("Final training feature dims:", N_FEATURES)

# ═══════════════════════════════════════════════════════════════
# BALANCED FOLD ASSIGNMENT
# ═══════════════════════════════════════════════════════════════
print("\nBalanced fold assignment for training ...")

labels_arr = np.array([tr["label"] for tr in train_trials])

subj_emo_trial = {}
for ti, tr in enumerate(train_trials):
    sid = tr["sid"]
    lbl = tr["label"]
    subj_emo_trial.setdefault(sid, {})[lbl] = ti

rng_fold = np.random.RandomState(123)
shuffled_subs = sorted(subj_emo_trial.keys())
rng_fold.shuffle(shuffled_subs)

trial_pos = {}
for g_idx, sid in enumerate(shuffled_subs):
    group = g_idx % 4
    for emo_idx in range(4):
        if emo_idx in subj_emo_trial[sid]:
            trial_pos[subj_emo_trial[sid][emo_idx]] = (group + emo_idx) % 4

win_pos = np.array([trial_pos[ti] for ti in allTI])

for fk in range(4):
    te_t = [ti for ti, p in trial_pos.items() if p == fk]
    c = Counter(labels_arr[te_t].tolist())
    print(f"Fold {fk}: test clips={len(te_t)} emotions={dict(sorted(c.items()))}")

# ═══════════════════════════════════════════════════════════════
# PER-FOLD MI
# ═══════════════════════════════════════════════════════════════
print("\nComputing per-fold MI ...")
mi_per_fold = {}

for fk in range(4):
    tr_mask_mi = (win_pos != fk) & (win_pos != (fk + 1) % 4)
    tr_idx_mi = np.where(tr_mask_mi)[0]

    sub = (
        np.random.RandomState(42).choice(tr_idx_mi, 2000, replace=False)
        if len(tr_idx_mi) > 2000 else tr_idx_mi
    )

    mi = mutual_info_classif(allF_n[sub], allY[sub], random_state=42, n_neighbors=5)
    mi_per_fold[fk] = mi

# ═══════════════════════════════════════════════════════════════
# SELECT FINAL LDA HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
print("\nSelecting final LDA hyperparameters ...")

K_GRID = [80, 120, N_FEATURES]
SH_GRID = ['auto', 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

grid_scores = []

for K in K_GRID:
    for sh in SH_GRID:
        fold_scores = []

        for fk in range(4):
            te_mask = (win_pos == fk)
            vl_mask = (win_pos == (fk + 1) % 4)
            tr_mask = ~te_mask & ~vl_mask

            trF = allF_n[tr_mask]
            trY = allY[tr_mask]
            vlF = allF_n[vl_mask]
            vlY = allY[vl_mask]

            mi_ranked = np.argsort(-mi_per_fold[fk])
            fi = mi_ranked[:K]

            try:
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=sh)
                clf.fit(trF[:, fi], trY)
                sc = float((clf.predict(vlF[:, fi]) == vlY).mean())
                fold_scores.append(sc)
            except Exception:
                fold_scores.append(-1.0)

        mean_sc = np.mean(fold_scores)
        grid_scores.append((mean_sc, K, sh))

grid_scores.sort(reverse=True, key=lambda x: x[0])
best_mean_val, FINAL_K, FINAL_SH = grid_scores[0]

print(f"Chosen final params: K={FINAL_K}, shrinkage={FINAL_SH}, mean_val={best_mean_val:.4f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN FINAL MODEL ON ALL TRAINING WINDOWS
# ═══════════════════════════════════════════════════════════════
print("\nTraining final LDA on all training windows ...")

sub_idx = (
    np.random.RandomState(42).choice(len(allF_n), 2000, replace=False)
    if len(allF_n) > 2000 else np.arange(len(allF_n))
)
mi_global = mutual_info_classif(allF_n[sub_idx], allY[sub_idx], random_state=42, n_neighbors=5)
mi_global_ranked = np.argsort(-mi_global)
final_feature_idx = mi_global_ranked[:FINAL_K]

final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=FINAL_SH)
final_clf.fit(allF_n[:, final_feature_idx], allY)

print("Final training complete.")

# ═══════════════════════════════════════════════════════════════
# OPTIONAL TRAINING SANITY REPORT
# ═══════════════════════════════════════════════════════════════
train_pred = final_clf.predict(allF_n[:, final_feature_idx])
print("\nTraining sanity-check report:")
print(classification_report(
    allY,
    train_pred,
    target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
    zero_division=0
))

# ═══════════════════════════════════════════════════════════════
# DISCOVER INFERENCE PARTICIPANT FOLDERS
# ═══════════════════════════════════════════════════════════════
print("\nDiscovering participant folders for inference ...")

participant_dirs = sorted([
    os.path.join(INFER_ROOT, d)
    for d in os.listdir(INFER_ROOT)
    if os.path.isdir(os.path.join(INFER_ROOT, d)) and d.lower().startswith("participant")
])

print(f"Found {len(participant_dirs)} participant folders.")

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════
trial_rows = []
window_rows = []

for pdir in participant_dirs:
    participant_name = os.path.basename(pdir)

    eeg_files = sorted(glob.glob(os.path.join(pdir, "*_eeg.csv")))
    ppg_files = sorted(glob.glob(os.path.join(pdir, "*_ppg_hr_ibi.csv")))

    eeg_map = {parse_trial_key_inference(os.path.basename(fp)): fp for fp in eeg_files}
    ppg_map = {parse_trial_key_inference(os.path.basename(fp)): fp for fp in ppg_files}

    common_trials = sorted(set(eeg_map.keys()) & set(ppg_map.keys()))
    print(f"\n{participant_name}: {len(common_trials)} paired trials")

    participant_trial_windows = {}
    participant_all_windows = []

    # ---- Build features for all trials of one participant ----
    for trial_key in common_trials:
        eeg_path = eeg_map[trial_key]
        ppg_path = ppg_map[trial_key]

        try:
            eeg_df = pd.read_csv(eeg_path)
            ppg_df = pd.read_csv(ppg_path)
        except Exception as e:
            print(f"  Skipping {trial_key}: read error -> {e}")
            continue

        missing_eeg = [c for c in EEG_CHANNELS + BAND_CHANNELS if c not in eeg_df.columns]
        if missing_eeg:
            print(f"  Skipping {trial_key}: missing EEG columns {missing_eeg}")
            continue

        if "ppg_green" not in ppg_df.columns:
            print(f"  Skipping {trial_key}: missing ppg_green")
            continue

        eeg_raw = np.stack(
            [safe_array(eeg_df[ch].values.astype(np.float32)) for ch in EEG_CHANNELS],
            axis=0
        )

        band_raw_256 = np.stack(
            [safe_array(eeg_df[ch].values.astype(np.float32)) for ch in BAND_CHANNELS],
            axis=0
        )

        # Band columns are on EEG grid -> convert to true 10 Hz
        band_true = resample_multich_by_time(band_raw_256, EEG_SR, BAND_SR)

        bvp = safe_array(ppg_df["ppg_green"].values.astype(np.float32))

        # Infer BVP SR from time_s if possible
        if "time_s" in ppg_df.columns:
            infer_bvp_sr = infer_sampling_rate_from_time_series(ppg_df["time_s"].values)
            bvp_sr = float(infer_bvp_sr) if infer_bvp_sr is not None else 25.0
        else:
            bvp_sr = 25.0

        bvp_win = int(round(WINDOW_SEC * bvp_sr))

        dur = eeg_raw.shape[1] / EEG_SR
        dur = min(dur, band_true.shape[1] / BAND_SR)
        dur = min(dur, len(bvp) / bvp_sr)

        eeg_raw = eeg_raw[:, :int(dur * EEG_SR)]
        band_true = band_true[:, :int(dur * BAND_SR)]
        bvp = bvp[:int(dur * bvp_sr)]

        n_wins = int((dur - WINDOW_SEC) / step) + 1
        if n_wins <= 0:
            print(f"  Skipping {trial_key}: shorter than one window")
            continue

        trial_feats = []
        trial_meta = []

        for wi in range(n_wins):
            t_start = wi * step

            e_s = int(t_start * EEG_SR);  e_e = e_s + EEG_WIN
            m_s = int(t_start * BAND_SR); m_e = m_s + BAND_WIN
            b_s = int(t_start * bvp_sr);  b_e = b_s + bvp_win

            if e_e > eeg_raw.shape[1] or m_e > band_true.shape[1] or b_e > len(bvp):
                break

            ew = eeg_raw[:, e_s:e_e]
            mw = band_true[:, m_s:m_e]
            bw = bvp[b_s:b_e]

            hr_win, ppi_win = derive_hr_ppi_from_bvp(bw, sr=bvp_sr)

            f_eeg  = extract_eeg_features(ew)
            f_muse = extract_band_features(mw)
            f_bvp  = extract_bvp_features(bw, sr=bvp_sr)
            f_hr   = extract_hr_features(hr_win)
            f_ppi  = extract_ppi_features(ppi_win)

            feat = np.concatenate([f_eeg, f_muse, f_bvp, f_hr, f_ppi]).astype(np.float32)

            trial_feats.append(feat)
            trial_meta.append({
                "window_idx": wi,
                "start_sec": t_start,
                "end_sec": t_start + WINDOW_SEC,
                "bvp_sr": bvp_sr
            })

        if len(trial_feats) == 0:
            print(f"  Skipping {trial_key}: no valid windows")
            continue

        trial_feats = np.vstack(trial_feats)
        true_label_name = parse_true_label_from_infer_trial_key(trial_key)

        participant_trial_windows[trial_key] = {
            "X_raw": trial_feats,
            "meta": trial_meta,
            "true_label_name": true_label_name
        }
        participant_all_windows.append(trial_feats)

    if len(participant_all_windows) == 0:
        print(f"  No usable trials for {participant_name}")
        continue

    # Participant-wise normalization after vt transform
    participant_all_windows = np.vstack(participant_all_windows)
    participant_all_windows_vt = vt.transform(participant_all_windows)

    p_mu = participant_all_windows_vt.mean(axis=0)
    p_sd = participant_all_windows_vt.std(axis=0)
    p_sd = np.where(p_sd < 1e-8, 1.0, p_sd)

    # ---- Predict each trial ----
    for trial_key, trial_obj in participant_trial_windows.items():
        X_raw = trial_obj["X_raw"]
        meta = trial_obj["meta"]
        true_label_name = trial_obj["true_label_name"]

        X_vt = vt.transform(X_raw)
        X_n = (X_vt - p_mu) / p_sd
        X_n = np.clip(safe_array(X_n), -10, 10)
        X_sel = X_n[:, final_feature_idx]

        probs = final_clf.predict_proba(X_sel)
        preds = np.argmax(probs, axis=1)

        trial_pred_idx, trial_conf, mean_prob = trial_vote_from_probs(probs)
        trial_pred_label = IDX_TO_LABEL[trial_pred_idx]

        # Save window rows
        for i in range(len(preds)):
            row = {
                "participant": participant_name,
                "trial_key": trial_key,
                "window_idx": meta[i]["window_idx"],
                "start_sec": meta[i]["start_sec"],
                "end_sec": meta[i]["end_sec"],
                "bvp_sr": meta[i]["bvp_sr"],
                "pred_idx": int(preds[i]),
                "pred_label": IDX_TO_LABEL[int(preds[i])],
                "prob_NEUTRAL": float(probs[i, 0]),
                "prob_ENTHUSIASM": float(probs[i, 1]),
                "prob_SADNESS": float(probs[i, 2]),
                "prob_FEAR": float(probs[i, 3]),
            }
            if true_label_name is not None:
                row["true_label"] = true_label_name
                row["true_idx"] = EMOTION_LABELS[true_label_name]
            window_rows.append(row)

        # Save trial row
        trial_row = {
            "participant": participant_name,
            "trial_key": trial_key,
            "n_windows": len(preds),
            "trial_pred_idx": trial_pred_idx,
            "trial_pred_label": trial_pred_label,
            "trial_confidence": trial_conf,
            "prob_NEUTRAL": float(mean_prob[0]),
            "prob_ENTHUSIASM": float(mean_prob[1]),
            "prob_SADNESS": float(mean_prob[2]),
            "prob_FEAR": float(mean_prob[3]),
        }
        if true_label_name is not None:
            trial_row["true_label"] = true_label_name
            trial_row["true_idx"] = EMOTION_LABELS[true_label_name]
        trial_rows.append(trial_row)

        print(f"  {trial_key} -> {trial_pred_label} ({trial_conf:.4f})")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
trial_df = pd.DataFrame(trial_rows)
window_df = pd.DataFrame(window_rows)

if len(trial_df) > 0:
    trial_df = trial_df.sort_values(["participant", "trial_key"]).reset_index(drop=True)
    trial_df.to_csv(OUT_TRIAL_CSV, index=False)
    print(f"\nSaved trial predictions to: {OUT_TRIAL_CSV}")
    print(trial_df.head(20))
else:
    print("\nNo trial predictions generated.")

if len(window_df) > 0:
    window_df = window_df.sort_values(["participant", "trial_key", "window_idx"]).reset_index(drop=True)
    window_df.to_csv(OUT_WINDOW_CSV, index=False)
    print(f"Saved window predictions to: {OUT_WINDOW_CSV}")
    print(window_df.head(20))
else:
    print("No window predictions generated.")

# ═══════════════════════════════════════════════════════════════
# WINDOW-LEVEL ACCURACY
# ═══════════════════════════════════════════════════════════════
if len(window_df) > 0 and "true_idx" in window_df.columns:
    valid_mask = window_df["true_idx"].notna()
    if valid_mask.any():
        y_true_w = window_df.loc[valid_mask, "true_idx"].astype(int).values
        y_pred_w = window_df.loc[valid_mask, "pred_idx"].astype(int).values

        win_acc = float((y_true_w == y_pred_w).mean())

        print("\n" + "=" * 60)
        print("INFERENCE WINDOW-LEVEL RESULTS")
        print("=" * 60)
        print(f"Window-level accuracy: {win_acc:.4f}")
        print("\nWindow-level classification report:")
        print(classification_report(
            y_true_w,
            y_pred_w,
            target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
            zero_division=0
        ))
        print("Window-level confusion matrix:")
        print(confusion_matrix(y_true_w, y_pred_w, labels=list(range(NUM_CLASSES))))
    else:
        print("\nNo valid true labels found for window-level accuracy.")
else:
    print("\nWindow-level accuracy not computed.")

# ═══════════════════════════════════════════════════════════════
# TRIAL-LEVEL ACCURACY
# ═══════════════════════════════════════════════════════════════
if len(trial_df) > 0 and "true_idx" in trial_df.columns:
    valid_mask = trial_df["true_idx"].notna()
    if valid_mask.any():
        y_true_t = trial_df.loc[valid_mask, "true_idx"].astype(int).values
        y_pred_t = trial_df.loc[valid_mask, "trial_pred_idx"].astype(int).values

        trial_acc = float((y_true_t == y_pred_t).mean())

        print("\n" + "=" * 60)
        print("INFERENCE TRIAL-LEVEL RESULTS")
        print("=" * 60)
        print(f"Trial-level accuracy: {trial_acc:.4f}")
        print("\nTrial-level classification report:")
        print(classification_report(
            y_true_t,
            y_pred_t,
            target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
            zero_division=0
        ))
        print("Trial-level confusion matrix:")
        print(confusion_matrix(y_true_t, y_pred_t, labels=list(range(NUM_CLASSES))))
    else:
        print("\nNo valid true labels found for trial-level accuracy.")
else:
    print("\nTrial-level accuracy not computed.")