# preprocess.py
# Shared preprocessing, feature extraction, and helper functions
# Used by both train.py and handler.py

import numpy as np
import warnings
from scipy.signal import welch, coherence as sp_coherence, find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
EEG_SR             = 256
BAND_SR            = 10
TRAIN_BAND_STORED_SR = 256
TRAIN_BVP_SR       = 20

WINDOW_SEC   = 10
OVERLAP_FRAC = 0.75
STEP_SEC     = WINDOW_SEC * (1 - OVERLAP_FRAC)

EEG_WIN      = WINDOW_SEC * EEG_SR
BAND_WIN     = WINDOW_SEC * BAND_SR
TRAIN_BVP_WIN = int(WINDOW_SEC * TRAIN_BVP_SR)

NUM_CLASSES = 4

EMOTION_LABELS = {
    "NEUTRAL":    0,
    "ENTHUSIASM": 1,
    "SADNESS":    2,
    "FEAR":       3,
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

N_FEAT_EEG  = 156
N_FEAT_MUSE = 62
N_FEAT_BVP  = 7
N_FEAT_HR   = 5
N_FEAT_PPI  = 8
N_FEATURES_RAW = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI

FREQ_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
CH_PAIRS   = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# ═══════════════════════════════════════════════════════════════
# GENERAL HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_array(x):
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)


def infer_sampling_rate_from_time_series(t):
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return (1.0 / np.median(dt)) if len(dt) > 0 else None


def resample_1d_by_time(arr, orig_sr, target_sr):
    arr = safe_array(np.asarray(arr, dtype=np.float32))
    if len(arr) == 0:
        return arr.astype(np.float32)
    old_t  = np.arange(len(arr), dtype=np.float64) / float(orig_sr)
    new_len = max(int(np.floor(len(arr) * float(target_sr) / float(orig_sr))), 1)
    new_t  = np.arange(new_len, dtype=np.float64) / float(target_sr)
    return np.interp(new_t, old_t, arr).astype(np.float32)


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
    return base_name.rsplit(".", 1)[0]


def parse_true_label_from_infer_trial_key(trial_key):
    low = trial_key.lower()
    if low.startswith("neutral"):     return "NEUTRAL"
    if low.startswith("enthusiasm"):  return "ENTHUSIASM"
    if low.startswith("sad"):         return "SADNESS"
    if low.startswith("fear"):        return "FEAR"
    return None


def trial_vote_from_probs(probs, num_classes=NUM_CLASSES):
    mean_prob = np.mean(probs, axis=0)
    pred_idx  = int(np.argmax(mean_prob))
    conf      = float(mean_prob[pred_idx])
    return pred_idx, conf, mean_prob

# ═══════════════════════════════════════════════════════════════
# BVP -> HR / PPI
# ═══════════════════════════════════════════════════════════════
def bandpass_bvp(sig, sr, low=0.7, high=3.5, order=2):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < max(10, order * 3):
        return sig
    nyq   = 0.5 * sr
    low_n  = max(low  / nyq, 1e-5)
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
    sstd  = np.std(sig_f)
    sig_n = (sig_f - np.mean(sig_f)) / sstd if sstd > 1e-10 else sig_f - np.mean(sig_f)

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

    return (60.0 / ibi_sec).astype(np.float32), (ibi_sec * 1000.0).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
def _compute_psd(sig, sr, nperseg=256):
    return welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))


def _band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return np.mean(psd[mask]) if mask.any() else 1e-10


def _differential_entropy(sig):
    v = np.var(sig)
    return 0.5 * np.log(2 * np.pi * np.e * v) if v > 1e-12 else 0.0


def _hjorth(sig):
    d1  = np.diff(sig)
    d2  = np.diff(d1)
    act = np.var(sig)
    if act < 1e-12:
        return 0.0, 0.0, 0.0
    mob  = np.sqrt(np.var(d1) / act)
    v_d1 = np.var(d1)
    if v_d1 < 1e-12:
        return float(act), float(mob), 0.0
    comp = np.sqrt(np.var(d2) / v_d1) / mob if mob > 1e-12 else 0.0
    return float(act), float(mob), float(comp)


def _zcr(sig):
    return float(np.sum(np.abs(np.diff(np.sign(sig))) > 0)) / max(len(sig) - 1, 1)


def spectral_entropy(sig, sr=EEG_SR, nperseg=256):
    _, psd = welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))
    psd_n  = psd / (psd.sum() + 1e-12)
    psd_n  = psd_n[psd_n > 0]
    return float(-np.sum(psd_n * np.log2(psd_n))) if len(psd_n) > 0 else 0.0


def permutation_entropy(sig, order=3, delay=1):
    n = len(sig)
    if n < (order - 1) * delay + 1:
        return 0.0
    indices = np.arange(n - (order - 1) * delay)
    cols    = np.column_stack([sig[indices + d * delay] for d in range(order)])
    perms   = np.argsort(cols, axis=1)
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
        coeffs  = pywt.wavedec(sig, wavelet, level=min(level, max_lev))
        energies = [float(np.mean(c ** 2)) for c in coeffs[:5]]
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
        sig       = eeg_4ch[ch].astype(np.float64)
        freqs, psd = _compute_psd(sig, sr)

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw         = _band_power(freqs, psd, lo, hi)
            bp[ch, bi] = np.log1p(pw)
        feats.extend(bp[ch].tolist())

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw = _band_power(freqs, psd, lo, hi)
            feats.append(0.5 * np.log(2 * np.pi * np.e * pw) if pw > 1e-12 else 0.0)

        feats.extend(_hjorth(sig))
        feats.extend([
            float(np.mean(sig)), float(np.std(sig)),
            float(skew(sig)),    float(kurtosis(sig))
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


# 62 MUSE band-power features
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
    sig   = bvp_window.astype(np.float64)
    feats = [
        float(np.mean(sig)), float(np.std(sig)),
        float(skew(sig)),    float(kurtosis(sig))
    ]
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs   = np.fft.rfftfreq(len(sig), 1.0 / sr)
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
        float(np.mean(hr)),  float(np.std(hr)),
        float(np.min(hr)),   float(np.max(hr)),
        float(np.max(hr) - np.min(hr))
    ], dtype=np.float32))


# 8 PPI features
def extract_ppi_features(ppi_values):
    if len(ppi_values) < 3:
        return np.zeros(8, dtype=np.float32)

    ipi   = ppi_values.astype(np.float64)
    ipi_s = ipi / 1000.0 if np.median(ipi) > 10 else ipi.copy()

    feats = [float(np.mean(ipi_s)), float(np.std(ipi_s))]
    sd    = np.diff(ipi_s)

    feats.append(float(np.sqrt(np.mean(sd ** 2))) if len(sd) > 0 else 0.0)
    feats.append(float(np.mean(np.abs(sd) > 0.05)) if len(sd) > 0 else 0.0)
    feats.append(float(np.std(sd)) if len(sd) > 0 else 0.0)

    if len(ipi_s) > 6:
        t_ipi     = np.cumsum(ipi_s)
        t_uniform = np.arange(t_ipi[0], t_ipi[-1], 0.25)
        if len(t_uniform) > 8:
            ipi_uniform = np.interp(t_uniform, t_ipi, ipi_s)
            f_ipi, psd_ipi = welch(ipi_uniform, fs=4.0, nperseg=min(len(ipi_uniform), 32))
            lf_m = (f_ipi >= 0.04) & (f_ipi < 0.15)
            hf_m = (f_ipi >= 0.15) & (f_ipi < 0.4)
            lf   = float(np.mean(psd_ipi[lf_m])) if lf_m.any() else 0.0
            hf   = float(np.mean(psd_ipi[hf_m])) if hf_m.any() else 0.0
            feats.extend([lf, hf, lf / hf if hf > 1e-10 else 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0])

    return safe_array(np.array(feats, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════
# WINDOW A SINGLE TRIAL INTO FEATURE VECTORS
# ═══════════════════════════════════════════════════════════════
def window_trial_to_features(eeg_raw, band_true, bvp, bvp_sr):
    """
    Given aligned arrays for one trial, slide a window and return
    (X_raw  [n_windows, N_FEATURES_RAW],  meta_list).
    """
    bvp_win = int(round(WINDOW_SEC * bvp_sr))

    dur = eeg_raw.shape[1] / EEG_SR
    dur = min(dur, band_true.shape[1] / BAND_SR)
    dur = min(dur, len(bvp) / bvp_sr)

    eeg_raw  = eeg_raw[:,  :int(dur * EEG_SR)]
    band_true= band_true[:, :int(dur * BAND_SR)]
    bvp      = bvp[       :int(dur * bvp_sr)]

    n_wins = int((dur - WINDOW_SEC) / STEP_SEC) + 1
    if n_wins <= 0:
        return np.empty((0, N_FEATURES_RAW), dtype=np.float32), []

    feats_list = []
    meta_list  = []

    for wi in range(n_wins):
        t_start = wi * STEP_SEC
        e_s = int(t_start * EEG_SR);  e_e = e_s + EEG_WIN
        m_s = int(t_start * BAND_SR); m_e = m_s + BAND_WIN
        b_s = int(t_start * bvp_sr);  b_e = b_s + bvp_win

        if e_e > eeg_raw.shape[1] or m_e > band_true.shape[1] or b_e > len(bvp):
            break

        ew = eeg_raw[:, e_s:e_e]
        mw = band_true[:, m_s:m_e]
        bw = bvp[b_s:b_e]

        hr_win, ppi_win = derive_hr_ppi_from_bvp(bw, sr=bvp_sr)

        feat = np.concatenate([
            extract_eeg_features(ew),
            extract_band_features(mw),
            extract_bvp_features(bw, sr=bvp_sr),
            extract_hr_features(hr_win),
            extract_ppi_features(ppi_win),
        ]).astype(np.float32)

        feats_list.append(feat)
        meta_list.append({
            "window_idx": wi,
            "start_sec":  t_start,
            "end_sec":    t_start + WINDOW_SEC,
            "bvp_sr":     bvp_sr,
        })

    X = np.vstack(feats_list) if feats_list else np.empty((0, N_FEATURES_RAW), dtype=np.float32)
    return X, meta_list


# ================================================================
# BAND POWER FROM RAW EEG (used by train.py, handler.py, client.py)
# ================================================================
_BAND_DEF_ORDER = [
    ("Alpha", (8,  13)),
    ("Beta",  (13, 30)),
    ("Delta", (1,   4)),
    ("Gamma", (30, 100)),
    ("Theta", (4,   8)),
]
_RMS_WIN = 256  # 1-second sliding window @ 256 Hz


def _bp_filter(sig, lo, hi, sr=EEG_SR, order=4):
    nyq = 0.5 * sr
    lon = max(lo / nyq, 1e-5)
    hin = min(hi / nyq, 0.999)
    if lon >= hin:
        return np.zeros_like(sig)
    b, a = butter(order, [lon, hin], btype="band")
    return filtfilt(b, a, sig.astype(np.float64)).astype(np.float32)


def compute_band_arr(eeg_4ch, sr=EEG_SR):
    rows = []
    kernel = np.ones(_RMS_WIN, dtype=np.float32) / _RMS_WIN
    for _, (lo, hi) in _BAND_DEF_ORDER:
        for ci in range(4):
            filt  = _bp_filter(eeg_4ch[ci], lo, hi, sr)
            power = np.convolve(filt ** 2, kernel, mode="same")
            rows.append(np.sqrt(np.maximum(power, 0)).astype(np.float32))
    return np.stack(rows, axis=0)  # (20, N) matching BAND_CHANNELS order
