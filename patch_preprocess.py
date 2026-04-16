import os

# ── 1. Add compute_band_arr to preprocess.py ─────────────────────
ADDITION = """

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
"""

base = os.path.dirname(__file__)
preprocess_path = os.path.join(base, "preprocess.py")
c = open(preprocess_path, encoding="utf-8").read()
if "compute_band_arr" not in c:
    with open(preprocess_path, "a", encoding="utf-8") as f:
        f.write(ADDITION)
    print("preprocess.py: compute_band_arr added.")
else:
    print("preprocess.py: already has compute_band_arr, skipping.")

# ── 2. Add compute_band_arr to the train.py import ───────────────
train_path = os.path.join(base, "train.py")
c = open(train_path, encoding="utf-8").read()

if "compute_band_arr" not in c:
    old = "extract_eeg_features, extract_band_features,"
    new = "extract_eeg_features, extract_band_features,\n    compute_band_arr,"
    c = c.replace(old, new, 1)
    open(train_path, "w", encoding="utf-8").write(c)
    print("train.py: compute_band_arr added to import.")
else:
    print("train.py: import already has compute_band_arr, skipping.")
