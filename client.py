"""
client.py  –  Middleware between your app and the RunPod endpoint.

Input format (what YOU send to this script):
{
  "input": {
    "eeg": [
      [/* TP9  samples @ 256 Hz */],
      [/* AF7  samples @ 256 Hz */],
      [/* AF8  samples @ 256 Hz */],
      [/* TP10 samples @ 256 Hz */]
    ],
    "bvp": [/* BVP/PPG samples */],
    "bvp_sr": 25,            # optional – defaults to 25 Hz
    "duration_seconds": 12,  # optional – used only for validation/logging
    "trial_key": "fear_trial1"  # optional
  }
}

What this script does:
  1. Bandpass-filters each EEG channel into 5 bands (Delta/Theta/Alpha/Beta/Gamma)
  2. Computes a smooth band-power time series at 256 Hz via sliding-window RMS
     (the handler will resample this down to 10 Hz internally, matching training)
  3. Builds  _eeg.csv   (RAW_* + Band_* columns, one row per 256 Hz sample)
             _ppg.csv   (time_s + ppg_green columns)
  4. Base64-encodes both CSVs
  5. POSTs to the RunPod endpoint and returns the JSON result
"""

import io
import json
import base64
import argparse

import numpy as np
import pandas as pd
import requests
from scipy.signal import butter, filtfilt

# ═══════════════════════════════════════════════════════════════
# CONFIG  –  set your RunPod details here or via CLI args
# ═══════════════════════════════════════════════════════════════
DEFAULT_ENDPOINT = "https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/runsync"
DEFAULT_API_KEY  = "<YOUR_RUNPOD_API_KEY>"

EEG_SR = 256   # must match training

# Band definitions: name -> (low_hz, high_hz)
BANDS = {
    "Delta": (1,  4),
    "Theta": (4,  8),
    "Alpha": (8,  13),
    "Beta":  (13, 30),
    "Gamma": (30, 100),
}

EEG_CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# Sliding-window RMS length for band-power envelope (samples @ 256 Hz)
# 256 samples = 1 second – gives a smooth 1-s power estimate
RMS_WIN = 256


# ═══════════════════════════════════════════════════════════════
# BAND POWER COMPUTATION
# ═══════════════════════════════════════════════════════════════
def _bandpass(sig: np.ndarray, lo: float, hi: float, sr: int = EEG_SR, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq   = 0.5 * sr
    lo_n  = max(lo / nyq, 1e-5)
    hi_n  = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        return np.zeros_like(sig)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig.astype(np.float64)).astype(np.float32)


def compute_band_power_series(raw_ch: np.ndarray, sr: int = EEG_SR) -> dict:
    """
    For one EEG channel, return a dict of {band_name: power_array}
    where each power_array has the same length as raw_ch and represents
    a sliding-window RMS power estimate at 256 Hz.
    This matches the format the MUSE headband stores (256 Hz grid with
    slow-changing band power values) that the training pipeline expects.
    """
    n = len(raw_ch)
    result = {}
    for band_name, (lo, hi) in BANDS.items():
        filtered   = _bandpass(raw_ch, lo, hi, sr)
        squared    = filtered ** 2
        # Sliding-window mean of squared signal (= RMS²)
        # Use np.convolve for efficiency
        kernel     = np.ones(RMS_WIN, dtype=np.float32) / RMS_WIN
        power_full = np.convolve(squared, kernel, mode="same")
        result[band_name] = np.sqrt(np.maximum(power_full, 0)).astype(np.float32)
    return result


# ═══════════════════════════════════════════════════════════════
# BUILD CSVs
# ═══════════════════════════════════════════════════════════════
def build_eeg_csv(eeg_4ch: np.ndarray) -> bytes:
    """
    eeg_4ch: shape (4, N) – raw EEG at 256 Hz for [TP9, AF7, AF8, TP10]
    Returns: CSV bytes with columns RAW_TP9 … + Alpha_TP9 … etc.
    """
    n_samples = eeg_4ch.shape[1]
    data      = {}

    # Raw EEG columns
    for i, ch in enumerate(EEG_CH_NAMES):
        data[f"RAW_{ch}"] = eeg_4ch[i]

    # Band power columns
    for i, ch in enumerate(EEG_CH_NAMES):
        bp = compute_band_power_series(eeg_4ch[i])
        for band_name in BANDS:
            data[f"{band_name}_{ch}"] = bp[band_name]

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def build_ppg_csv(bvp: np.ndarray, bvp_sr: float) -> bytes:
    """
    bvp: 1-D array of BVP/PPG samples
    Returns: CSV bytes with columns time_s, ppg_green
    """
    n = len(bvp)
    t = np.arange(n, dtype=np.float32) / bvp_sr
    df = pd.DataFrame({"time_s": t, "ppg_green": bvp.astype(np.float32)})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════
def run_inference(payload: dict, endpoint: str, api_key: str) -> dict:
    """
    payload: the full input dict as described at the top of this file.
    Returns the RunPod response dict.
    """
    inp = payload.get("input", payload)   # accept both wrapped and unwrapped

    # ---- parse inputs ----
    eeg_list = inp["eeg"]   # list of 4 lists
    bvp_list = inp["bvp"]   # flat list
    bvp_sr   = float(inp.get("bvp_sr", 25.0))
    trial_key = inp.get("trial_key", "unknown_trial")
    dur       = inp.get("duration_seconds")

    eeg_4ch = np.array(eeg_list, dtype=np.float32)  # (4, N)
    bvp     = np.array(bvp_list, dtype=np.float32)  # (M,)

    if eeg_4ch.shape[0] != 4:
        raise ValueError(f"Expected 4 EEG channels, got {eeg_4ch.shape[0]}")

    n_eeg_sec = eeg_4ch.shape[1] / EEG_SR
    n_bvp_sec = len(bvp) / bvp_sr
    print(f"[client] EEG: {eeg_4ch.shape[1]} samples ({n_eeg_sec:.1f}s @ {EEG_SR} Hz)")
    print(f"[client] BVP: {len(bvp)} samples ({n_bvp_sec:.1f}s @ {bvp_sr} Hz)")
    if dur:
        print(f"[client] Declared duration: {dur}s")

    # ---- build CSVs ----
    print("[client] Computing band powers from raw EEG ...")
    eeg_csv_bytes = build_eeg_csv(eeg_4ch)
    ppg_csv_bytes = build_ppg_csv(bvp, bvp_sr)

    print(f"[client] EEG CSV size : {len(eeg_csv_bytes)/1024:.1f} KB")
    print(f"[client] PPG CSV size : {len(ppg_csv_bytes)/1024:.1f} KB")

    # ---- base64 encode ----
    eeg_b64 = base64.b64encode(eeg_csv_bytes).decode("utf-8")
    ppg_b64 = base64.b64encode(ppg_csv_bytes).decode("utf-8")

    # ---- build RunPod request ----
    runpod_payload = {
        "input": {
            "eeg_csv":   eeg_b64,
            "ppg_csv":   ppg_b64,
            "trial_key": trial_key,
        }
    }

    print(f"[client] Sending to RunPod: {endpoint}")
    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json=runpod_payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()

    # RunPod wraps the handler return in an "output" key for runsync
    return result.get("output", result)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod inference client")
    parser.add_argument("--input",    required=True,            help="Path to input JSON file")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="RunPod runsync URL")
    parser.add_argument("--api-key",  default=DEFAULT_API_KEY,  help="RunPod API key")
    parser.add_argument("--output",   default=None,             help="Optional path to save result JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        payload = json.load(f)

    result = run_inference(payload, endpoint=args.endpoint, api_key=args.api_key)

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")
