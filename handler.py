# handler.py
import os, io, pickle, base64, traceback
import numpy as np
import pandas as pd
import runpod

from preprocess import (
    EEG_CHANNELS, BAND_CHANNELS,
    IDX_TO_LABEL, EMOTION_LABELS, NUM_CLASSES,
    EEG_SR, BAND_SR,
    safe_array, resample_multich_by_time,
    infer_sampling_rate_from_time_series,
    parse_true_label_from_infer_trial_key,
    trial_vote_from_probs,
    window_trial_to_features,
)

# ── Model load ──────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model_artifacts/lda_model.pkl")
print(f"[startup] Loading model from {MODEL_PATH} ...")
with open(MODEL_PATH, "rb") as f:
    _artifacts = pickle.load(f)
_vt                = _artifacts["vt"]
_clf               = _artifacts["final_clf"]
_final_feature_idx = _artifacts["final_feature_idx"]
print(f"[startup] Model loaded. K={_artifacts['FINAL_K']}, shrinkage={_artifacts['FINAL_SH']}")

# ── Label mapping: internal UPPERCASE -> output lowercase ────────
LABEL_DISPLAY = {
    "NEUTRAL":    "neutral",
    "ENTHUSIASM": "enthusiasm",
    "SADNESS":    "sadness",
    "FEAR":       "fear",
}

# ── Inference helper ─────────────────────────────────────────────
def _predict_trial(eeg_df: pd.DataFrame, ppg_df: pd.DataFrame):
    missing_eeg = [c for c in EEG_CHANNELS + BAND_CHANNELS if c not in eeg_df.columns]
    if missing_eeg:
        raise ValueError(f"EEG CSV missing columns: {missing_eeg}")
    if "ppg_green" not in ppg_df.columns:
        raise ValueError("PPG CSV missing column: ppg_green")

    eeg_raw = np.stack(
        [safe_array(eeg_df[ch].values.astype(np.float32)) for ch in EEG_CHANNELS], axis=0)
    band_raw_256 = np.stack(
        [safe_array(eeg_df[ch].values.astype(np.float32)) for ch in BAND_CHANNELS], axis=0)
    band_true = resample_multich_by_time(band_raw_256, EEG_SR, BAND_SR)
    bvp = safe_array(ppg_df["ppg_green"].values.astype(np.float32))

    bvp_sr = 25.0
    if "time_s" in ppg_df.columns:
        sr_infer = infer_sampling_rate_from_time_series(ppg_df["time_s"].values)
        if sr_infer is not None:
            bvp_sr = float(sr_infer)

    X_raw, meta_list = window_trial_to_features(eeg_raw, band_true, bvp, bvp_sr)
    if X_raw.shape[0] == 0:
        raise ValueError("Trial is shorter than one window (need >= 10 s of data).")

    X_vt = _vt.transform(X_raw)
    p_mu = X_vt.mean(axis=0)
    p_sd = np.where(X_vt.std(axis=0) < 1e-8, 1.0, X_vt.std(axis=0))
    X_n  = np.clip(safe_array((X_vt - p_mu) / p_sd), -10, 10)

    probs = _clf.predict_proba(X_n[:, _final_feature_idx])
    preds = np.argmax(probs, axis=1)
    trial_pred_idx, trial_conf, mean_prob = trial_vote_from_probs(probs)
    return trial_pred_idx, trial_conf, mean_prob, preds, probs, meta_list


# ── RunPod handler ───────────────────────────────────────────────
def handler(event):
    try:
        job_input = event.get("input", {})
        eeg_b64   = job_input.get("eeg_csv")
        ppg_b64   = job_input.get("ppg_csv")
        trial_key = job_input.get("trial_key", "unknown_trial")

        if not eeg_b64 or not ppg_b64:
            return {"error": "Both 'eeg_csv' and 'ppg_csv' (base64-encoded) are required."}

        eeg_df = pd.read_csv(io.BytesIO(base64.b64decode(eeg_b64)))
        ppg_df = pd.read_csv(io.BytesIO(base64.b64decode(ppg_b64)))

        trial_pred_idx, trial_conf, mean_prob, preds, probs, meta_list =             _predict_trial(eeg_df, ppg_df)

        trial_pred_label = IDX_TO_LABEL[trial_pred_idx]

        # Per-window results
        windows = []
        for i, m in enumerate(meta_list):
            win_label = IDX_TO_LABEL[int(preds[i])]
            windows.append({
                "window_idx": m["window_idx"],
                "start_sec":  round(m["start_sec"], 3),
                "end_sec":    round(m["end_sec"],   3),
                "emotion":    LABEL_DISPLAY[win_label],
                "scores": {
                    "neutral":    round(float(probs[i, 0]), 4),
                    "enthusiasm": round(float(probs[i, 1]), 4),
                    "sadness":    round(float(probs[i, 2]), 4),
                    "fear":       round(float(probs[i, 3]), 4),
                },
            })

        # Top-level response
        response = {
            "output": {
                "emotion":    LABEL_DISPLAY[trial_pred_label],
                "confidence": round(trial_conf, 4),
                "scores": {
                    "neutral":    round(float(mean_prob[0]), 4),
                    "enthusiasm": round(float(mean_prob[1]), 4),
                    "sadness":    round(float(mean_prob[2]), 4),
                    "fear":       round(float(mean_prob[3]), 4),
                },
                "n_windows": len(windows),
                "windows":   windows,
            }
        }

        # Optional ground-truth echo (when trial_key starts with an emotion name)
        true_label = parse_true_label_from_infer_trial_key(trial_key)
        if true_label:
            response["output"]["true_emotion"] = LABEL_DISPLAY[true_label]
            response["output"]["correct"]      = (true_label == trial_pred_label)

        return response

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
