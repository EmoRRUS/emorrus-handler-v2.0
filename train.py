# train.py
# ═══════════════════════════════════════════════════════════════
# Run this ONCE (e.g. on Kaggle) to train the LDA model and save
# all artifacts to MODEL_OUT_DIR.  The saved bundle is then baked
# into the RunPod Docker image (or mounted as a network volume).
# ═══════════════════════════════════════════════════════════════

import os
import json
import time
import pickle
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

from preprocess import (
    PROCESSED_EEG_ROOT_DEFAULT,  # not defined – we set paths below
    EEG_CHANNELS, BAND_CHANNELS,
    EMOTION_LABELS, IDX_TO_LABEL, NUM_CLASSES,
    EEG_SR, BAND_SR, TRAIN_BAND_STORED_SR, TRAIN_BVP_SR,
    WINDOW_SEC, STEP_SEC, EEG_WIN, BAND_WIN, TRAIN_BVP_WIN,
    N_FEATURES_RAW,
    safe_array, resample_1d_by_time,
    parse_emotion_from_training_trial_name,
    derive_hr_ppi_from_bvp,
    extract_eeg_features, extract_band_features,
    extract_bvp_features, extract_hr_features, extract_ppi_features,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# PATHS  – edit these before running
# ═══════════════════════════════════════════════════════════════
PROCESSED_EEG_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
RAW_WATCH_ROOT     = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
RAW_MUSE_ROOT      = "/kaggle/input/datasets/ruchiabey/emognition"

MODEL_OUT_DIR      = "/kaggle/working/model_artifacts"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOAD TRAINING TRIALS
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
            with open(eeg_json) as f:
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
            with open(muse_json) as f:
                md = json.load(f)
            band_list = []
            for bch in BAND_CHANNELS:
                raw_band  = safe_array(np.array(md[bch], dtype=np.float32))
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
            with open(sw_json) as f:
                sw = json.load(f)
            if "BVPProcessed" not in sw:
                skipped["missing_BVPProcessed"] += 1
                continue
            bvp_vals = safe_array(np.array([r[1] for r in sw["BVPProcessed"]], dtype=np.float32))
        except Exception:
            skipped["bad_watch_json"] += 1
            continue

        dur      = min(T_eeg / EEG_SR, band_arr.shape[1] / BAND_SR, len(bvp_vals) / TRAIN_BVP_SR)
        eeg      = eeg[:,      :int(dur * EEG_SR)]
        band_arr = band_arr[:, :int(dur * BAND_SR)]
        bvp_vals = bvp_vals[   :int(dur * TRAIN_BVP_SR)]

        train_trials.append({
            "sid": sid, "emotion": emotion, "label": label_idx,
            "eeg": eeg, "band": band_arr, "bvp": bvp_vals,
            "trial_key": f"{sid}_{emotion}",
        })

print(f"Training trials loaded: {len(train_trials)}  ({time.time()-t0:.1f}s)")
if skipped:
    print("Skipped:", dict(skipped))

# ═══════════════════════════════════════════════════════════════
# WINDOW ALL TRIALS
# ═══════════════════════════════════════════════════════════════
print("\nWindowing training trials ...")
t1 = time.time()

all_feat_w, all_labels_w = [], []
all_tkeys_w, all_sids_w, all_tidx_w = [], [], []

for ti, tr in enumerate(train_trials):
    eeg, band, bvp = tr["eeg"], tr["band"], tr["bvp"]
    dur = min(eeg.shape[1] / EEG_SR, band.shape[1] / BAND_SR, len(bvp) / TRAIN_BVP_SR)
    n_wins = int((dur - WINDOW_SEC) / STEP_SEC) + 1
    if n_wins <= 0:
        continue

    for wi in range(n_wins):
        t_start = wi * STEP_SEC
        e_s = int(t_start * EEG_SR);       e_e = e_s + EEG_WIN
        m_s = int(t_start * BAND_SR);      m_e = m_s + BAND_WIN
        b_s = int(t_start * TRAIN_BVP_SR); b_e = b_s + TRAIN_BVP_WIN

        if e_e > eeg.shape[1] or m_e > band.shape[1] or b_e > len(bvp):
            break

        hr_win, ppi_win = derive_hr_ppi_from_bvp(bvp[b_s:b_e], sr=TRAIN_BVP_SR)
        feat = np.concatenate([
            extract_eeg_features(eeg[:, e_s:e_e]),
            extract_band_features(band[:, m_s:m_e]),
            extract_bvp_features(bvp[b_s:b_e], sr=TRAIN_BVP_SR),
            extract_hr_features(hr_win),
            extract_ppi_features(ppi_win),
        ]).astype(np.float32)

        all_feat_w.append(feat)
        all_labels_w.append(tr["label"])
        all_tkeys_w.append(tr["trial_key"])
        all_sids_w.append(tr["sid"])
        all_tidx_w.append(ti)

allF_raw = np.array(all_feat_w, dtype=np.float32)
allY     = np.array(all_labels_w)
allSID   = np.array(all_sids_w)
allTI    = np.array(all_tidx_w)
print(f"Training windows: {len(allY)}  ({time.time()-t1:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# VARIANCE THRESHOLD
# ═══════════════════════════════════════════════════════════════
vt = VarianceThreshold(threshold=0.001)
allF_sel = vt.fit_transform(allF_raw)
print(f"VarianceThreshold kept {allF_sel.shape[1]} / {N_FEATURES_RAW} features")

# ═══════════════════════════════════════════════════════════════
# PER-SUBJECT NORMALISATION
# ═══════════════════════════════════════════════════════════════
allF_n = np.empty_like(allF_sel)
train_sid_stats = {}

for sid in sorted(set(allSID)):
    mask = (allSID == sid)
    data = allF_sel[mask]
    mu   = data.mean(axis=0)
    sd   = data.std(axis=0)
    sd   = np.where(sd < 1e-8, 1.0, sd)
    train_sid_stats[sid] = (mu, sd)
    allF_n[mask] = (data - mu) / sd

allF_n = np.clip(safe_array(allF_n), -10, 10)

# ═══════════════════════════════════════════════════════════════
# BALANCED FOLD ASSIGNMENT (4-fold)
# ═══════════════════════════════════════════════════════════════
labels_arr       = np.array([tr["label"] for tr in train_trials])
subj_emo_trial   = {}
for ti, tr in enumerate(train_trials):
    subj_emo_trial.setdefault(tr["sid"], {})[tr["label"]] = ti

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

# ═══════════════════════════════════════════════════════════════
# PER-FOLD MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════
print("\nComputing per-fold MI ...")
mi_per_fold = {}
for fk in range(4):
    tr_mask = (win_pos != fk) & (win_pos != (fk + 1) % 4)
    tr_idx  = np.where(tr_mask)[0]
    sub     = (
        np.random.RandomState(42).choice(tr_idx, 2000, replace=False)
        if len(tr_idx) > 2000 else tr_idx
    )
    mi_per_fold[fk] = mutual_info_classif(allF_n[sub], allY[sub], random_state=42, n_neighbors=5)

# ═══════════════════════════════════════════════════════════════
# GRID SEARCH FOR BEST K + SHRINKAGE
# ═══════════════════════════════════════════════════════════════
print("\nGrid search for LDA hyperparameters ...")
K_GRID  = [80, 120, allF_n.shape[1]]
SH_GRID = ['auto', 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
grid_scores = []

for K in K_GRID:
    for sh in SH_GRID:
        fold_scores = []
        for fk in range(4):
            te_mask = (win_pos == fk)
            vl_mask = (win_pos == (fk + 1) % 4)
            tr_mask = ~te_mask & ~vl_mask
            fi = np.argsort(-mi_per_fold[fk])[:K]
            try:
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=sh)
                clf.fit(allF_n[tr_mask][:, fi], allY[tr_mask])
                fold_scores.append(float((clf.predict(allF_n[vl_mask][:, fi]) == allY[vl_mask]).mean()))
            except Exception:
                fold_scores.append(-1.0)
        grid_scores.append((np.mean(fold_scores), K, sh))

grid_scores.sort(reverse=True, key=lambda x: x[0])
best_mean_val, FINAL_K, FINAL_SH = grid_scores[0]
print(f"Best params: K={FINAL_K}, shrinkage={FINAL_SH}, mean_val={best_mean_val:.4f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN FINAL MODEL
# ═══════════════════════════════════════════════════════════════
print("\nTraining final LDA on all windows ...")
sub_idx = (
    np.random.RandomState(42).choice(len(allF_n), 2000, replace=False)
    if len(allF_n) > 2000 else np.arange(len(allF_n))
)
mi_global        = mutual_info_classif(allF_n[sub_idx], allY[sub_idx], random_state=42, n_neighbors=5)
final_feature_idx = np.argsort(-mi_global)[:FINAL_K]

final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=FINAL_SH)
final_clf.fit(allF_n[:, final_feature_idx], allY)

# Sanity check
train_pred = final_clf.predict(allF_n[:, final_feature_idx])
print("\nTraining sanity-check:")
print(classification_report(
    allY, train_pred,
    target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
    zero_division=0
))

# ═══════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════
artifacts = {
    "vt":                 vt,
    "final_clf":          final_clf,
    "final_feature_idx":  final_feature_idx,
    "train_sid_stats":    train_sid_stats,
    "FINAL_K":            FINAL_K,
    "FINAL_SH":           FINAL_SH,
}

artifact_path = os.path.join(MODEL_OUT_DIR, "lda_model.pkl")
with open(artifact_path, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\nArtifacts saved to: {artifact_path}")
print("Keys:", list(artifacts.keys()))
