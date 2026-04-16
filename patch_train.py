import re

with open('train.py', encoding='utf-8') as f:
    content = f.read()

paths_start = content.index('# PATHS')
window_start = content.index('# WINDOW ALL TRIALS')

new_block = r"""# PATHS
MUSE_ROOT  = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
WATCH_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"

MODEL_OUT_DIR = "/kaggle/working/model_artifacts"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# LOAD TRAINING TRIALS
print("\nLoading training trials from Emognition ...")
t0 = time.time()

train_trials = []
skipped = defaultdict(int)

for subj in sorted(os.listdir(MUSE_ROOT)):
    subj_dir = os.path.join(MUSE_ROOT, subj)
    if not os.path.isdir(subj_dir) or not subj.isdigit():
        continue
    sid = subj

    for emotion, label_idx in EMOTION_LABELS.items():
        muse_clean = os.path.join(subj_dir, f"{sid}_{emotion}_STIMULUS_MUSE_cleaned.json")
        muse_raw   = os.path.join(subj_dir, f"{sid}_{emotion}_STIMULUS_MUSE.json")
        muse_json  = muse_clean if os.path.isfile(muse_clean) else muse_raw

        if not os.path.isfile(muse_json):
            skipped["missing_muse_json"] += 1
            continue

        sw_json = os.path.join(WATCH_ROOT, sid, f"{sid}_{emotion}_STIMULUS_SAMSUNG_WATCH.json")
        if not os.path.isfile(sw_json):
            skipped["missing_watch_json"] += 1
            continue

        try:
            with open(muse_json) as f:
                md = json.load(f)
            eeg = np.stack([safe_array(np.array(md[ch], dtype=np.float32)) for ch in EEG_CHANNELS])
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

        dur      = min(eeg.shape[1] / EEG_SR, band_arr.shape[1] / BAND_SR, len(bvp_vals) / TRAIN_BVP_SR)
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

if len(train_trials) == 0:
    raise RuntimeError("No training trials loaded. Check paths and emotion names.")

"""

content = content[:paths_start] + new_block + content[window_start:]

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("train.py patched successfully.")
