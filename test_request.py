"""
test_request.py  - Send a synthetic test request to the RunPod endpoint.
Run with:
    python test_request.py --endpoint https://api.runpod.ai/v2/<ID>/runsync --api-key <KEY>
"""

import json
import argparse
import numpy as np
from client import run_inference

parser = argparse.ArgumentParser(description="Test RunPod endpoint with synthetic data")
parser.add_argument("--endpoint", required=True,        help="RunPod runsync URL")
parser.add_argument("--api-key",  required=True,        help="RunPod API key")
parser.add_argument("--duration", type=int, default=12, help="Seconds of data to generate (min 10)")
args = parser.parse_args()

ENDPOINT = args.endpoint
API_KEY  = args.api_key
DURATION = args.duration

EEG_SR = 256
BVP_SR = 25
n_eeg  = EEG_SR * DURATION
n_bvp  = BVP_SR * DURATION

np.random.seed(42)
t     = np.linspace(0, DURATION, n_eeg)
t_bvp = np.linspace(0, DURATION, n_bvp)

eeg = [
    (5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20).tolist(),
    (5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20).tolist(),
    (5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20).tolist(),
    (5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20).tolist(),
]

bvp = (
    np.sin(2 * np.pi * 1.17 * t_bvp) +
    0.3 * np.sin(2 * np.pi * 2.34 * t_bvp) +
    np.random.randn(n_bvp) * 0.05
).tolist()

payload = {
    "input": {
        "eeg":              eeg,
        "bvp":              bvp,
        "bvp_sr":           BVP_SR,
        "duration_seconds": DURATION,
        "trial_key":        "test_trial",
    }
}

print(f"Sending {DURATION}s of synthetic EEG+BVP to RunPod ...")
print(f"  EEG : 4 channels x {n_eeg} samples @ {EEG_SR} Hz")
print(f"  BVP : {n_bvp} samples @ {BVP_SR} Hz\n")

result = run_inference(payload, endpoint=ENDPOINT, api_key=API_KEY)

print("\n" + "=" * 50)
print("RESPONSE")
print("=" * 50)
print(json.dumps(result, indent=2))
