#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import event_classification as ec

# ------------------------------------------------------------------
# Copy-paste your list of run-folders
# Shortcut for generating a copy-paste-friendly folder list:
# find . -maxdepth 1 -type d -name '*LOSO*' -printf '"%P"\n'
# ------------------------------------------------------------------
model_folders = [
    "LOSO_s8_2025-06-30_11-17-03_pc1",
    "LOSO_s9_2025-06-30_11-27-29_pc1",
    "LOSO_s24_2025-06-30_14-10-13_pc1",
    "LOSO_s17_2025-06-30_12-52-06_pc1",
    "LOSO_s22_2025-06-30_13-47-36_pc1",
    "LOSO_s1_2025-06-30_10-03-53_pc1",
    "LOSO_s7_2025-06-30_11-06-36_pc1",
    "LOSO_s6_2025-06-30_10-56-09_pc1",
    "LOSO_s19_2025-06-30_13-13-57_pc1",
    "LOSO_s14_2025-06-30_12-19-36_pc1",
    "LOSO_s20_2025-06-30_13-24-53_pc1",
    "LOSO_s16_2025-06-30_12-41-10_pc1",
    "LOSO_s10_2025-06-30_11-37-55_pc1",
    "LOSO_s18_2025-06-30_13-03-02_pc1",
    "LOSO_s13_2025-06-30_12-09-11_pc1",
    "LOSO_s11_2025-06-30_11-48-20_pc1",
    "LOSO_s26_2025-06-30_14-32-51_pc1",
    "LOSO_s5_2025-06-30_10-45-43_pc1",
    "LOSO_s4_2025-06-30_10-35-18_pc1",
    "LOSO_s3_2025-06-30_10-24-50_pc1",
    "LOSO_s2_2025-06-30_10-14-23_pc1",
    "LOSO_s27_2025-06-30_14-44-10_pc1",
    "LOSO_s25_2025-06-30_14-21-32_pc1",
    "LOSO_s28_2025-06-30_14-55-28_pc1",
    "LOSO_s23_2025-06-30_13-58-54_pc1",
    "LOSO_s15_2025-06-30_12-30-00_pc1",
    "LOSO_s12_2025-06-30_11-58-45_pc1",
    "LOSO_s21_2025-06-30_13-36-17_pc1",
]

COMMON_KWARGS = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=20,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=0,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=1,
    maj_vote_range="single",
    epn_eval=0,
    recog_threshold=0.5,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
)

event_accs = []
raw_accs = []

for folder in model_folders:
    ckpt = Path("model_checkpoints") / folder / "epoch_4.pth"
    print(f"→ evaluating {ckpt}")

    evt_acc, raw_acc = ec.main(
        saved_checkpoint_pth=str(ckpt),
        **COMMON_KWARGS,
    )

    event_accs.append(evt_acc)
    raw_accs.append(raw_acc)

print("\n=========== FINAL SUMMARY ===========")
print(f"Models evaluated   : {len(model_folders)}")
print(f"Average Transition Accuracy: {np.mean(event_accs):.4f}")
print(f"Average Raw Accuracy  : {np.mean(raw_accs):.4f}")
print("=====================================")
