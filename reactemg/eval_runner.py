#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import event_classification as ec

# ------------------------------------------------------------------
# Copy-paste your list of run-folders
# Shortcut for generating a copy-paste-friendly folder list:
# find . -maxdepth 1 -type d -name '*LOSO*' -printf '"%P",\n'
# ------------------------------------------------------------------
model_folders = [
"LOSO_s2_2025-07-19_14-27-54_pc1",
"LOSO_s1_2025-07-19_14-16-50_pc1",
"LOSO_s19_2025-07-19_17-37-56_pc1",
"LOSO_s18_2025-07-19_17-26-45_pc1",
"LOSO_s22_2025-07-19_18-11-27_pc1",
"LOSO_s24_2025-07-19_18-33-38_pc1",
"LOSO_s11_2025-07-19_16-08-14_pc1",
"LOSO_s25_2025-07-19_18-44-44_pc1",
"LOSO_s15_2025-07-19_16-53-03_pc1",
"LOSO_s16_2025-07-19_17-04-21_pc1",
"LOSO_s14_2025-07-19_16-41-56_pc1",
"LOSO_s21_2025-07-19_18-00-20_pc1",
"LOSO_s6_2025-07-19_15-12-23_pc1",
"LOSO_s13_2025-07-19_16-30-46_pc1",
"LOSO_s4_2025-07-19_14-49-58_pc1",
"LOSO_s12_2025-07-19_16-19-30_pc1",
"LOSO_s20_2025-07-19_17-49-06_pc1",
"LOSO_s26_2025-07-19_18-55-59_pc1",
"LOSO_s28_2025-07-19_19-18-22_pc1",
"LOSO_s9_2025-07-19_15-45-51_pc1",
"LOSO_s10_2025-07-19_15-57-08_pc1",
"LOSO_s23_2025-07-19_18-22-30_pc1",
"LOSO_s27_2025-07-19_19-07-08_pc1",
"LOSO_s7_2025-07-19_15-23-32_pc1",
"LOSO_s5_2025-07-19_15-01-02_pc1",
"LOSO_s17_2025-07-19_17-15-37_pc1",
"LOSO_s8_2025-07-19_15-34-44_pc1",
"LOSO_s3_2025-07-19_14-38-53_pc1",
]

"""
# No lookahead
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
"""

# With lookahead
COMMON_KWARGS = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=1,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=50,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=20,
    maj_vote_range="future",
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
