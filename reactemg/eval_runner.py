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
"Finetuned_s1_2025-07-21_18-33-34_pc1",
"Finetuned_s2_2025-07-21_18-34-22_pc1",
"Finetuned_s3_2025-07-21_18-35-12_pc1",
"Finetuned_s4_2025-07-21_18-35-58_pc1",
"Finetuned_s5_2025-07-21_18-36-48_pc1",
"Finetuned_s6_2025-07-21_18-37-36_pc1",
"Finetuned_s7_2025-07-21_18-38-22_pc1",
"Finetuned_s8_2025-07-21_18-39-08_pc1",
"Finetuned_s9_2025-07-21_18-39-51_pc1",
"Finetuned_s10_2025-07-21_18-40-22_pc1",
"Finetuned_s11_2025-07-21_18-40-55_pc1",
"Finetuned_s12_2025-07-21_18-41-26_pc1",
"Finetuned_s13_2025-07-21_18-41-55_pc1",
"Finetuned_s14_2025-07-21_18-42-27_pc1",
"Finetuned_s15_2025-07-21_18-43-00_pc1",
"Finetuned_s16_2025-07-21_18-43-27_pc1",
"Finetuned_s17_2025-07-21_18-43-59_pc1",
"Finetuned_s18_2025-07-21_18-44-28_pc1",
"Finetuned_s19_2025-07-21_18-44-59_pc1",
"Finetuned_s20_2025-07-21_18-45-29_pc1",
"Finetuned_s21_2025-07-21_18-46-02_pc1",
"Finetuned_s22_2025-07-21_18-46-31_pc1",
"Finetuned_s23_2025-07-21_18-47-00_pc1",
"Finetuned_s24_2025-07-21_18-47-32_pc1",
"Finetuned_s25_2025-07-21_18-48-01_pc1",
"Finetuned_s26_2025-07-21_18-48-28_pc1",
"Finetuned_s27_2025-07-21_18-48-57_pc1",
"Finetuned_s28_2025-07-21_18-49-26_pc1",

]

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
"""
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
