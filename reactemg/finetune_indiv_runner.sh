#!/usr/bin/env bash
set -euo pipefail

#############################################
# 1) List of validation patient IDs (order matters)
#############################################
patient_ids=(
  "s1"  "s2"  "s3"  "s4"  "s5"  "s6"  "s7"
  "s8"  "s9"  "s10" "s11" "s12" "s13" "s14"
  "s15" "s16" "s17" "s18" "s19" "s20" "s21"
  "s22" "s23" "s24" "s25" "s26" "s27" "s28"
)

#############################################
# 2) Matching list of fine-tuning checkpoints
# MUST be in the same order and length
#############################################
checkpoint_pths=(
"LOSO_s1_2025-07-19_14-16-50_pc1"
"LOSO_s2_2025-07-19_14-27-54_pc1"
"LOSO_s3_2025-07-19_14-38-53_pc1"
"LOSO_s4_2025-07-19_14-49-58_pc1"
"LOSO_s5_2025-07-19_15-01-02_pc1"
"LOSO_s6_2025-07-19_15-12-23_pc1"
"LOSO_s7_2025-07-19_15-23-32_pc1"
"LOSO_s8_2025-07-19_15-34-44_pc1"
"LOSO_s9_2025-07-19_15-45-51_pc1"
"LOSO_s10_2025-07-19_15-57-08_pc1"
"LOSO_s11_2025-07-19_16-08-14_pc1"
"LOSO_s12_2025-07-19_16-19-30_pc1"
"LOSO_s13_2025-07-19_16-30-46_pc1"
"LOSO_s14_2025-07-19_16-41-56_pc1"
"LOSO_s15_2025-07-19_16-53-03_pc1"
"LOSO_s16_2025-07-19_17-04-21_pc1"
"LOSO_s17_2025-07-19_17-15-37_pc1"
"LOSO_s18_2025-07-19_17-26-45_pc1"
"LOSO_s19_2025-07-19_17-37-56_pc1"
"LOSO_s20_2025-07-19_17-49-06_pc1"
"LOSO_s21_2025-07-19_18-00-20_pc1"
"LOSO_s22_2025-07-19_18-11-27_pc1"
"LOSO_s23_2025-07-19_18-22-30_pc1"
"LOSO_s24_2025-07-19_18-33-38_pc1"
"LOSO_s25_2025-07-19_18-44-44_pc1"
"LOSO_s26_2025-07-19_18-55-59_pc1"
"LOSO_s27_2025-07-19_19-07-08_pc1"
"LOSO_s28_2025-07-19_19-18-22_pc1"
)

#############################################
# 3) Sanity-check the array sizes
#############################################
if [[ ${#patient_ids[@]} -ne ${#checkpoint_pths[@]} ]]; then
  echo "ERROR: patient_ids and checkpoint_pths must have the same length." >&2
  exit 1
fi

#############################################
# 4) Loop through both arrays in lock-step
#############################################
for idx in "${!patient_ids[@]}"; do
  patient_id="${patient_ids[$idx]}"
  saved_checkpoint_pth="model_checkpoints/${checkpoint_pths[$idx]}/epoch_4.pth"

  # Optional: make sure the checkpoint file exists
  if [[ ! -f "$saved_checkpoint_pth" ]]; then
    echo "WARNING: Checkpoint $saved_checkpoint_pth not found - skipping $patient_id." >&2
    continue
  fi

  exp_name="Finetuned_${patient_id}"

  cmd=(
    python3 main.py
      --offset 30
      --num_classes 3
      --task_selection 0 1 2
      --dataset_selection finetune  # IMPORTANT! Set to enable finetuning
      --window_size 600
      --val_patient_ids "${patient_id}"
      --exp_name "${exp_name}"
      --epn_subset_percentage 1.0
      --epochs 5
      --model_choice any2any
      --inner_window_size 600
      --saved_checkpoint_pth "${saved_checkpoint_pth}"
      --use_lora 1
  )

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
done
