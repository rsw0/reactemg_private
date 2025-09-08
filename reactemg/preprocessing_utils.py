import argparse
import random
import os
import re
import glob
import math
from typing import List, Tuple, Sequence
from nn_models import (
    Any2Any_Model,
    EDTCN_Model,
    LSTM_Model,
    ANN_Model,
)
from dataset import (
    Any2Any_Dataset,
    EDTCN_Dataset,
    LSTM_Dataset,
    ANN_Dataset,
)


def _collect_epn_subject_files(
    epn_root: str, num_classes: int, include_testing: bool = False
):
    """
    Return: dict {subject_folder_name -> [file_paths]}
    subject_folder_name is e.g. 'trainingJSON_user123'
    """
    subject_to_files = {}
    if not os.path.isdir(epn_root):
        raise ValueError(f"EPN data folder does not exist: {epn_root}")

    for subject_folder in os.listdir(epn_root):
        subj_path = os.path.join(epn_root, subject_folder)
        if not os.path.isdir(subj_path):
            continue

        # Control which subject folders are eligible for labeled data
        if subject_folder.startswith("testingJSON_user") and not include_testing:
            continue
        if not (
            subject_folder.startswith("trainingJSON_user")
            or subject_folder.startswith("testingJSON_user")
        ):
            # Ignore any odd folders
            continue

        # Flat (no deeper nesting); take only .npy
        files = []
        for fname in os.listdir(subj_path):
            if not fname.lower().endswith(".npy"):
                continue
            fname_lc = fname.lower()
            # still respect 3-class skips
            if num_classes == 3:
                if any(k in fname_lc for k in ("wavein", "waveout", "pinch")):
                    continue
            files.append(os.path.join(subj_path, fname))
        if files:
            subject_to_files[subject_folder] = sorted(files)
    return subject_to_files


def parse_tuple(s):
    try:
        return tuple(map(float, s.strip("()").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuples must be in the form x,y")


def get_csv_paths(
    dataset_selection,
    num_classes,
    roam_data_master_folder,
    roam_data_subfolders,
    public_data_folders,
    epn_data_master_folder,
    val_patient_ids,
    epn_subset_percentage,
    discard_labeled_percentage,
):
    """
    Construct dataset paths

    Parameters
    ----------
    dataset_selection : str
        Which dataset combination is selected:
          - roam_only
          - pub_only
          - pub_with_roam
          - epn_only
          - pub_with_roam_with_epn
          - pub_with_epn

    num_classes : int
        Number of classes (affects which EPN files are skipped).
    roam_data_master_folder : str
        Path to the main ROAM dataset folder.
    roam_data_subfolders : list of str
        Subfolders within the ROAM dataset that contain CSVs.
    public_data_folders : list of str
        Folders for the public datasets.
    epn_data_master_folder : str
        Path to the main EPN dataset folder.
    val_patient_ids : list of str, optional
        List of two-letter patient IDs for the ROAM validation set.
    epn_subset_percentage : float, optional
        Fraction of EPN files (0.0 - 1.0) to keep from the eligible set if
        dataset_selection == "pub_with_roam_with_epn" OR == "pub_with_epn".

    Returns
    -------
    train_paths : list of str
        Paths to selected training files.
    val_paths : list of str
        Paths to selected validation files.
    """

    if val_patient_ids is None:
        val_patient_ids = []

    # =========================================================
    # MODE: "pub_with_epn" and "epn_only" (SUBJECT-WISE EPN)
    # =========================================================
    if dataset_selection in ["pub_with_epn", "epn_only"]:
        # 1) Public data (only for pub_with_epn)
        pub_paths = []
        if dataset_selection == "pub_with_epn":
            for pub_folder in public_data_folders:
                pub_paths.extend(
                    glob.glob(os.path.join(pub_folder, "**", "*.csv"), recursive=True)
                )

        # 2) EPN by subject (labeled split)
        subject_to_files = _collect_epn_subject_files(
            epn_data_master_folder, num_classes=num_classes, include_testing=False
        )
        subjects = sorted(subject_to_files.keys())

        # Choose a SUBJECT subset for labeled training according to epn_subset_percentage
        # (deterministic because you seed random in main.py)
        random.shuffle(subjects)
        if epn_subset_percentage >= 1.0:
            labeled_subjects = set(subjects)
        else:
            n_lab_subj = max(1, int(len(subjects) * epn_subset_percentage))
            labeled_subjects = set(subjects[:n_lab_subj])

        epn_labeled_paths = []
        for s in labeled_subjects:
            epn_labeled_paths.extend(subject_to_files[s])

        # For pub_with_epn we also previously filtered to only open/fist; keep that if desired:
        if dataset_selection == "pub_with_epn":
            epn_labeled_paths = [
                p
                for p in epn_labeled_paths
                if (
                    "open" in os.path.basename(p).lower()
                    or "fist" in os.path.basename(p).lower()
                )
            ]

        # 3) Combine Public + EPN (labeled)
        combined_paths = pub_paths + epn_labeled_paths
        combined_paths.sort()

        # 4) Validation split (same simple 5% file-wise split as before; consider switching to subject-wise later)
        val_size = int(0.05 * len(combined_paths))
        val_paths = combined_paths[:val_size]
        train_paths = combined_paths[val_size:]

        # We will derive the set of labeled EPN subjects later from train/val paths

    else:
        # ================================
        # For all other modes, same logic
        # ================================
        train_paths = []
        val_paths = []

        # -----------------------------------------
        # 1) Process ROAM data
        # -----------------------------------------
        # We will always gather validation data from ROAM using val_patient_ids,
        # but only add to the training set if dataset_selection uses ROAM data.
        for folder in roam_data_subfolders:
            folder_path = os.path.join(roam_data_master_folder, folder)

            # Recursively find all .csv files
            csv_files = glob.glob(
                os.path.join(folder_path, "**", "*.csv"), recursive=True
            )

            for csv_file in csv_files:
                basename = os.path.basename(csv_file)
                root, _ = os.path.splitext(basename)
                patient_id = root.split("_", 1)[0]

                if patient_id in val_patient_ids:
                    val_paths.append(csv_file)
                elif dataset_selection in [
                    "roam_only",
                    "pub_with_roam",
                    "pub_with_roam_with_epn",
                ]:
                    train_paths.append(csv_file)

        # -----------------------------------------
        # 2) Process Public data
        # -----------------------------------------
        if dataset_selection in ["pub_only", "pub_with_roam", "pub_with_roam_with_epn"]:
            # Add any CSV from the public datasets to TRAIN set
            for pub_folder in public_data_folders:
                csv_files = glob.glob(
                    os.path.join(pub_folder, "**", "*.csv"), recursive=True
                )
                train_paths.extend(csv_files)

        # -----------------------------------------
        # 3) Process EPN data (if requested)
        # -----------------------------------------
        # 'epn_only' => use all EPN files, skipping wave/pinch if needed
        # 'pub_with_roam_with_epn' => same skipping logic, and sample epn_subset_percentage.
        if dataset_selection in ["pub_with_roam_with_epn"]:
            if not os.path.isdir(epn_data_master_folder):
                raise ValueError(
                    f"EPN data folder does not exist: {epn_data_master_folder}"
                )

            # Collect all valid EPN .npy file paths here, then sample if needed
            epn_file_paths = []

            for subject_folder in os.listdir(epn_data_master_folder):
                subject_folder_path = os.path.join(
                    epn_data_master_folder, subject_folder
                )

                # Skip anything that is not a directory
                if not os.path.isdir(subject_folder_path):
                    continue

                # Skip "testingJSON_user..." folders entirely
                if subject_folder.startswith("testingJSON_user"):
                    continue

                # Ensure no deeper nesting
                for entry in os.listdir(subject_folder_path):
                    sub_sub_path = os.path.join(subject_folder_path, entry)
                    if os.path.isdir(sub_sub_path):
                        raise ValueError(
                            f"Unexpected folder '{sub_sub_path}' under subject folder "
                            f"'{subject_folder_path}'. Only one level of nesting is allowed."
                        )

                # Gather the .npy files in this subject folder
                for fname in os.listdir(subject_folder_path):
                    if not fname.lower().endswith(".npy"):
                        continue

                    fname_lc = fname.lower()
                    full_path = os.path.join(subject_folder_path, fname)

                    # Skip logic depending on num_classes
                    if num_classes == 3:
                        # skip wavein, waveout, pinch
                        skip_substrings = ["wavein", "waveout", "pinch"]
                        if any(sub in fname_lc for sub in skip_substrings):
                            continue

                    epn_file_paths.append(full_path)

            if dataset_selection == "pub_with_roam_with_epn":
                # Keep only 'open' or 'fist' in filename
                epn_file_paths = [
                    p
                    for p in epn_file_paths
                    if (
                        "open" in os.path.basename(p).lower()
                        or "fist" in os.path.basename(p).lower()
                    )
                ]

                # Randomly sample epn_subset_percentage
                random.shuffle(epn_file_paths)
                subset_size = int(len(epn_file_paths) * epn_subset_percentage)
                epn_file_paths = epn_file_paths[:subset_size]

            # Add these EPN paths to train
            train_paths.extend(epn_file_paths)

    # Sorting then discarding (to ensure reproducibility)
    train_paths.sort()
    val_paths.sort()
    if discard_labeled_percentage > 0.0:
        # shuffle and discard
        random.shuffle(train_paths)
        num_to_discard = int(discard_labeled_percentage * len(train_paths))
        train_paths = train_paths[num_to_discard:]
        train_paths.sort()  # re-sort to show in ascending order (easy to debug)

    return train_paths, val_paths


def get_unlabeled_csv_paths(
    unlabeled_data_folder: str,
    labeled_paths_train: Sequence[str],
    labeled_paths_val: Sequence[str],
    unlabeled_percentage: float,
    exclude_subjects: Sequence[str] = None,
) -> List[str]:
    """
    Recursively gather .csv/.npy under `unlabeled_data_folder` (e.g. '../data/unlabeled_data'),
    apply EMG-EPN-612/testingJSON filtering, exclude whole EPN subjects via `exclude_subjects`,
    remove collisions with labeled train/val using normalized keys, then sub-sample
    by `unlabeled_percentage` deterministically.

    Args
    ----
    unlabeled_data_folder : str
        Root folder that contains an 'EMG-EPN-612' mirror (and optionally others).
    labeled_paths_train : Sequence[str]
        Labeled training file paths (for collision removal).
    labeled_paths_val : Sequence[str]
        Labeled validation file paths (for collision removal).
    unlabeled_percentage : float
        0.0 => none, 1.0 => all, (0,1) => deterministic sample.
    exclude_subjects : Sequence[str], optional
        EPN subject folder names (e.g. 'trainingJSON_user123') to EXCLUDE entirely
        from unlabeled selection. This enforces subject-disjoint splits.

    Returns
    -------
    List[str] : Sorted unlabeled file paths that passed all filters.
    """

    labeled_paths_train = labeled_paths_train or []
    labeled_paths_val = labeled_paths_val or []
    exclude_subjects = set(exclude_subjects or [])

    # Identify if current walk location is under the top-level EMG-EPN-612 subtree
    def _top_is_epn612(curr_root: str) -> bool:
        rel = os.path.relpath(curr_root, unlabeled_data_folder)
        if rel == ".":
            return False
        top_level = rel.split(os.sep, 1)[0]
        return top_level == "EMG-EPN-612"  # exact match (Linux case-sensitive)

    # Extract the EPN subject folder name (immediate child under EMG-EPN-612)
    def _epn_subject_from_path(p: str) -> str:
        parts = os.path.normpath(p).split(os.sep)
        if "EMG-EPN-612" in parts:
            i = parts.index("EMG-EPN-612")
            if i + 1 < len(parts):
                return parts[i + 1]
        # Fallback: parent dirname
        return os.path.basename(os.path.dirname(p.rstrip(os.sep)))

    # Normalize basenames by stripping optional "...unlabel" before extension
    UNLABEL_RE = re.compile(r"([_\-.]?unlabel)(?=\.[^.]+$)", re.IGNORECASE)

    def _normalize_basename(base: str) -> str:
        return UNLABEL_RE.sub("", base)

    # Key used for collision removal: (immediate_parent_dir, normalized_basename)
    def _key_parent_normbase(p: str) -> Tuple[str, str]:
        parent = os.path.basename(os.path.dirname(p.rstrip(os.sep)))
        base = os.path.basename(p)
        return (parent, _normalize_basename(base))

    # -------- Walk unlabeled tree --------
    all_unlabeled_paths: List[str] = []
    for root, dirs, files in os.walk(unlabeled_data_folder):
        if _top_is_epn612(root):
            # Fully prune any testingJSON* subtree
            if "testingjson" in os.path.basename(root).lower():
                dirs[:] = []  # don't descend further
                continue
            # Also prevent descending into any immediate testingJSON* children
            dirs[:] = [d for d in dirs if "testingjson" not in d.lower()]

        for fname in sorted(files):  # deterministic order
            lower = fname.lower()
            if not (lower.endswith(".csv") or lower.endswith(".npy")):
                continue

            full_path = os.path.join(root, fname)

            # If this file is under the EPN mirror, enforce subject exclusions
            if _top_is_epn612(root):
                subj = _epn_subject_from_path(full_path)
                if subj in exclude_subjects:
                    continue

            all_unlabeled_paths.append(full_path)

    print(
        f"[get_unlabeled_csv_paths] Found {len(all_unlabeled_paths)} raw unlabeled files after subject/testing filters."
    )

    # -------- Collision removal vs labeled train/val (handles '...unlabel' renames) --------
    labeled_keys = {
        _key_parent_normbase(p)
        for p in list(labeled_paths_train) + list(labeled_paths_val)
    }
    cleaned_unlabeled = [
        p for p in all_unlabeled_paths if _key_parent_normbase(p) not in labeled_keys
    ]
    print(
        f"[get_unlabeled_csv_paths] After removing collisions with labeled train/val: {len(cleaned_unlabeled)}"
    )

    cleaned_unlabeled = sorted(cleaned_unlabeled)

    # -------- Deterministic sub-sampling --------
    if unlabeled_percentage <= 0.0:
        final_unlabeled_paths: List[str] = []
    elif unlabeled_percentage >= 1.0:
        final_unlabeled_paths = cleaned_unlabeled
    else:
        num_to_keep = int(len(cleaned_unlabeled) * unlabeled_percentage)
        if num_to_keep > 0:
            rng = random.Random(0)  # deterministic
            final_unlabeled_paths = sorted(rng.sample(cleaned_unlabeled, num_to_keep))
        else:
            final_unlabeled_paths = []

    # Optional sanity: verify no excluded subjects slipped through
    if exclude_subjects:
        unlabeled_subjects = set()
        for p in final_unlabeled_paths:
            parts = os.path.normpath(p).split(os.sep)
            if "EMG-EPN-612" in parts:
                i = parts.index("EMG-EPN-612")
                if i + 1 < len(parts):
                    unlabeled_subjects.add(parts[i + 1])
        overlap = unlabeled_subjects & exclude_subjects
        if overlap:
            print(
                f"[get_unlabeled_csv_paths][WARNING] {len(overlap)} excluded EPN subjects still present: {sorted(list(overlap))[:5]} ..."
            )

    print(
        f"[get_unlabeled_csv_paths] Sub-sampled {len(final_unlabeled_paths)} unlabeled files "
        f"out of {len(cleaned_unlabeled)} possible."
    )
    return final_unlabeled_paths


def get_finetune_csv_paths(
    val_patient_ids: List[str],
    roam_root: str = "../data/ROAM_EMG",
    patient_root: str = "../data/patient_data",
) -> Tuple[List[str], List[str]]:
    """
    Build train/val file lists for the special 'finetune' setting.
    Parameters
    ----------
    val_patient_ids : List[str]
        Expect **exactly one** subject ID, e.g. ["s4"] or ["p1"].
    roam_root : str
        Root folder that contains ROAM_EMG subject directories (default: ../data/ROAM_EMG).
    patient_root : str
        Root folder that contains patient_data sub-directories (default: ../data/patient_data).
    Returns
    -------
    (train_paths, val_paths) : Tuple[List[str], List[str]]
        Sorted lists of absolute CSV paths for training and validation.
    """
    if len(val_patient_ids) != 1:
        raise ValueError(
            "get_finetune_csv_paths expects a single subject ID "
            "(e.g. ['s4'] or ['p1']), got: {}".format(val_patient_ids)
        )
    val_id = val_patient_ids[0].lower()
    train_files, val_files = [], []
    # ----------------------------------------------------------
    # ROAM subject (folder name exactly 'sX', where X is integer)
    # ----------------------------------------------------------
    if val_id.startswith("s"):
        subj_dir = os.path.join(roam_root, val_id)
        if not os.path.isdir(subj_dir):
            raise FileNotFoundError(f"No folder found: {subj_dir}")
        for fp in glob.glob(os.path.join(subj_dir, "*.csv")):
            name = os.path.basename(fp).lower()
            if any(k in name for k in ("static_resting", "static_reaching")):
                train_files.append(fp)
            elif any(k in name for k in ("static_hanging", "static_unsupported")):
                val_files.append(fp)
    # ----------------------------------------------------------
    # Patient (folder that *ends with* val_id, e.g. *p1 or *p16)
    # ----------------------------------------------------------
    elif val_id.startswith("p"):
        # strict “endswith” match so p1 ≠ p16
        matches = [
            d
            for d in os.listdir(patient_root)
            if os.path.isdir(os.path.join(patient_root, d)) and d.endswith(val_id)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one folder in {patient_root} ending in '{val_id}', "
                f"found: {matches}"
            )
        subj_dir = os.path.join(patient_root, matches[0])
        for fp in glob.glob(os.path.join(subj_dir, "*.csv")):
            if fp.endswith("_1.csv"):
                train_files.append(fp)
            elif fp.endswith("_2.csv"):
                val_files.append(fp)
    else:
        raise ValueError(f"val_patient_id must start with 's' or 'p', got '{val_id}'")
    return sorted(set(train_files)), sorted(set(val_files))


def initialize_dataset(
    args,
    labeled_csv_paths_train,
    unlabeled_csv_paths_train,
    labeled_csv_paths_val,
    mask_tokens_dict,
):
    if args.model_choice == "any2any":
        if not all(x < y for x, y in zip(args.task_selection, args.task_selection[1:])):
            raise ValueError(
                "task_selection is not in increasing order, which is not compatible with __getitem__()"
            )
        if args.use_mav_for_emg == 1:
            # compute the effective length
            # scale down lambda, transition_buffer, etc.
            effective_mav_length = (
                args.window_size - args.inner_window_size
            ) // args.mav_inner_stride + 1
            args.lambda_poisson = 2
            scale_factor = effective_mav_length / args.window_size
            args.transition_buffer = max(1, int(args.transition_buffer * scale_factor))

        dataset_train = Any2Any_Dataset(
            labeled_csv_paths_train,
            unlabeled_csv_paths_train,
            args.median_filter_size,
            args.window_size,
            args.offset,
            args.embedding_method,
            args.lambda_poisson,
            False,
            args.sampling_probability_poisson,
            args.poisson_mask_percentage_sampling_range,
            args.end_mask_percentage_sampling_range,
            args.task_selection,
            args.stage_1_weights,
            args.stage_2_weights,
            args.mask_alignment,
            args.transition_buffer,
            mask_tokens_dict,
            args.with_training_curriculum,
            args.num_classes,
            args.medfilt_order,
            args.noise,
            args.hand_choice,
            args.inner_window_size,
            args.use_mav_for_emg,
        )
        # Note that for the validation set, we always fix the masks by setting seeded_mask = True
        # seeded_mask controls not just whether the masks are reproducible, but also if training uses a different mask for each sample at every epoch
        # If the mask is not seeded, then it's free to change over the epoch
        # Also, the unlabeled dataset have placeholder values
        dataset_val = Any2Any_Dataset(
            labeled_csv_paths_val,
            [],
            args.median_filter_size,
            args.window_size,
            args.offset,
            args.embedding_method,
            args.lambda_poisson,
            True,
            args.sampling_probability_poisson,
            args.poisson_mask_percentage_sampling_range,
            args.end_mask_percentage_sampling_range,
            args.task_selection,
            args.stage_1_weights,
            args.stage_2_weights,
            args.mask_alignment,
            args.transition_buffer,
            mask_tokens_dict,
            args.with_training_curriculum,
            args.num_classes,
            args.medfilt_order,
            0.0,  # no noise
            args.hand_choice,
            args.inner_window_size,
            args.use_mav_for_emg,
        )

    elif args.model_choice == "ed_tcn":
        dataset_train = EDTCN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_train,
            inner_window_size=150,
            inner_stride=25,
        )
        dataset_val = EDTCN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_val,
            inner_window_size=150,
            inner_stride=25,
        )
    elif args.model_choice == "lstm":
        dataset_train = LSTM_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            csv_paths=labeled_csv_paths_train,
            num_classes=args.num_classes,
            precomputed_mean=None,
            precomputed_std=None,
        )
        dataset_val = LSTM_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            csv_paths=labeled_csv_paths_val,
            num_classes=args.num_classes,
            precomputed_mean=dataset_train.global_mean,
            precomputed_std=dataset_train.global_std,
        )
    elif args.model_choice == "ann":
        dataset_train = ANN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_train,
            num_classes=args.num_classes,
            use_precomputed_stats=False,
        )
        dataset_val = ANN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_val,
            num_classes=args.num_classes,
            use_precomputed_stats=True,
            precomputed_mean=dataset_train.mean_,
            precomputed_std=dataset_train.std_,
        )
    else:
        raise ValueError(f"Unknown model_choice: {args.model_choice}")

    return dataset_train, dataset_val


def initialize_model(args):
    if args.model_choice == "any2any":
        model = Any2Any_Model(
            args.embedding_dim,
            args.nhead,
            args.dropout,
            args.activation,
            args.num_layers,
            args.window_size,
            args.embedding_method,
            args.mask_alignment,
            args.share_pe,
            args.tie_weight,
            args.use_decoder,
            args.use_input_layernorm,
            args.num_classes,
            args.output_reduction_method,
            args.chunk_size,
            args.inner_window_size,
            args.use_mav_for_emg,
            args.mav_inner_stride,
        )
    elif args.model_choice == "ed_tcn":
        model = EDTCN_Model(
            num_channels=8,
            num_classes=args.num_classes,
            enc_filters=(128, 288),
            kernel_size=9,
        )
    elif args.model_choice == "lstm":
        model = LSTM_Model(
            input_size=8,
            fc_size=400,
            hidden_size=256,
            num_classes=args.num_classes,
        )
    elif args.model_choice == "ann":
        model = ANN_Model(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model_choice: {args.model_choice}")

    return model


def lr_lambda_cosine(current_step, warmup_steps, total_steps):
    """
    Cosine schedule with warmup:
    During warmup_steps: increases linearly from 0 to 1
    After warmup_steps:  0.5*(1 + cos(pi * progress))
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_lambda_linear(current_step, warmup_steps, total_steps):
    """
    Linear schedule with warmup:
    During warmup_steps: increases linearly from 0 to 1
    After warmup_steps:  decreases linearly from 1 to 0
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Linear decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.0, 1.0 - progress)


def lr_lambda_exponential(current_step, warmup_steps, total_steps, decay_rate):
    """
    Exponential schedule with linear warmup:
      - During warmup_steps: increases LR linearly from 0 to 1.
      - After warmup_steps:  decays LR via exp(-decay_rate * progress),
                             where progress goes from 0 to 1 across the
                             remaining steps.
    """
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    # Exponential decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    # progress goes from 0 at step == warmup_steps to 1 at step == total_steps
    return math.exp(-decay_rate * progress)
