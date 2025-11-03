"""
Unified channel analysis script for EMG datasets.
Supports both .npy (EMG-EPN-612) and .csv (ROAM-EMG) formats.

Usage:
    python analyze_channels.py <dataset_path> [--output <prefix>]

Examples:
    python analyze_channels.py data/EMG-EPN-612
    python analyze_channels.py data/ROAM_EMG --output roam_analysis
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_emg_data(file_path):
    """
    Load EMG data from either .npy or .csv file.
    Returns data in (C, T) format where C=channels, T=time samples.
    """
    file_path = Path(file_path)

    if file_path.suffix == '.npy':
        # Load numpy file
        data = np.load(file_path)
        # For EPN dataset: column 0 is ground truth, columns 1-8 are EMG channels
        # Extract only EMG channels (skip column 0 which is ground truth)
        if data.ndim == 2 and data.shape[1] >= 2:
            # Check if first column might be ground truth (typically has few unique values)
            if len(np.unique(data[:, 0])) < 20:  # Heuristic: labels have few unique values
                data = data[:, 1:]  # Skip first column (ground truth)

        # Transpose if needed (expecting C, T format)
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data

    elif file_path.suffix == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path)

        # Extract only emg0, emg1, ..., emgN columns
        emg_cols = [col for col in df.columns if col.startswith('emg') and col[3:].isdigit()]
        emg_cols = sorted(emg_cols, key=lambda x: int(x[3:]))  # Sort by channel number

        if len(emg_cols) == 0:
            return None

        # Extract EMG data as array (T, C) then transpose to (C, T)
        data = df[emg_cols].values.T
        return data

    else:
        return None


def analyze_dataset(dataset_path, output_prefix="channel_analysis"):
    """
    Analyze all EMG files in a dataset and generate statistics and plots.
    """
    dataset_path = Path(dataset_path)

    # Find all relevant files
    print("Scanning dataset for EMG files...")
    all_files = list(dataset_path.rglob("*.npy")) + list(dataset_path.rglob("*.csv"))
    print(f"Found {len(all_files)} files")

    if len(all_files) == 0:
        print("No .npy or .csv files found in dataset path")
        return

    # Statistics storage
    all_channels_mean = []
    all_channels_std = []
    all_channels_max = []
    all_channels_var = []
    all_channels_rms = []

    file_count = 0

    print("\nAnalyzing all files...")
    for file_path in tqdm(all_files):
        try:
            data = load_emg_data(file_path)

            if data is None:
                continue

            C, T = data.shape

            # Calculate statistics per channel
            channel_mean = np.mean(np.abs(data), axis=1)
            channel_std = np.std(data, axis=1)
            channel_max = np.max(np.abs(data), axis=1)
            channel_var = np.var(data, axis=1)
            channel_rms = np.sqrt(np.mean(data**2, axis=1))

            all_channels_mean.append(channel_mean)
            all_channels_std.append(channel_std)
            all_channels_max.append(channel_max)
            all_channels_var.append(channel_var)
            all_channels_rms.append(channel_rms)

            file_count += 1

        except Exception as e:
            # Silently skip problematic files
            continue

    print(f"\nProcessed {file_count} files successfully")

    if file_count == 0:
        print("No valid files processed")
        return

    # Convert to arrays for analysis
    all_channels_mean = np.array(all_channels_mean)  # (N_files, C)
    all_channels_std = np.array(all_channels_std)
    all_channels_max = np.array(all_channels_max)
    all_channels_var = np.array(all_channels_var)
    all_channels_rms = np.array(all_channels_rms)

    print(f"Data shape: {all_channels_mean.shape}")
    print(f"Number of channels: {all_channels_mean.shape[1]}")

    # Compute statistics across all files per channel
    mean_per_channel = np.mean(all_channels_mean, axis=0)
    std_per_channel = np.mean(all_channels_std, axis=0)
    max_per_channel = np.mean(all_channels_max, axis=0)
    var_per_channel = np.mean(all_channels_var, axis=0)
    rms_per_channel = np.mean(all_channels_rms, axis=0)

    # Print statistics
    print("\n" + "="*70)
    print("CHANNEL STATISTICS ACROSS ENTIRE DATASET")
    print("="*70)

    print(f"\nAverage MEAN amplitude per channel:")
    for i, val in enumerate(mean_per_channel):
        ratio = val / mean_per_channel.mean() if mean_per_channel.mean() > 0 else 1.0
        print(f"  Channel {i}: {val:.4f} (ratio to average: {ratio:.3f})")

    print(f"\nAverage STD per channel:")
    for i, val in enumerate(std_per_channel):
        ratio = val / std_per_channel.mean() if std_per_channel.mean() > 0 else 1.0
        print(f"  Channel {i}: {val:.4f} (ratio to average: {ratio:.3f})")

    print(f"\nAverage RMS per channel:")
    for i, val in enumerate(rms_per_channel):
        ratio = val / rms_per_channel.mean() if rms_per_channel.mean() > 0 else 1.0
        print(f"  Channel {i}: {val:.4f} (ratio to average: {ratio:.3f})")

    print(f"\nAverage MAX amplitude per channel:")
    for i, val in enumerate(max_per_channel):
        ratio = val / max_per_channel.mean() if max_per_channel.mean() > 0 else 1.0
        print(f"  Channel {i}: {val:.4f} (ratio to average: {ratio:.3f})")

    # Identify potentially problematic channels
    print(f"\n" + "="*70)
    print("CHANNEL HEALTH ASSESSMENT")
    print("="*70)

    problematic_channels = []
    for i in range(len(mean_per_channel)):
        other_channels_mean = np.mean([mean_per_channel[j] for j in range(len(mean_per_channel)) if j != i])
        ratio = mean_per_channel[i] / (other_channels_mean + 1e-8)

        if ratio < 0.2:
            status = "VERY LOW (< 20% of others)"
            problematic_channels.append(i)
        elif ratio < 0.5:
            status = "LOW (< 50% of others)"
            problematic_channels.append(i)
        elif ratio > 2.0:
            status = "VERY HIGH (> 200% of others)"
            problematic_channels.append(i)
        elif ratio > 1.5:
            status = "HIGH (> 150% of others)"
        else:
            status = "OK"

        print(f"  Channel {i}: {status} (ratio={ratio:.3f})")

    if problematic_channels:
        print(f"\nPotentially problematic channels: {problematic_channels}")
    else:
        print(f"\nAll channels appear healthy!")

    print("="*70)

    # Subject-level analysis
    print("\n" + "="*70)
    print("ANALYZING BY SUBJECT...")
    print("="*70)

    subject_stats = {}
    for file_path in tqdm(all_files, desc="Analyzing by subject"):
        try:
            data = load_emg_data(file_path)
            if data is None:
                continue

            # Extract subject ID from path
            parts = str(file_path).split('/')
            # Look for patterns like 'user123' or 's12'
            subject_dir = None
            for p in parts:
                if 'user' in p.lower():
                    subject_dir = p
                    break
                elif p.startswith('s') and len(p) <= 4 and p[1:].isdigit():
                    subject_dir = p
                    break

            if not subject_dir:
                continue

            subject = subject_dir
            channel_mean = np.mean(np.abs(data), axis=1)

            if subject not in subject_stats:
                subject_stats[subject] = []

            subject_stats[subject].append(channel_mean)
        except:
            continue

    # Aggregate subject statistics
    print(f"\nTotal unique subjects found: {len(subject_stats)}")

    if len(subject_stats) > 0:
        print(f"\nShowing statistics for first 30 subjects:")
        for subject in sorted(subject_stats.keys())[:30]:
            stats = np.array(subject_stats[subject])
            channel_avgs = np.mean(stats, axis=0)

            # Check for problematic patterns
            problematic = []
            for i in range(len(channel_avgs)):
                other_mean = np.mean([channel_avgs[j] for j in range(len(channel_avgs)) if j != i])
                ratio = channel_avgs[i] / (other_mean + 1e-8)
                if ratio < 0.5:
                    problematic.append(f"Ch{i}")

            status = f"Problematic: {', '.join(problematic)}" if problematic else "OK"
            print(f"{subject}: {status}")

    print("="*70)

    # Generate plots
    generate_plots(all_channels_mean, all_channels_std, all_channels_rms,
                   mean_per_channel, std_per_channel, rms_per_channel,
                   output_prefix)


def generate_plots(all_channels_mean, all_channels_std, all_channels_rms,
                   mean_per_channel, std_per_channel, rms_per_channel,
                   output_prefix):
    """
    Generate visualization plots for channel analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    num_channels = len(mean_per_channel)
    channels = np.arange(num_channels)

    # Plot 1: Mean amplitude per channel
    ax1 = axes[0, 0]
    ax1.bar(channels, mean_per_channel, color='steelblue')
    ax1.set_xlabel('Channel', fontsize=12)
    ax1.set_ylabel('Average Mean Amplitude', fontsize=12)
    ax1.set_title('Average Mean Amplitude per Channel', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(channels)

    # Plot 2: Standard deviation per channel
    ax2 = axes[0, 1]
    ax2.bar(channels, std_per_channel, color='steelblue')
    ax2.set_xlabel('Channel', fontsize=12)
    ax2.set_ylabel('Average Standard Deviation', fontsize=12)
    ax2.set_title('Average Standard Deviation per Channel', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(channels)

    # Plot 3: RMS per channel
    ax3 = axes[1, 0]
    ax3.bar(channels, rms_per_channel, color='steelblue')
    ax3.set_xlabel('Channel', fontsize=12)
    ax3.set_ylabel('Average RMS', fontsize=12)
    ax3.set_title('Average RMS per Channel', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(channels)

    # Plot 4: Distribution of channel amplitudes (boxplot)
    ax4 = axes[1, 1]
    positions = np.arange(num_channels)
    bp = ax4.boxplot([all_channels_mean[:, i] for i in range(num_channels)],
                       positions=positions,
                       widths=0.6,
                       patch_artist=True,
                       showfliers=False)
    # Color all channels uniformly
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax4.set_xlabel('Channel', fontsize=12)
    ax4.set_ylabel('Mean Amplitude Distribution', fontsize=12)
    ax4.set_title('Distribution of Mean Amplitude per Channel', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(positions)

    plt.tight_layout()
    output_file = f'{output_prefix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze EMG channels across a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('dataset_path', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='channel_analysis',
                       help='Output file prefix (default: channel_analysis)')

    args = parser.parse_args()

    analyze_dataset(args.dataset_path, args.output)


if __name__ == "__main__":
    main()
