import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load CSV files
lora_path = 'output/lora_finetune_with_la.csv'
no_path = 'output/no_finetune_with_la.csv'

df_lora = pd.read_csv(lora_path)
df_no = pd.read_csv(no_path)

# Round to two decimals
for col in ['avg_raw_accuracy', 'avg_transition_accuracy']:
    df_lora[col] = df_lora[col].round(2)
    df_no[col] = df_no[col].round(2)

# Merge on subject_id
merged = pd.merge(
    df_lora,
    df_no,
    on='subject_id',
    suffixes=('_lora', '_no')
)

# Compute delta percentages (difference * 100)
merged['delta_raw_pct'] = (merged['avg_raw_accuracy_lora'] - merged['avg_raw_accuracy_no']) * 100
merged['delta_transition_pct'] = (merged['avg_transition_accuracy_lora'] - merged['avg_transition_accuracy_no']) * 100

# Helper for plotting
def plot_delta(df, delta_col, title):
    # Filter out zero deltas
    df_nonzero = df[df[delta_col] != 0].copy()
    # Sort by delta descending
    df_nonzero.sort_values(by=delta_col, ascending=False, inplace=True)
    
    subjects = df_nonzero['subject_id']
    deltas = df_nonzero[delta_col]
    
    # Colors based on sign
    colors = ['green' if val > 0 else 'red' for val in deltas]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects, deltas, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(title)
    plt.ylabel('Delta percentage points')
    plt.xlabel('Subject')
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    legend_elements = [Patch(facecolor='green', label='positive delta'),
                       Patch(facecolor='red', label='negative delta')]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

# Plot for raw accuracy
plot_delta(merged, 'delta_raw_pct', 'Per-subject raw accuracy')

# Plot for transition accuracy
plot_delta(merged, 'delta_transition_pct', 'Per-subject transition accuracy')
