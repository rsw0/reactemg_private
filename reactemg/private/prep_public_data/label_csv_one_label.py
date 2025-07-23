import os
import pandas as pd
import matplotlib.pyplot as plt

def interactive_label_emg(csv_file_path):
    """
    A simpler labeling script:
    1. Asks the user once: 'Which label do you want (1 or 2)?'
    2. Collects transition points (where you click).
    3. Press 'u' anytime to undo the last transition.
    4. After you close the plot, it applies the toggle logic:
       - Start at 0, then toggle to user_label, then back to 0, etc.
    5. Saves the result as <originalfilename>_labeled.csv.
    """

    # -------------------------
    # 1) Ask user for a single label, default to 1 if invalid
    # -------------------------
    chosen_label_str = input(
        "Please enter the label to use for this session (1 or 2): "
    ).strip()
    try:
        user_label = int(chosen_label_str)
        if user_label not in [1, 2]:
            print("Invalid label. Defaulting to 1.")
            user_label = 1
    except ValueError:
        print("Invalid input. Defaulting to 1.")
        user_label = 1

    print(f"Using label = {user_label} for transitions.")

    # -------------------------
    # 2) Read CSV and prepare
    # -------------------------
    df = pd.read_csv(csv_file_path)
    # Initialize a 'gt' column to 0
    df['gt'] = 0

    transitions = []       # will store [x_index1, x_index2, ...]
    transition_lines = []  # store the axvline objects so we can remove them on undo

    # -------------------------
    # 3) Plot EMG signals
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    emg_columns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']
    for col in emg_columns:
        ax.plot(df.index, df[col], label=col)

    ax.set_xlabel('CSV Index (Row Number)')
    ax.set_ylabel('EMG Amplitude')
    ax.set_title(
        "Simplified EMG Labeling\n"
        "Left-click to mark transitions, press 'u' to undo last transition."
    )
    ax.legend(loc='upper right')
    ax.grid(True)

    # -------------------------
    # 4) Define callbacks
    # -------------------------
    def on_click(event):
        if event.inaxes == ax and event.xdata is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < len(df):
                # Record this transition
                transitions.append(idx)
                line_obj = ax.axvline(idx, color='r', linestyle='--')
                transition_lines.append(line_obj)
                fig.canvas.draw()
                print(f"Recorded transition at index={idx}.")

    def on_key_press(event):
        if event.key == 'u':
            if transitions:
                removed_idx = transitions.pop()
                removed_line = transition_lines.pop()
                removed_line.remove()
                fig.canvas.draw()
                print(f"Undo! Removed last transition at index={removed_idx}.")
            else:
                print("No transitions to undo.")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()  # Blocking until window is closed

    # -------------------------
    # 5) Apply toggle logic after user closes the plot
    # -------------------------
    transitions.sort()  # sort ascending
    print("\nApplying toggle logic with user_label=", user_label)

    current_start_idx = 0
    current_label = 0  # Start at 0

    for t_idx in transitions:
        # Label from current_start_idx up to t_idx with current_label
        df.loc[current_start_idx:t_idx, 'gt'] = current_label
        # Toggle
        if current_label == 0:
            current_label = user_label
        else:
            current_label = 0
        # Update start
        current_start_idx = t_idx + 1

    # Label everything after the last transition
    df.loc[current_start_idx:, 'gt'] = current_label

    # Ensure the 'gt' column is strictly integer
    df['gt'] = df['gt'].astype(int)

    # -------------------------
    # 6) Save <filename>_labeled.csv
    # -------------------------
    input_dir = os.path.dirname(csv_file_path)
    base_name = os.path.basename(csv_file_path)
    file_stem, file_ext = os.path.splitext(base_name)
    labeled_filename = f"{file_stem}_labeled{file_ext}"
    output_path = os.path.join(input_dir, labeled_filename)

    df.to_csv(output_path, index=False)
    print(f"\nAll done! Labeled data saved to: {output_path}")


# -----------------------
# Usage Example
# -----------------------
if __name__ == "__main__":
    input_csv = "All_EMG_datasets/SS-STM_for_MyoDatasets_temp/sub2/M13T5_close.csv"
    interactive_label_emg(input_csv)
