'''
import pandas as pd
import matplotlib.pyplot as plt

def plot_emg_signals(csv_file_path):
    # 1. Read the CSV file
    df = pd.read_csv(csv_file_path)

    df = df.iloc[:4000]
    
    # 2. Plot each EMG column against the CSV index
    plt.figure(figsize=(10, 6))
    
    # We assume your CSV has columns named 'emg1', 'emg2', ..., 'emg8'.
    emg_columns = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
    
    for col in emg_columns:
        # Plot using DataFrame index on the x-axis, EMG column on the y-axis
        plt.plot(df.index, df[col], label=col)
    
    # 3. Configure the plot
    plt.xlabel('CSV Index (Row Number)')
    plt.ylabel('EMG Amplitude')
    plt.title('EMG Signals (emg1 – emg8) vs. CSV Index')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 4. Show the plot
    plt.savefig('csv_file.png')


# Example usage:
if __name__ == "__main__":
    csv_path = "d4y7fm3g79-1/User 3/Closed Grip/u3-closed-fist-set-1.csv"  # Replace with your actual CSV path
    plot_emg_signals(csv_path)
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

def interactive_label_emg(csv_file_path, output_csv_file='labeled.csv'):
    """
    Interactively label EMG data by clicking on the plot.
    - Left-click to pick a time index, then enter a label (0, 1, or 2) in the console.
      If invalid, defaults to 1.
    - Press the 'u' key anytime to undo (remove) the last recorded transition.
    When you close the plot:
        - All transition points are sorted by index.
        - The label is assigned in ascending order of index.
        - The final labeled DataFrame is saved.
    """

    # 1. Read the CSV file
    df = pd.read_csv(csv_file_path)

    # 2. Create a label column, initialize to 0
    df['label'] = 0

    # We'll store clicks here as tuples: (click_index, chosen_label)
    transitions = []
    # We'll store the corresponding line objects so we can remove them if we undo
    transition_lines = []

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot EMG signals
    emg_columns = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
    for col in emg_columns:
        ax.plot(df.index, df[col], label=col)

    ax.set_xlabel('CSV Index (Row Number)')
    ax.set_ylabel('EMG Amplitude')
    ax.set_title('Interactive EMG Labeling\n(Left-click to label, press "u" to undo last transition)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # -----------------------
    # Mouse click event: add a transition
    # -----------------------
    def on_click(event):
        # Only consider clicks inside the main axes
        if event.inaxes == ax and event.xdata is not None:
            click_index = int(round(event.xdata))
            if 0 <= click_index < len(df):
                # Ask user which label they want (0, 1, or 2)
                chosen_label_str = input(
                    f"You clicked near index {click_index}. "
                    "Enter a label (0, 1, 2): "
                ).strip()

                # Try to parse it as an integer label
                try:
                    chosen_label = int(chosen_label_str)
                    if chosen_label not in [0, 1, 2]:
                        print("Invalid label. Defaulting to 1.")
                        chosen_label = 1
                except ValueError:
                    print("Invalid input. Defaulting to 1.")
                    chosen_label = 1

                # Record the transition
                transitions.append((click_index, chosen_label))

                # Visually mark the transition on the plot
                line_obj = ax.axvline(click_index, color='r', linestyle='--')
                transition_lines.append(line_obj)
                fig.canvas.draw()

                print(f"Recorded transition at index={click_index}, label={chosen_label}")

    # -----------------------
    # Key press event: 'u' to undo last transition
    # -----------------------
    def on_key_press(event):
        if event.key == 'u':
            if transitions:
                removed_transition = transitions.pop()
                removed_line = transition_lines.pop()
                removed_line.remove()  # remove the red vertical line from the plot
                fig.canvas.draw()
                print(f"Undid the last transition: index={removed_transition[0]}, label={removed_transition[1]}")
            else:
                print("No transitions to undo.")

    # 4. Connect the event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # 5. Show the interactive plot (blocking)
    plt.show()

    # 6. Once the window is closed, we apply the transitions in ascending order
    transitions.sort(key=lambda x: x[0])  # sort by click_index

    # We'll "walk" through the DataFrame in ascending index order
    current_start_idx = 0
    current_label = 0

    for (transition_idx, transition_label) in transitions:
        # Assign the current_label from current_start_idx up to (but not including) transition_idx
        df.loc[current_start_idx:transition_idx-1, 'label'] = current_label
        # Now, from this transition_idx onward, we set transition_label (until the next transition)
        current_start_idx = transition_idx
        current_label = transition_label

    # Finally, assign the last label from the final transition onward
    df.loc[current_start_idx:, 'label'] = current_label

    # 7. Save the labeled data with the original file name plus "_labeled"
    import os
    input_dir = os.path.dirname(csv_file_path)
    base_name = os.path.basename(csv_file_path)
    file_stem, file_ext = os.path.splitext(base_name)
    labeled_filename = f"{file_stem}_labeled{file_ext}"
    output_path = os.path.join(input_dir, labeled_filename)

    df.to_csv(output_path, index=False)
    print(f"\nAll done! Labeled data saved to: {output_path}")


# --------------------------
# Usage Example (Run as main)
# --------------------------
if __name__ == "__main__":
    input_csv = "d4y7fm3g79-1/User 1/Closed Grip/u1-closed-fist-set-1.csv"
    interactive_label_emg(input_csv)
