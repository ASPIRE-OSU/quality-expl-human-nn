import os
import random
import csv


root_dir = os.getcwd() # Adjust to the parent directory containing batch folders
num_batches = 18
num_pairs = 16
audios_per_pair = 2
mapping_csv_path = os.path.join(root_dir, 'map_to_original_name.csv')

# Initialize mapping list
mapping_rows = []

for batch_num in range(num_batches):
    batch_name = f"batch{batch_num}"
    batch_path = os.path.join(root_dir, batch_name)

    # Collect all original pairs
    pairs = []
    for pair_num in range(0, num_pairs):
        pair = []
        for audio_num in range(1, audios_per_pair + 1):
            filename = f"{batch_name}_pair{pair_num}-audio{audio_num}.wav"
            filepath = os.path.join(batch_path, filename)
            pair.append(filepath)
        pairs.append(pair)

    # Shuffle pairs
    random.shuffle(pairs)

    # Rename to temporary files to avoid name collisions
    temp_pairs = []
    for idx, pair in enumerate(pairs):
        temp_pair = []
        for j, file in enumerate(pair):
            old_filename = os.path.basename(file)
            temp_filename = f"{batch_name}_TEMP_{idx}-audio{j+1}.wav"
            temp_path = os.path.join(batch_path, temp_filename)
            os.rename(file, temp_path)
            temp_pair.append((temp_path, old_filename))  # Save old name
        temp_pairs.append(temp_pair)

    # Final renaming and recording mapping
    for new_pair_index, pair in enumerate(temp_pairs, start=1):
        for audio_index, (temp_path, old_filename) in enumerate(pair, start=1):
            new_filename = f"{batch_name}_pair{new_pair_index}-audio{audio_index}.wav"
            new_path = os.path.join(batch_path, new_filename)
            os.rename(temp_path, new_path)

            # Add to mapping rows
            mapping_rows.append({
                "batch": batch_name,
                "old_filename": old_filename,
                "new_filename": new_filename
            })

# Write or append to CSV file
csv_exists = os.path.exists(mapping_csv_path)
with open(mapping_csv_path, mode='w', newline='') as csvfile:
    fieldnames = ["batch", "old_filename", "new_filename"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if not csv_exists:
        writer.writeheader()

    writer.writerows(mapping_rows)

print(f"Shuffling complete. Mapping written to {mapping_csv_path}")
