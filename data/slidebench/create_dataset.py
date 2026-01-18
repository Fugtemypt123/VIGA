"""Script to create dataset CSV from slide examples."""
import csv
import os

# Define the root directory where all the folders are located
ROOT_DIR = "../slidesbench/examples"

# Define the output CSV file
OUTPUT_FILE = "dataset.csv"

def create_dataset(root_dir):
    dataset = []

    # Walk through the directory tree starting at root_dir
    for root, dirs, files in os.walk(root_dir):
        # Check if the current folder contains both 'instruction_model.txt' and 'generate_presentation.py'
        instruction_file = os.path.join(root, "instruction_model.txt")
        code_file = os.path.join(root, "generate_presentation.py")
        
        if os.path.exists(instruction_file) and os.path.exists(code_file):
            # Read the instruction
            with open(instruction_file, "r") as f:
                instruction = f.read()

            # Read the code
            with open(code_file, "r") as f:
                code = f.read()

            # Add the data to the dataset list (one entry per pair)
            dataset.append({
                "instruction": instruction,
                "code": code,
                "slide_folder": os.path.abspath(root)  # Optional: folder name for tracking
            })

    # Write the dataset to a CSV file
    with open(OUTPUT_FILE, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['instruction', 'code', 'slide_folder']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write rows of data
        for data in dataset:
            writer.writerow(data)

    print(f"Dataset created successfully and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset(ROOT_DIR)
