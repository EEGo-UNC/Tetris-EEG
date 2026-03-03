import kagglehub
import shutil
import os


# Download latest version
path = kagglehub.dataset_download("manh123df/deap-dataset")

print("Path to dataset files:", './'+path)


# Define the source and destination paths
destination_path = "./datasets/DEAP"  # Replace with the actual path where you want to move the folder

try:
    # Move the folder
    shutil.move(path, destination_path)
    print(f"Folder '{path}' successfully moved to '{destination_path}'.")
except FileNotFoundError:
    print(f"Error: Source folder '{path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")