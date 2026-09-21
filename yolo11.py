# -*- coding: utf-8 -*-
"""
yolo11.py

This script is designed to train a YOLO (You Only Look Once) model for object detection.
It performs several key steps:
1.  **Environment Check**: Verifies that the NVIDIA GPU is accessible and the `ultralytics` library is installed and properly configured.
2.  **Model Training**: Initiates the training of a YOLOv11 model on a specified dataset using command-line arguments.
3.  **Model Saving**: After training, it copies the best-performing model to a designated directory for later use in inference.
"""

import subprocess  # This module is used to run external commands, like `nvidia-smi` and the `yolo` training command.
import sys         # This module provides access to system-specific parameters and functions, used here to find the Python executable.
import shutil      # This module offers high-level file operations, such as copying the trained model file.
from pathlib import Path  # This provides a more object-oriented way to handle filesystem paths, making path manipulation cleaner.
import ultralytics # The core library for working with YOLO models, providing the training and inference functionalities.

# This block executes the command `nvidia-smi` to check the status of the NVIDIA GPU.
# This is a crucial first step in any deep learning project to ensure the hardware is properly
# recognized and available for accelerated computations.
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# The output from the `nvidia-smi` command is printed to the console, showing details about the GPU.
print(result.stdout)

# This line uses the `pip` package manager to install the `ultralytics` library.
# It ensures that the necessary framework for YOLO is present in the current Python environment.
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

# The `ultralytics.checks()` function performs a series of system checks to ensure
# that the environment (dependencies, hardware, etc.) is correctly configured for the library.
ultralytics.checks()

# This line imports the `YOLO` class from the `ultralytics` library, which is the
# main entry point for creating, training, and using YOLO models.
from ultralytics import YOLO

# This block defines the command to start the training process for the YOLO model.
# The command is broken down into a list of strings for better readability and to be
# passed to `subprocess.run`.
training_command = [
    "yolo",           # The main executable for the ultralytics framework.
    "task=detect",    # Specifies the task to be performed is object detection.
    "mode=train",     # Sets the mode to training.
    "data=/home/mml_sc1/FER2013_dataset/data.yaml", # The path to the dataset configuration file.
    "model=yolo11n.pt", # Specifies the pre-trained YOLOv11 nano model to be used as a starting point.
    "epochs=50",      # Sets the number of training iterations (epochs) to 50.
    "imgsz=640"       # Defines the input image size for training.
]
# This line executes the training command and captures the output.
result = subprocess.run(training_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# The training logs (including progress and performance metrics) are printed here.
print(result.stdout)
# This conditional block checks for and prints any error messages that may have occurred during training.
if result.stderr:
    print("Errors during training:")
    print(result.stderr)

# After training, the model is automatically saved to a default location.
# These lines define the path to the best-performing model (`best.pt`) and the desired destination.
source_model_path = 'runs/train/exp/weights/best.pt'  # Path where the trained model is saved by default.
destination_path = '/home/mml_sc1/yolov11_trained_models/best_model.pt'  # The final destination for the model.

# This line uses `pathlib` to ensure the destination directory exists.
# `parents=True` creates any necessary parent directories, and `exist_ok=True`
# prevents an error if the directory already exists.
Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

# This line copies the trained model file from its temporary training location to the permanent destination.
shutil.copy(source_model_path, destination_path)

# Finally, a confirmation message is printed to the user, indicating the model's new location.
print(f"Model saved at {destination_path}")