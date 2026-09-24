import cv2
import tensorflow as tf
import torch
import transformers

print("========== LIBRARY INFORMATION ==========")
print("OpenCV Version       :", cv2.__version__)
print("TensorFlow Version   :", tf.__version__)
print("PyTorch Version      :", torch.__version__)
print("Transformers Version :", transformers.__version__)

print("\n========== HARDWARE INFORMATION ==========")
print("TensorFlow GPU Devices :", tf.config.list_physical_devices("GPU"))
print("PyTorch CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
