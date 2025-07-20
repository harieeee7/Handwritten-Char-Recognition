import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_names():
    return [chr(i) for i in range(ord('A'), ord('Z') + 1)]
