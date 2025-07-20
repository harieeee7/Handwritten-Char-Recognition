import pandas as pd
import numpy as np
from PIL import Image
import os

# Load CSV
df = pd.read_csv("data/emnist-letters-test.csv")
os.makedirs("samples", exist_ok=True)

# Save 10 samples per letter (1–26 = A–Z)
counts = [0] * 26
for i, row in df.iterrows():
    label = int(row[0]) - 1  # 1–26 → 0–25
    if counts[label] < 10:
        img = row[1:].values.reshape(28, 28).astype(np.uint8)
        img = np.transpose(img, (1, 0))  # fix EMNIST orientation
        im = Image.fromarray(img)
        im.save(f"samples/{chr(label + 65)}_{counts[label]}.png")
        counts[label] += 1
    if sum(counts) >= 260:
        break

print("✅ 10 samples for each letter A–Z saved in /samples")
