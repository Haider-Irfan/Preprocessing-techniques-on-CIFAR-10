import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def preprocess_image(img):
    """Applies the chain of 5 preprocessing techniques."""
    
    # 1. RESIZE (32x32 → 64x64)
    processed = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    processed = processed.astype("float32")

    # 2. NORMALIZE (0–1)
    processed = processed / 255.0

    # 3. DENOISING
    processed = cv2.GaussianBlur(processed, (3, 3), 0)

    # 4. CONTRAST ADJUSTMENT
    alpha = 1.5
    processed = np.clip(processed * alpha, 0, 1)

    # 5. AUGMENTATION
    processed = cv2.flip(processed, 1)   # horizontal flip
    processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)

    return processed



df = pd.read_csv("C:/Users/IAT/Documents/archive (1)/train.csv")


IMG_INDEX = 0
row = df.iloc[IMG_INDEX].values

# Reconstruct image (CIFAR10 format)
pixels = row[1:]   # first col = label
r = pixels[0:1024].reshape(32, 32)
g = pixels[1024:2048].reshape(32, 32)
b = pixels[2048:].reshape(32, 32)

img_original = np.dstack((r, g, b)).astype('uint8')

# PREPROCESS
img_processed = preprocess_image(img_original)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(img_original.ravel(), bins=256, color='black')
plt.title("Histogram – Original Image")
plt.xlabel("Pixel Value")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_processed)
plt.title("Processed Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(img_processed.ravel(), bins=256, color='blue')
plt.title("Histogram – Processed Image")
plt.xlabel("Pixel Value")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# Resize original to match 64×64
orig_resized = cv2.resize(img_original, (64, 64)).astype("float32") / 255.0

diff = np.abs(img_processed - orig_resized)

plt.figure(figsize=(4, 4))
plt.imshow(diff, cmap="inferno")
plt.title("Difference Heatmap")
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure(figsize=(5, 3))
plt.hist(diff.flatten(), bins=50)
plt.title("Pixel Difference Histogram")
plt.xlabel("Difference Value")
plt.ylabel("Count")
plt.show()
