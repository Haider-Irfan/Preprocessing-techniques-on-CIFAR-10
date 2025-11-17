import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# LOAD CIFAR-10 CSV IMAGE

df = pd.read_csv("C:/Users/IAT/Documents/archive (1)/train.csv")

IMG_INDEX = 0
row = df.iloc[IMG_INDEX].values

pixels = row[1:]   # first column = label
r = pixels[0:1024].reshape(32, 32)
g = pixels[1024:2048].reshape(32, 32)
b = pixels[2048:].reshape(32, 32)

img_original = np.dstack((r, g, b)).astype('uint8')


# SHOW EVERYTHING IN ONE FRAME

def show_side_by_side(title, processed_img):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

   
    axes[0].imshow(img_original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

   
    axes[1].hist(img_original.ravel(), bins=256, color='black')
    axes[1].set_title("Original Histogram")
    axes[1].set_xlabel("Pixel Value")

    
    # (if processed is float 0–1 convert to 0–255)
    disp_img = processed_img
    if disp_img.dtype != np.uint8:
        disp_img = (disp_img * 255).astype("uint8")

    axes[2].imshow(disp_img)
    axes[2].set_title("Processed Image")
    axes[2].axis("off")

    # --- PROCESSED HIST ---
    axes[3].hist(processed_img.ravel(), bins=256, color='red')
    axes[3].set_title("Processed Histogram")
    axes[3].set_xlabel("Pixel Value")

    plt.tight_layout()
    plt.show()


resize_img = cv2.resize(img_original, (64, 64))
show_side_by_side("RESIZE (32 → 64)", resize_img)


norm_img = img_original.astype("float32") / 255.0
show_side_by_side("NORMALIZATION (0–1 Range)", norm_img)


denoise_img = cv2.GaussianBlur(img_original, (3, 3), 0)
show_side_by_side("DENOISING (Gaussian Blur)", denoise_img)



alpha = 1.5
contrast_img = np.clip(img_original.astype("float32") * alpha, 0, 255).astype('uint8')
show_side_by_side("CONTRAST ENHANCEMENT", contrast_img)


aug_img = cv2.flip(img_original, 1)
aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
show_side_by_side("AUGMENTATION (Flip + Rotate)", aug_img)
