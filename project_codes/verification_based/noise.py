import os
import cv2
import numpy as np

# Input folder containing original probe images
input_folder = "probe_images/cleanImages"

# Output folders
noise1_folder = "probe_images/noise_level1"
noise2_folder = "probe_images/noise_level2"

os.makedirs(noise1_folder, exist_ok=True)
os.makedirs(noise2_folder, exist_ok=True)


def add_gaussian_noise(image, sigma):
    """
    sigma controls noise intensity
    larger sigma = more degradation
    """

    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)

    noisy_image = image.astype(np.float32) + noise

    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)

    img = cv2.imread(image_path)

    if img is None:
        continue

    # Noise Level 1 (mild)
    noisy1 = add_gaussian_noise(img, sigma=15)

    # Noise Level 2 (strong)
    noisy2 = add_gaussian_noise(img, sigma=30)

    cv2.imwrite(os.path.join(noise1_folder, filename), noisy1)

    cv2.imwrite(os.path.join(noise2_folder, filename), noisy2)

    print(f"Processed {filename}")

print("\nNoise generation completed.")
