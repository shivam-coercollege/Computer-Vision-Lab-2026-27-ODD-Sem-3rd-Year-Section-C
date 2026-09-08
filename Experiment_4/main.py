import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image in grayscale
image = cv2.imread("Dogs_love.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Step 2: Compute DFT
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Step 3: Shift zero frequency to the center
dft_shift = np.fft.fftshift(dft)

# Step 4: Calculate magnitude spectrum
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)

# Image dimensions
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create distance matrix
y, x = np.ogrid[:rows, :cols]
distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

# Step 5: Low-Pass Filter
radius = 50

low_pass_mask = np.zeros((rows, cols), np.float32)
low_pass_mask[distance <= radius] = 1

low_pass = dft_shift * low_pass_mask[:, :, np.newaxis]

# Step 6: High-Pass Filter
high_pass_mask = np.ones((rows, cols), np.float32)
high_pass_mask[distance <= radius] = 0

high_pass = dft_shift * high_pass_mask[:, :, np.newaxis]

# Step 7: Inverse Fourier Transform - Low Pass
low_pass_shift = np.fft.ifftshift(low_pass)
low_pass_image = cv2.idft(low_pass_shift)
low_pass_image = cv2.magnitude(
    low_pass_image[:, :, 0],
    low_pass_image[:, :, 1]
)

# Normalize low-pass image
low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Inverse Fourier Transform - High Pass
high_pass_shift = np.fft.ifftshift(high_pass)
high_pass_image = cv2.idft(high_pass_shift)
high_pass_image = cv2.magnitude(
    high_pass_image[:, :, 0],
    high_pass_image[:, :, 1]
)

# Normalize high-pass image
high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Save output images
cv2.imwrite("original.jpg", image)
cv2.imwrite("frequency_spectrum.jpg", magnitude_spectrum.astype(np.uint8))
cv2.imwrite("low_pass.jpg", low_pass_image)
cv2.imwrite("high_pass.jpg", high_pass_image)

# Step 8: Compare all results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Frequency Spectrum")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass_image, cmap="gray")
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass_image, cmap="gray")
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 9: Display basic analysis
print("Frequency Domain Filtering completed.")
print("Low-pass filtering reduces high-frequency components and smooths the image.")
print("High-pass filtering removes low-frequency components and highlights edges and fine details.")

# Step 10: Observations
print("\nObservations:")
print("1. The original image contains both low and high frequency components.")
print("2. The frequency spectrum shows the distribution of image frequencies.")
print("3. Low-pass filtering produces a smoother image and reduces noise.")
print("4. High-pass filtering highlights edges and fine details.")
print("5. Frequency domain filtering is useful for image enhancement and noise reduction.")