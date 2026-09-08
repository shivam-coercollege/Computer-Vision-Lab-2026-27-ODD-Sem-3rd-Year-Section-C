import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Load images
image = cv2.imread("tiger.jpg")
image2 = cv2.imread("image2.jpg")

if image is None or image2 is None:
    print("Image file not found!")
    exit()

# Step 2: Convert images to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Basic preprocessing
gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)

# ---------------------------------------------------
# Step 3: SIFT Feature Extraction
# ---------------------------------------------------

sift = cv2.SIFT_create()

start_time = time.time()

keypoints, descriptors = sift.detectAndCompute(gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

sift_time = time.time() - start_time

print("SIFT keypoints in image 1:", len(keypoints))
print("SIFT keypoints in image 2:", len(keypoints2))

# Draw SIFT keypoints
sift_image = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# ---------------------------------------------------
# Steps 5-6: HOG Feature Extraction
# ---------------------------------------------------

# Resize image for HOG
hog_image = cv2.resize(gray, (128, 128))

# HOG parameters
win_size = (128, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
bins = 9

hog = cv2.HOGDescriptor(
    win_size,
    block_size,
    block_stride,
    cell_size,
    bins
)

start_time = time.time()

hog_features = hog.compute(hog_image)

hog_time = time.time() - start_time

print("HOG feature vector length:", len(hog_features))
print("HOG extraction time:", round(hog_time, 5), "seconds")

# ---------------------------------------------------
# HOG Visualization
# ---------------------------------------------------

def draw_hog(gray_image):
    image = cv2.resize(gray_image, (128, 128))
    image = np.float32(image) / 255.0

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    visual = np.zeros_like(image)

    cell_size = 8

    for y in range(0, 128, cell_size):
        for x in range(0, 128, cell_size):

            cell_mag = magnitude[y:y+cell_size, x:x+cell_size]
            cell_angle = angle[y:y+cell_size, x:x+cell_size]

            if cell_mag.size == 0:
                continue

            avg_mag = np.mean(cell_mag)
            avg_angle = np.mean(cell_angle)

            center_x = x + cell_size // 2
            center_y = y + cell_size // 2

            length = int(avg_mag * 20)

            angle_rad = np.deg2rad(avg_angle)

            x1 = int(center_x - length * np.cos(angle_rad))
            y1 = int(center_y - length * np.sin(angle_rad))

            x2 = int(center_x + length * np.cos(angle_rad))
            y2 = int(center_y + length * np.sin(angle_rad))

            cv2.line(
                visual,
                (x1, y1),
                (x2, y2),
                1,
                1
            )

    return visual


hog_visual = draw_hog(gray)

# ---------------------------------------------------
# Step 8: SIFT Image Matching
# ---------------------------------------------------

if descriptors is not None and descriptors2 is not None:

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        descriptors,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print("Good SIFT matches:", len(good_matches))

    match_image = cv2.drawMatches(
        image,
        keypoints,
        image2,
        keypoints2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite("sift_matches.jpg", match_image)

else:
    print("SIFT descriptors could not be calculated.")

# ---------------------------------------------------
# Save results
# ---------------------------------------------------

cv2.imwrite("sift_keypoints.jpg", sift_image)
cv2.imwrite(
    "hog_visualization.jpg",
    (hog_visual * 255).astype(np.uint8)
)

# ---------------------------------------------------
# Step 8: Display results
# ---------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(hog_visual, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Similar Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Steps 9-10: Observations
# ---------------------------------------------------

print("\nObservations:")
print("1. SIFT detects local keypoints from corners and textured regions.")
print("2. SIFT descriptors are useful for image matching.")
print("3. HOG describes the shape of an object using gradient directions.")
print("4. HOG works well for objects where shape and edges are important.")
print("5. SIFT provides better scale and rotation invariance.")
print("6. HOG is generally simpler and computationally efficient.")
print("7. SIFT is useful for feature matching and object recognition.")
print("8. HOG is commonly useful for object detection and shape analysis.")