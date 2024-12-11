# -*- coding: utf-8 -*-
"""
VNU-UIT_CS231.P11
21520850 - Lê Trung Hiếu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import skfuzzy as fuzz
from concurrent.futures import ThreadPoolExecutor  # For parallel processing

DIR = "datasets"


def load_image(image_path):
    img = cv2.imread(DIR + "/" + image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
    return img


def represent_rgb(img):
    return img.reshape((-1, 3))


def represent_rgb_xy(img):
    height, width = img.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.dstack((img, x, y))  # Stack r, g, b, x, y in one array
    return pixels.reshape((-1, 5))


def mean_shift_clustering(img, bandwidth=30):
    data = represent_rgb(img)
    mean_shift = MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(data)
    clustered_img = (
        mean_shift.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)
    )
    return clustered_img


def fcm_clustering(img, n_clusters=6):
    data = represent_rgb_xy(img)

    cntr, u, _, _, _, _, _ = fuzz.cmeans(
        data.T, n_clusters, 2, error=0.005, maxiter=100
    )

    labels = np.argmax(u, axis=0)

    clustered_img = np.zeros_like(img)
    for i, label in enumerate(labels):
        y, x = divmod(i, img.shape[1])
        clustered_img[y, x] = cntr[label, :3]
    return clustered_img


def visualize_images(original, clustered, title="Clustered Image"):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(clustered, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_images_grid(images, titles, rows, cols, wspace=0.5, hspace=0.5):
    """
    Display images in a grid with titles and customizable gaps between them.

    Parameters:
    - wspace: Horizontal space between images (default is 0.2).
    - hspace: Vertical space between images (default is 0.3).
    """
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    # Display images and set titles
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    # Hide any remaining axes if there are more axes than images
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust space between the subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.tight_layout()
    plt.show()


# Function to process a single image
def process_single_image(img_filename):
    img = load_image(img_filename)
    if img is None:
        print(f"Failed to load image {img_filename}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Processing image {img_filename}...")

    # Apply FCM clustering
    clusterd_fcm = fcm_clustering(img, n_clusters=6)
    print(f"Finished FCM clustering for {img_filename}")

    print(f"Processing image for {img_filename}")
    # Apply Mean Shift clustering
    clusterd_mean_shift = mean_shift_clustering(img, bandwidth=10)
    print(f"Finished Mean Shift clustering for {img_filename}")

    return img_rgb, clusterd_fcm, clusterd_mean_shift


# Main function to process images with optional parallelization
def process_images(parallel=False):
    image_paths = ["vegetables.jpg", "hand.jpg", "thuoc.jpg", "dogcat.jpg"]
    images = []
    titles = []

    if parallel:
        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_image, image_paths))
            for img_rgb, clusterd_fcm, clusterd_mean_shift, img_filename in results:
                if img_rgb is not None:
                    images.extend([img_rgb, clusterd_fcm, clusterd_mean_shift])
                    titles.extend(
                        [
                            f"Original - {img_filename}",
                            f"FCM - {img_filename}",
                            f"Mean Shift - {img_filename}",
                        ]
                    )
    else:
        # Process images sequentially
        for img_filename in image_paths:
            result = process_single_image(img_filename)
            if result:
                img_rgb, clusterd_fcm, clusterd_mean_shift = result
                images.extend([img_rgb, clusterd_fcm, clusterd_mean_shift])
                titles.extend(
                    [
                        f"Original - {img_filename}",
                        f"FCM - {img_filename}",
                        f"Mean Shift - {img_filename}",
                    ]
                )

    # Visualize the images in a grid
    visualize_images_grid(images, titles, rows=4, cols=3, wspace=0.5, hspace=0.5)


if __name__ == "__main__":
    process_images(
        parallel=True
    )  # Set parallel to False to disable parallel processing
