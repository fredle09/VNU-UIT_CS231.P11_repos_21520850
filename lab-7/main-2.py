# -*- coding: utf-8 -*-
"""
VNU-UIT_CS231.P11
21520850 - Lê Trung Hiếu
"""

import os
import cv2
from enum import Enum
import matplotlib.pyplot as plt


class ENUM_FEATURE_DETECTOR_TYPE(Enum):
    SIFT = "SIFT"
    ORB = "ORB"
    FAST_BRIEF = "FAST+BRIEF"


DETECTORS = {
    ENUM_FEATURE_DETECTOR_TYPE.SIFT: [cv2.SIFT_create()],
    ENUM_FEATURE_DETECTOR_TYPE.ORB: [cv2.ORB_create()],
    ENUM_FEATURE_DETECTOR_TYPE.FAST_BRIEF: [
        cv2.FastFeatureDetector_create(),
        cv2.xfeatures2d.BriefDescriptorExtractor_create(),
    ],
}


def load_image(img_path: str):
    """Load an image in grayscale mode."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return image


def detect_keypoints_descriptor_from_image(
    img, detector_type: ENUM_FEATURE_DETECTOR_TYPE
):
    detectors = DETECTORS[detector_type]
    keypoints = detectors[0].detect(img, None)
    keypoints, descriptors = detectors[-1].compute(img, keypoints)
    return keypoints, descriptors


def get_img_keypoints(img_path: str, detector_type: ENUM_FEATURE_DETECTOR_TYPE):
    """Detect keypoints in an image and save the output."""
    # Load image
    img = load_image(img_path)

    # Detect keypoints and descriptors
    keypoints, _ = detect_keypoints_descriptor_from_image(img, detector_type)

    # Draw keypoints on the image
    img_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 255, 0),  # Fixed color: Green
    )

    return img_keypoints


def display_images(images: list, titles: list, cols: int = 1, rows: int = 1):
    """Display a list of images with corresponding titles."""
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Configuration
DIR = "datasets"
CONFIG = {
    "images": ["butterfly.jpg", "home.jpg", "simple.jpg"],
    "detectors": [
        ENUM_FEATURE_DETECTOR_TYPE.FAST_BRIEF,
        ENUM_FEATURE_DETECTOR_TYPE.ORB,
    ],
}


def main():
    images, titles = [], []
    for detector_type in CONFIG["detectors"]:
        images.extend(
            [
                get_img_keypoints(
                    os.path.join(DIR, image_path),
                    detector_type,
                )
                for image_path in CONFIG["images"]
            ]
        )
        titles.extend(
            [
                f"{image_path} with {detector_type.value}"
                for image_path in CONFIG["images"]
            ]
        )
    display_images(images, titles, 3, 2)


if __name__ == "__main__":
    main()
