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


def initialize_feature_detector(
    detector_type: ENUM_FEATURE_DETECTOR_TYPE,
) -> cv2.Feature2D:
    if detector_type == ENUM_FEATURE_DETECTOR_TYPE.SIFT:
        return cv2.SIFT_create()
    elif detector_type == ENUM_FEATURE_DETECTOR_TYPE.ORB:
        return cv2.ORB_create()
    raise ValueError("Unsupported detector type. Use 'SIFT' or 'ORB'.")


def load_image(img_path: str):
    """Load an image in grayscale mode."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return image


def detect_keypoints_from_image(img, feature_detector: cv2.Feature2D):
    """Detect keypoints and descriptors using the specified feature detector."""
    return feature_detector.detectAndCompute(img, None)


def get_img_keypoints(img_path: str, feature_detector: cv2.Feature2D):
    """Detect keypoints in an image and save the output."""
    # Load image
    img = load_image(img_path)

    # Detect keypoints and descriptors
    keypoints, _ = detect_keypoints_from_image(img, feature_detector)

    # Draw keypoints on the image
    img_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 255, 0),  # Fixed color: Green
    )

    return img_keypoints


def compare_img_keypoints(
    img_path1: str, img_path2: str, feature_detector: cv2.Feature2D
):
    """Compare keypoints between two images and show the output."""
    img_keypoints1 = get_img_keypoints(img_path1, feature_detector)
    img_keypoints2 = get_img_keypoints(img_path2, feature_detector)

    # Show the output
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_keypoints1)
    ax[0].set_title(f"{img_path1} Keypoints with {feature_detector.__class__}")
    ax[0].axis("off")
    ax[1].imshow(img_keypoints2)
    ax[1].set_title(f"{img_path2} Keypoints with {feature_detector.__class__}")
    ax[1].axis("off")
    plt.show()


# Configuration
DIR = "datasets"
CONFIG = [
    ("cow1.jpg", "cow2.jpg"),
    ("match1.jpg", "match2.jpg"),
    ("graf_img1.jpg", "graf_img5.jpg"),
]

SIFT_DETECTOR = initialize_feature_detector(ENUM_FEATURE_DETECTOR_TYPE.SIFT)


def main():
    for img1, img2 in CONFIG:
        compare_img_keypoints(
            os.path.join(DIR, img1),
            os.path.join(DIR, img2),
            SIFT_DETECTOR,
        )


if __name__ == "__main__":
    main()
