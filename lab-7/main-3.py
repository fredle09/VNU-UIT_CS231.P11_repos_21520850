import cv2
import matplotlib.pyplot as plt
import os
from enum import Enum


class ENUM_FEATURE_DETECTOR_TYPE(Enum):
    ORB = "ORB"
    FAST_BRIEF = "FAST+BRIEF"


def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return image


DETECTORS = {
    ENUM_FEATURE_DETECTOR_TYPE.ORB: [cv2.ORB_create()],
    ENUM_FEATURE_DETECTOR_TYPE.FAST_BRIEF: [
        cv2.FastFeatureDetector_create(),
        cv2.xfeatures2d.BriefDescriptorExtractor_create(),
    ],
}


def detect_keypoints_and_descriptors(img, detector_type: ENUM_FEATURE_DETECTOR_TYPE):
    detectors = DETECTORS[detector_type]
    keypoints = detectors[0].detect(img, None)
    keypoints, descriptors = detectors[-1].compute(img, keypoints)
    return keypoints, descriptors


def brute_force_matching(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # For BRIEF descriptors
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def filter_good_matches(matches, ratio=0.7):
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


def flann_matching(descriptors1, descriptors2):
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = filter_good_matches(matches)
    return good_matches


# Function to draw matches
def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


DIR = "datasets"
CONFIG = {
    "images": ["left.jpg", "right.jpg"],
    "detectors": [
        ENUM_FEATURE_DETECTOR_TYPE.FAST_BRIEF,
        ENUM_FEATURE_DETECTOR_TYPE.ORB,
    ],
    "matchers": [brute_force_matching, flann_matching],
}


def main():
    images = [load_image(os.path.join(DIR, img_path)) for img_path in CONFIG["images"]]

    # kp1_orb, des1_orb = detect_keypoints_and_descriptors(
    #     img1, ENUM_FEATURE_DETECTOR_TYPE.ORB
    # )
    # kp2_orb, des2_orb = detect_keypoints_and_descriptors(
    #     img2, ENUM_FEATURE_DETECTOR_TYPE.ORB
    # )
    i = 1
    for detector in CONFIG["detectors"]:
        for matcher in CONFIG["matchers"]:
            params = [
                detect_keypoints_and_descriptors(image, detector) for image in images
            ]
            matches = matcher(params[0][1], params[1][1])
            img_matches = draw_matches(
                images[0], params[0][0], images[1], params[1][0], matches
            )
            plt.subplot(2, 2, i)
            plt.imshow(img_matches)
            plt.title(f"{detector.value} with {matcher.__name__}")
            plt.axis("off")
            i += 1

    plt.tight_layout()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
