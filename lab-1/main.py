# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:43:46 2019
@author: Metin Mert Akçay

Modified on Thu Sep 26 2024
@author: fredle09

VNU-UIT_CS231.P11
21520850 - Lê Trung Hiếu
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import skew, kurtosis
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import cv2
import os


BIN_SIZE = 16
TEST_PATH = "test"
TRAIN_PATH = "train"


def read_image(image_path: str) -> np.ndarray:
    """This function is used to read images by LUV.

    :param image_path: path of the image
    :return image: image
    """
    image: np.ndarray = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
    # Check if image is read properly
    if image is None:
        raise ValueError(
            f"Image at path {image_path} could not be loaded. Please check the path or file."
        )

    image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image


# Chuẩn hóa histogram feature
def normalized_color_histogram(image: np.ndarray) -> list[np.ndarray]:
    """This function is used to create histogram. After creation of histogram, histogram is
    divided by total number of pixels and normalizing each histogram value between 0 and 1.

    :param image: image
    :return feature: normalized histogram values for each channel
    """
    row, column, channel = image.shape[:3]
    size: int = row * column

    feature: list[np.ndarray] = []
    for k in range(channel):
        histogram: list[np.ndarray] = np.squeeze(
            cv2.calcHist([image], [k], None, [BIN_SIZE], [0, 256])
        )
        histogram = histogram / size
        feature.extend(histogram)
    return feature


# Sử dụng thuộc tính moment
def moment(channel: np.ndarray) -> list[float]:
    """This function is used for find color moments.

    :param channel: channel (L, a, b)
    :return feature: color moment results of the examined channel
    """
    feature = []
    feature.append(np.mean(channel))
    std = np.std(channel)
    feature.append(std)
    if std == 0:
        feature.extend([np.float64(0), np.float64(0)])
    else:
        feature.extend([skew(channel), kurtosis(channel)])
    return feature


# Hàm chuẩn hóa min, max
def normalize_vector_min_max(feature):
    """This function is used to normalize the vector using Min-Max normalization.

    :param feature: vector to be normalized
    :return: normalized vector
    """
    min_val = np.min(feature)  # Find the minimum value in the vector
    max_val = np.max(feature)  # Find the maximum value in the vector

    if max_val == min_val:  # Avoid division by zero
        return feature  # If all values are the same, return the original vector

    # Apply Min-Max normalization
    return (feature - min_val) / (max_val - min_val)


# Hàm chuẩn hóa L2
def normalize_vector_norm(feature: list[float]) -> list[float]:
    """This function is used to normalize the vector using L2 normalization.

    :param feature: vector to be normalized
    :return: normalized vector
    """
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature
    return (feature / norm).tolist()


# Chuẩn hóa color-moment feature
def color_moment(image: np.ndarray) -> list[float]:
    """This function is used to create color moment features.

    :param image: image
    :return feature: calculated color moment values for each channel
    """
    if image is None or image.size == 0:
        raise ValueError("The image is empty or not valid.")

    channel = image.shape[2]

    # Efficiently extract channel data using numpy
    channel_list = [image[:, :, k].flatten() for k in range(channel)]

    feature = []
    for channel_data in channel_list:
        feature.extend(moment(channel_data))

    return feature


# Tính toán và chuẩn hóa cdc feature
def calculate_cdc(image: np.ndarray) -> list[float]:
    """This function is used to calculate the Color Difference Coherence (CDC) feature.

    :param image: image
    :return: CDC feature
    """
    cdc_feature: list[float] = []
    for i in range(image.shape[2] - 1):
        cdc_feature.append(np.mean(np.abs(image[:, :, i] - image[:, :, i + 1])))

    return cdc_feature


# Tính toán và chuẩn hóa ccv feature
def calculate_ccv(image: np.ndarray) -> list[float]:
    """This function is used to calculate the Color Coherence Vector (CCV) feature.

    :param image: image
    :return: CCV feature
    """
    _, coherence_map = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ccv_feature = coherence_map.flatten().tolist()

    return ccv_feature


# Trích xuất các đặc trưng từ ảnh
def extract_features(image, feature: str):
    """This function is used to extract features from the given image.

    :param image: image
    :return: extracted features
    """
    if feature == "histogram":
        return normalized_color_histogram(image)
    elif feature == "moment":
        return normalize_vector_min_max(color_moment(image))
    elif feature == "cdc":
        return normalize_vector_norm(calculate_cdc(image))
    elif feature == "ccv":
        return normalize_vector_norm(calculate_ccv(image))

    raise ValueError("Missing feature value in extract_feature")


# Định nghĩa hàm Chi-square
def chi_square_distance(x, y):
    """This function is used to calculate the Chi-square distance between two vectors.

    :param x: first vector
    :param y: second vector
    :return: Chi-square distance between x and y
    """
    return 0.5 * np.sum(((x - y) ** 2) / (x + y + 1e-10))


# Định nghĩa hàm Intersection
def intersection_distance(x, y):
    """This function is used to calculate the Intersection distance between two vectors.

    :param x: first vector
    :param y: second vector
    """
    return np.sum(np.minimum(x, y))


# Định nghĩa hàm Bhattacharyya
def bhattacharyya_distance(p, q):
    """This function is used to calculate the Bhattacharyya distance between two vectors.

    :param p: first vector
    :param q: second vector
    :return: Bhattacharyya distance between p and q
    """
    return -np.log(np.sum(np.sqrt(p * q)))


# KNeighborsClassifierExtends
class KNeighborsClassifierExtends(KNeighborsClassifier):
    """Extends KNeighborsClassifier to allow chi-square, intersection,
    and bhattacharyya distance functions.
    """

    def __init__(self, metric, **kwargs):
        if metric == "chi-square":
            super().__init__(metric=chi_square_distance, **kwargs)
        elif metric == "intersection":
            super().__init__(metric=intersection_distance, **kwargs)
        elif metric == "bhattacharyya":
            super().__init__(metric=bhattacharyya_distance, **kwargs)
        else:
            super().__init__(metric=metric, **kwargs)


# Đếm số lượng ảnh trong thư mục
def count_data(path: str) -> int:
    """Count number of images in the given path.

    :param path: Path to the directory containing images
    :return: Number of images in the directory
    """
    number_of_image_count = 0
    color_list = os.listdir(path)
    for color_name in color_list:
        path_ = os.path.join(path, color_name)
        image_list = os.listdir(os.path.join(path_))
        number_of_image_count += len(image_list)

    return number_of_image_count


# Load dữ liệu từ thư mục
def load_data(path: str, number_of_image_count: int, feature: str):
    """Load data from the given path.

    :param path: Path to the directory containing images
    :param number_of_image_count: Number of images in the directory
    :return: Data and labels
    """
    data = []
    label = []
    color_list = os.listdir(path)
    with tqdm(total=number_of_image_count) as pbar:
        for index, color_name in enumerate(color_list):
            path_ = os.path.join(path, color_name)
            image_list = os.listdir(os.path.join(path_))
            for image_name in image_list:
                image = read_image(os.path.join(path_, image_name))
                image_features = extract_features(image, feature)
                data.append(image_features)
                label.append(index)
                pbar.update(1)
    return data, label


# Train and test KNN
def train_and_test_knn(
    k: int, train_data: np.ndarray, train_label: np.ndarray, metric: str, feature: str
) -> float:
    """Train and test KNN model using the given hyperparameters.

    :param k: Number of neighbors
    :param metric: Distance metric
    :return: Accuracy of the model"""
    model = KNeighborsClassifierExtends(n_neighbors=k, metric=metric)
    model.fit(train_data, train_label)

    print(f"<---------- TEST START: K = {k}, Metric = {metric} ---------->")
    test_data, test_label = load_data(TEST_PATH, number_of_test_image_count, feature)

    prediction = model.predict(test_data)

    accuracy: float = metrics.accuracy_score(test_label, prediction)
    # print()
    # print(f">> Accuracy K = {k}, metric = {metric}:", accuracy)
    # print()
    print(">> Label Order:", color_list)
    print(confusion_matrix(test_label, prediction))
    print()
    print("<---------------------->")
    print()

    return accuracy


FEATURES = ["histogram", "moment", "cdc", "ccv"]
K_VALUES = [1, 5]
METRICS = ["euclidean", "correlation", "chi-square", "intersection", "bhattacharyya"]


if __name__ == "__main__":
    # find number of train images
    number_of_train_image_count = count_data(TRAIN_PATH)

    # find number of test images
    number_of_test_image_count = count_data(TEST_PATH)

    color_list = os.listdir(TRAIN_PATH)

    result_metric = pd.DataFrame(columns=["Feature", "K", "Metric", "Accuracy"])

    for feature in FEATURES:
        print(f"<---------- TRAIN START: Feature = {feature} ---------->")
        train_data, train_label = load_data(
            TRAIN_PATH, number_of_train_image_count, feature
        )

        for k in K_VALUES:
            for metric in METRICS:
                accuracy = train_and_test_knn(
                    k, train_data, train_label, metric, feature
                )
                new_record = pd.DataFrame(
                    [
                        {
                            "Feature": feature,
                            "K": k,
                            "Metric": metric,
                            "Accuracy": accuracy,
                        }
                    ]
                )
                result_metric = pd.concat(
                    [result_metric, new_record], ignore_index=True
                )

    print(
        result_metric.pivot(index=["K", "Feature"], columns="Metric", values="Accuracy")
    )
