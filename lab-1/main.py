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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.stats import skew, kurtosis
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import sys
import cv2
import os

from typing import List

BIN_SIZE = 16
TEST_PATH = 'test'
TRAIN_PATH = 'train'


def read_image(image_path: str) -> np.ndarray:
    """ 
    This function is used to read images by LUV.
    :param image_path: path of the image
    :return image: image
    """
    image: np.ndarray = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
    # Check if image is read properly
    if image is None:
        raise ValueError(
            f"Image at path {image_path} could not be loaded. Please check the path or file.")

    image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image

# Chuẩn hóa histogram feature


def normalized_color_histogram(image: np.ndarray) -> List[np.ndarray]:
    """
    This function is used to create histogram. After creation of histogram, histogram is 
        divided by total number of pixels and normalizing each histogram value between 0 and 1.
    :param image: image
    :return feature: normalized histogram values for each channel
    """
    row, column, channel = image.shape[:3]
    size: int = row * column

    feature: List[np.ndarray] = []
    for k in range(channel):
        histogram: List[np.ndarray] = np.squeeze(
            cv2.calcHist([image], [k], None, [BIN_SIZE], [0, 256]))
        histogram = histogram / size
        feature.extend(histogram)
    return feature


def moment(channel: np.ndarray) -> List[float]:
    """
    This function is used for find color moments.
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


def normalize_vector_min_max(feature):
    """
    This function is used to normalize the vector using Min-Max normalization.
    :param feature: vector to be normalized
    :return: normalized vector
    """
    min_val = np.min(feature)  # Find the minimum value in the vector
    max_val = np.max(feature)  # Find the maximum value in the vector

    if max_val == min_val:  # Avoid division by zero
        return feature  # If all values are the same, return the original vector

    # Apply Min-Max normalization
    return (feature - min_val) / (max_val - min_val)


def normalize_vector_norm(feature: List[float]) -> List[float]:
    """
    This function is used to normalize the vector using L2 normalization.
    :param feature: vector to be normalized
    :return: normalized vector
    """
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature
    return (feature / norm).tolist()


# Chuẩn hóa color-moment feature
def color_moment(image: np.ndarray) -> List[float]:
    """
    This function is used to create color moment features.
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
        normalize_vector = normalize_vector_min_max(moment(channel_data))
        feature.extend(normalize_vector)
        # feature.extend(moment(channel_data))

    return feature


# Định nghĩa hàm Chi-square
def chi_square_distance(x, y):
    """
    This function is used to calculate the Chi-square distance between two vectors.
    :param x: first vector
    :param y: second vector
    :return: Chi-square distance between x and y
    """
    return 0.5 * np.sum(((x - y) ** 2) / (x + y + 1e-10))


# Định nghĩa hàm Intersection
def intersection_distance(x, y):
    """
    This function is used to calculate the Intersection distance between two vectors.
    :param x: first vector
    :param y: second vector
    """
    return np.sum(np.minimum(x, y))


# Định nghĩa hàm Bhattacharyya
def bhattacharyya_distance(p, q):
    """
    This function is used to calculate the Bhattacharyya distance between two vectors.
    :param p: first vector
    :param q: second vector
    :return: Bhattacharyya distance between p and q
    """
    return -np.log(np.sum(np.sqrt(p * q)))


# Tính toán và chuẩn hóa cdc feature
def calculate_cdc(image: np.ndarray) -> List[float]:
    """
    This function is used to calculate the Color Difference Coherence (CDC) feature.
    :param image: image
    :return: CDC feature
    """
    cdc_feature: List[float] = []
    for i in range(image.shape[2] - 1):
        cdc_feature.append(
            np.mean(np.abs(image[:, :, i] - image[:, :, i + 1])))

    # print(">> cdc_feature:", cdc_feature)
    # return cdc_feature
    normalize_vector = normalize_vector_norm(cdc_feature)
    return normalize_vector


# Tính toán và chuẩn hóa ccv feature
def calculate_ccv(image: np.ndarray) -> List[float]:
    """
    This function is used to calculate the Color Coherence Vector (CCV) feature.
    :param image: image
    :return: CCV feature
    """
    _, coherence_map = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ccv_feature = coherence_map.flatten().tolist()

    # return ccv_feature
    # normalize_vector = normalize_vector_min_max(ccv_feature)
    # return normalize_vector
    normalize_vector = normalize_vector_norm(ccv_feature)
    return normalize_vector


# Trích xuất các đặc trưng từ ảnh
def extract_features(image):
    histogram_features = normalized_color_histogram(image)
    moment_features = color_moment(image)
    cdc_features = calculate_cdc(image)
    ccv_features = calculate_ccv(image)

    # print(">> histogram_features:", histogram_features)
    # print(">> moment_features:", moment_features)
    # print(">> cdc_features:", cdc_features)
    # print(">> ccv_features:", ccv_features)
    features = histogram_features + moment_features + cdc_features + ccv_features
    # print(">> Features:", features.shape)
    return features


# KNeighborsClassifierExtends
class KNeighborsClassifierExtends(KNeighborsClassifier):
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
    """
    Count number of images in the given path.
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
def load_data(path: str, number_of_image_count: int):
    """
    Load data from the given path.
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
                image_features = extract_features(image)
                data.append(image_features)
                label.append(index)
                pbar.update(1)
    return data, label


# Train and test KNN
def train_and_test_knn(k: int, metric: str) -> float:
    model = KNeighborsClassifierExtends(n_neighbors=k, metric=metric)
    model.fit(train_data, train_label)

    print(f'<----------TEST START K = {k}, metric = {metric} ---------->')
    test_data, test_label = load_data(TEST_PATH, number_of_test_image_count)

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


if __name__ == '__main__':
    # find number of train images
    number_of_train_image_count = count_data(TRAIN_PATH)

    # find number of test images
    number_of_test_image_count = count_data(TEST_PATH)

    color_list = os.listdir(TRAIN_PATH)
    print('<----------TRAIN START ---------->')
    train_data, train_label = load_data(
        TRAIN_PATH, number_of_train_image_count)

    result_metric = pd.DataFrame(columns=['K', 'Metric', 'Accuracy'])
    for k in range(1, 6):
        for metric in ["euclidean", "correlation", "chi-square",
                        "intersection",
                       "bhattacharyya"]:
            accuracy = train_and_test_knn(k, metric)
            new_record = pd.DataFrame(
                [{'K': k, 'Metric': metric, 'Accuracy': accuracy}])
            result_metric = pd.concat(
                [result_metric, new_record], ignore_index=True)

    print(result_metric.pivot(index='K', columns='Metric', values='Accuracy'))
