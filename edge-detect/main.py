import numpy as np
from scipy.ndimage import convolve

# Define Sobel kernels for x and y direction
K_X = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
K_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def sobel_edge_detection(image, direction: str):
    """
    Perform Sobel edge detection on the given image in the specified direction.
    Skips edge pixels and shows calculation steps with absolute values.
    """
    # Select the appropriate kernel based on the direction
    if direction == "x":
        K = K_X
    elif direction == "y":
        K = K_Y
    else:
        raise ValueError("Invalid direction. Please use 'x' or 'y'.")

    # Initialize the gradient array
    gradient = np.zeros_like(image, dtype=float)

    # Get the dimensions of the image
    rows, cols = image.shape

    # Iterate over inner pixels only (skip edges)
    print("\nCalculating gradient values for inner pixels:")
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # Show the calculation step by step
            print(f"\nCalculating value at [{row}, {col}]:")

            value = (
                image[row - 1, col - 1] * K[0, 0]
                + image[row - 1, col] * K[0, 1]
                + image[row - 1, col + 1] * K[0, 2]
                + image[row, col - 1] * K[1, 0]
                + image[row, col] * K[1, 1]
                + image[row, col + 1] * K[1, 2]
                + image[row + 1, col - 1] * K[2, 0]
                + image[row + 1, col] * K[2, 1]
                + image[row + 1, col + 1] * K[2, 2]
            )

            print(
                f"{image[row-1, col-1]} * {K[0,0]} + "
                f"{image[row-1, col]} * {K[0,1]} + "
                f"{image[row-1, col+1]} * {K[0,2]} + "
            )
            print(
                f"{image[row, col-1]} * {K[1,0]} + "
                f"{image[row, col]} * {K[1,1]} + "
                f"{image[row, col+1]} * {K[1,2]} + "
            )
            print(
                f"{image[row+1, col-1]} * {K[2,0]} + "
                f"{image[row+1, col]} * {K[2,1]} + "
                f"{image[row+1, col+1]} * {K[2,2]} = {value}"
            )
            print(f"Absolute value = |{value}| = {abs(value)}")

            gradient[row, col] = abs(value)

    print("\nNote: Edge pixels are skipped (set to 0)")
    print("\nGradient image:\n", gradient)

    # Calculate the threshold T for edge detection
    T = 0.5 * np.max(np.abs(gradient))
    print(f"\nThreshold T = 0.5 * max(gradient) = {T}")

    # Create a binary edge map based on the threshold
    binary_edge_map = (np.abs(gradient) > T).astype(np.uint8)
    print("\nBinary Edge Map:\n", binary_edge_map)

    return gradient, T, binary_edge_map


def main():
    # Define a sample grayscale image
    image = np.array(
        [
            [210, 220, 10, 10],
            [210, 180, 30, 20],
            [220, 250, 60, 50],
            [206, 50, 20, 90],
            [250, 30, 20, 60],
        ]
    )

    print("Original image:\n", image)

    # Perform Sobel edge detection on the sample image in the x direction
    gradient, threshold, edge_map = sobel_edge_detection(image, "x")


if __name__ == "__main__":
    main()
