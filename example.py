import pcl
import numpy as np
from matplotlib import pyplot as plt


def get_rectangle(left, top, right, bottom):
    left_edge = np.stack(
        (np.array([left] * (top - bottom)), np.arange(bottom, top, 1)), axis=1)
    top_edge = np.stack(
        (np.arange(left, right + 1, 1), np.array([top] * (right - left + 1))), axis=1)
    right_edge = np.stack(
        (np.array([right] * (top - bottom)), np.arange(bottom, top, 1)), axis=1)
    bottom_edge = np.stack(
        (np.arange(left, right, 1), np.array([bottom] * (right - left))), axis=1)

    return np.concatenate((left_edge, top_edge, right_edge, bottom_edge), axis=0)


def get_homo_coord(rectangle):
    row, col = rectangle.shape
    padded = np.ones((row, col + 1))
    padded[:, :-1] = rectangle

    return padded


def get_center(rectangle):
    left, right = np.min(rectangle[:, 0]), np.max(rectangle[:, 0])
    bottom, top = np.min(rectangle[:, 1]), np.max(rectangle[:, 1])
    center = (
        left + (right - left) / 2.0,
        bottom + (top - bottom) / 2.0
    )

    return center


def rotate(rectangle, angle, pivot=(0, 0)):
    theta = (np.pi / 180.0) * angle
    a0, b0 = np.cos(theta), np.sin(theta)
    a1, b1 = pivot[0] - pivot[0] * a0 + pivot[1] * b0,\
        pivot[1] - pivot[0] * b0 - pivot[1] * a0

    R = np.array([[a0, -b0, a1],
                  [b0, a0, b1],
                  [0, 0, 1]])

    return np.dot(rectangle, R.T)


def translate(rectangle, trans_x=0, trans_y=0):
    T = np.array([[1, 0, trans_x],
                  [0, 1, trans_y],
                  [0, 0, 1]])

    return np.dot(rectangle, T.T)


def main():
    left, top, right, bottom = 1, 21, 21, 11
    rotation_angle = 60
    trans_x, trans_y = 30, 20

    rectangle = get_rectangle(left, top, right, bottom)
    homo_rect = get_homo_coord(rectangle)
    center = get_center(rectangle)
    transformed_rect = rotate(homo_rect, rotation_angle, center)
    transformed_rect = translate(transformed_rect, trans_x, trans_y)

    x = rectangle[:, 0]
    y = rectangle[:, 1]
    xx = transformed_rect[:, 0]
    yy = transformed_rect[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=15, c='r', marker="o", label='target')
    ax.scatter(xx, yy, s=15, c='b', marker="o", label='source')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
