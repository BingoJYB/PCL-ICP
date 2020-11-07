import numpy as np
from icp import icp
from matplotlib import pyplot as plt


def get_rectangle(left_top_vertex, right_bottom_vertex):
    left, top = left_top_vertex
    right, bottom = right_bottom_vertex

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
    homo_rect = np.ones((row, col + 1))
    homo_rect[:, :-1] = rectangle

    return homo_rect


def dehomo(homo_rect):

    return homo_rect[:, :-1]


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


def translate(rectangle, delta_x=0, delta_y=0):
    T = np.array([[1, 0, delta_x],
                  [0, 1, delta_y],
                  [0, 0, 1]])

    return np.dot(rectangle, T.T)


def main():
    left_top_vertex = (1, 21)
    right_bottom_vertex = (21, 11)
    rotation_angle = 45
    delta_x, delta_y = 30, 15

    rectangle = get_rectangle(left_top_vertex, right_bottom_vertex)
    homo_rect = get_homo_coord(rectangle)
    center = get_center(rectangle)
    transformed_homo_rect = homo_rect.copy()
    transformed_homo_rect = rotate(
        transformed_homo_rect, rotation_angle, center)
    transformed_homo_rect = translate(transformed_homo_rect, delta_x, delta_y)
    transformed_rect = dehomo(transformed_homo_rect)

    transformation_history, aligned_points = icp(
        rectangle, transformed_rect, verbose=True)

    x = rectangle[:, 0]
    y = rectangle[:, 1]
    xx = transformed_rect[:, 0]
    yy = transformed_rect[:, 1]
    xxx = aligned_points[:, 0]
    yyy = aligned_points[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=15, c='r', marker="o", label='source')
    ax.scatter(xx, yy, s=15, c='b', marker="o", label='target')
    ax.scatter(xxx, yyy, s=15, c='g', marker="x", label='aligned')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
