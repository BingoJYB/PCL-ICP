import pcl
import numpy as np
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


def convert_2d_to_3d(rectangle):
    rect = rectangle.copy()
    rect[:, -1] = 0

    return rect.astype(np.float32)


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


def run_icp(points_in, points_out, max_iter=10):
    cloud_in = pcl.PointCloud()
    cloud_out = pcl.PointCloud()

    cloud_in.from_array(points_in)
    cloud_out.from_array(points_out)

    icp = cloud_in.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(
        cloud_in, cloud_out, max_iter=1000)

    return converged, transf, estimate, fitness


def main():
    left_top_vertex = (1, 21)
    right_bottom_vertex = (21, 11)
    rotation_angle = 60
    delta_x, delta_y = 30, 20
    max_iter = 1000

    rectangle = get_rectangle(left_top_vertex, right_bottom_vertex)
    homo_rect = get_homo_coord(rectangle)
    center = get_center(rectangle)
    transformed_homo_rect = homo_rect.copy()
    transformed_homo_rect = rotate(
        transformed_homo_rect, rotation_angle, center)
    transformed_homo_rect = translate(transformed_homo_rect, delta_x, delta_y)
    rectangle_3d = convert_2d_to_3d(homo_rect)
    transformed_rect_3d = convert_2d_to_3d(transformed_homo_rect)

    converged, transf, estimate, fitness = run_icp(
        rectangle_3d, transformed_rect_3d, max_iter)

    rectangle_homo_3d = get_homo_coord(rectangle_3d)
    aligned_rect = np.dot(rectangle_homo_3d, transf.T)

    x = rectangle[:, 0]
    y = rectangle[:, 1]
    xx = transformed_rect_3d[:, 0]
    yy = transformed_rect_3d[:, 1]
    xxx = aligned_rect[:, 0]
    yyy = aligned_rect[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=15, c='r', marker="o", label='source')
    ax.scatter(xx, yy, s=15, c='b', marker="o", label='target')
    ax.scatter(xxx, yyy, s=15, c='g', marker="x", label='aligned')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
