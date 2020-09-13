import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from depth_evaluation_utils import *
import cv2 as cv


def get_rbgd(rgb_fpath="data/drive.png", d_fpath="data/drive_depth_norm.png"):
    color_raw = o3d.io.read_image(rgb_fpath)
    depth_raw = o3d.io.read_image(d_fpath)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)
    return rgbd_image


def plot_image_and_depth(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()


def downsample_point_cloud(pcd):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.scale(1000, center=pcd.get_center())
    return pcd


def get_volume_bound_box():
    bound_box = pcd.get_oriented_bounding_box()
    bound_box.color = (1, 0, 0)
    o3d.visualization.draw_geometries([pcd, bound_box])
    print(bound_box.extent)
    return bound_box


def pcd_to_mesh(pcd):
    # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    downpcd = pcd.sample_points_poisson_disk(3000)
    print('original: {}, downsampled: {}'.format(pcd, downpcd))

    pcd = downpcd
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    print('average distance: {}'.format(avg_dist))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3*radius, max_nn=30))
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    radius *= 2

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    o3d.visualization.draw_geometries([pcd, bpa_mesh], point_show_normal=True)
    # dec_mesh = mesh.simplify_quadric_decimation(100000)


def scale_depth_range(depth):
    depth = np.interp(depth, (depth.min(), depth.max()),
                      (0, 255)).astype(np.uint8)
    return depth


def plot_depth_distribution(depth):
    depth_1d = depth.flatten()
    plt.hist(depth_1d, bins=100)
    plt.gca().set(title='depth distribution')
    plt.show()


def normalize_depth_distribution(depth):
    im_shape = depth.shape
    depth_norm = cv.equalizeHist(depth.flatten())
    plot_depth_distribution(depth_norm)
    return np.reshape(scale_depth_range(depth_norm), im_shape)


def plot_lidar_pcd(kitti_dir):
    gt_file, gt_calib, im_size, im_file, camera_id = read_file_data(
        kitti_base_dir=kitti_dir)
    vel_points = generate_depth_map(gt_calib, gt_file,
                                    im_size,
                                    camera_id,
                                    ret_vel_sparse=True, vel_depth=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vel_points)

    visualizer = o3d.visualization.Visualizer()
    coor_frame_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame()
    visualizer.create_window(width=1000, height=1000, left=0, top=250)
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(coor_frame_geometry)

    cam = view_ctl.convert_to_pinhole_camera_parameters()
    pose = np.eye(4)
    R = np.eye(3)
    T = [0, 0, 0]
    pose[:3, :3] = R
    pose[:3, 3] = T
    cam.extrinsic = pose
    # load changed extrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam)
    visualizer.run()
    visualizer.destroy_window()


def main():
    kitti_dir = '/Users/akshitjain/ext/workspace/datasets/kitti_2012/2011_09_26/2011_09_26_drive_0035_sync'
    camera_id = 2
    gt_file, gt_calib, im_size, im_file, _ = read_file_data(
        kitti_base_dir=kitti_dir, cam=camera_id)
    depth_sparse = generate_depth_map(gt_calib,
                                      gt_file,
                                      im_size,
                                      camera_id,
                                      interp=False,
                                      vel_depth=True)
    # plot_depth_distribution(depth_dense)
    # gt_depth = scale_depth_range(depth_dense)
    # gt_depth = normalize_depth_distribution(gt_depth)
    gt_depth = depth_sparse
    plt.imshow(depth_sparse)
    plt.show()


if __name__ == "__main__":
    main()
