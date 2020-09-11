import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

color_raw = o3d.io.read_image("data/drive.png")
depth_raw = o3d.io.read_image("data/drive_depth_norm.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)

# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()


pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd.scale(1000, center=pcd.get_center())


bound_box = pcd.get_oriented_bounding_box()
bound_box.color = (1, 0, 0)
# o3d.visualization.draw_geometries([pcd, bound_box])
print(bound_box.extent)

# downpcd = pcd.voxel_down_sample(voxel_size=0.01)
downpcd = pcd.sample_points_poisson_disk(3000)
print('downsampled')
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
