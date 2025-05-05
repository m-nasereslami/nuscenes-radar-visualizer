import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import random

# ------------------------------
# Transform Radar to Ego Frame
# ------------------------------
def transform_to_ego(points, cs_record, ep_record, ref_pose):
    # Sensor to ego
    cs_rot = Quaternion(cs_record['rotation']).rotation_matrix
    cs_trans = np.array(cs_record['translation']).reshape((3, 1))
    points[:3, :] = np.dot(cs_rot, points[:3, :]) + cs_trans
    points[8:10, :] = np.dot(cs_rot[:2, :2], points[8:10, :])

    # Ego to global
    ep_rot = Quaternion(ep_record['rotation']).rotation_matrix
    ep_trans = np.array(ep_record['translation']).reshape((3, 1))
    points[:3, :] = np.dot(ep_rot, points[:3, :]) + ep_trans

    # Global to reference ego
    ref_rot = Quaternion(ref_pose['rotation']).inverse.rotation_matrix
    ref_trans = -np.array(ref_pose['translation']).reshape((3, 1))
    points[:3, :] = np.dot(ref_rot, points[:3, :] + ref_trans)
    return points


# Plot Radar in Ego Frame
def plot_radar_visualization(all_points_dict):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    for sensor, (points, color) in all_points_dict.items():
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        vx, vy = points[:, 3], points[:, 4]
        ax1.scatter(x, y, z, s=5, c=color, label=sensor, alpha=0.7)
        ax2.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, width=0.002, color=color, label=sensor)

    # Ego vehicle marker at (0,0,0)
    ax1.scatter(0, 0, 0, c='black', marker='x', s=60, label='Ego Vehicle')

    ax1.set_title('3D Radar Point Cloud (Ego Frame)')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_box_aspect([1, 1, 0.2])
    ax1.legend()

    ax2.set_title('2D Radar Velocities (vx, vy)')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Project Radar to Camera Views
def project_to_cameras(sample, all_points_dict, nusc, data_path):
    camera_sensors = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for i, cam in enumerate(camera_sensors):
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
        cam_rot = Quaternion(cam_calib['rotation']).rotation_matrix
        cam_trans = np.array(cam_calib['translation']).reshape((3, 1))

        img = imread(os.path.join(data_path, cam_data['filename']))
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(cam)
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        # Project each sensor's points
        for sensor, (points, color) in all_points_dict.items():
            radar_points = points[:, :3].T  # shape (3, N)
            radar_cam = np.dot(cam_rot.T, radar_points - cam_trans)
            in_front = radar_cam[2, :] > 0
            radar_cam = radar_cam[:, in_front]

            radar_2d = view_points(radar_cam, cam_intrinsic, normalize=True)
            ax.scatter(radar_2d[0], radar_2d[1], s=10, c=color, label=sensor)

        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.show()



def main(nuscenes_path, version):
    nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
    sample = nusc.sample[0]
    # sample = random.choice(nusc.sample)

    radar_sensors = [
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
    ]

    sensor_colors = {
        "RADAR_FRONT": 'blue',
        "RADAR_FRONT_LEFT": 'green',
        "RADAR_FRONT_RIGHT": 'cyan',
        "RADAR_BACK_LEFT": 'orange',
        "RADAR_BACK_RIGHT": 'red',
    }

    all_points_dict = {}
    all_points = []

    ref_ep = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['RADAR_FRONT'])['ego_pose_token'])

    for sensor in radar_sensors:
        sd_token = sample['data'][sensor]
        sd_record = nusc.get('sample_data', sd_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        radar_path = os.path.join(nuscenes_path, sd_record['filename'])
        radar_pc = RadarPointCloud.from_file(radar_path)

        transformed = transform_to_ego(radar_pc.points.copy(), cs_record, ep_record, ref_ep)

        x, y, z = transformed[0], transformed[1], transformed[2]
        vx, vy = transformed[8], transformed[9]
        ego_points = np.stack([x, y, z, vx, vy], axis=-1)

        all_points_dict[sensor] = (ego_points, sensor_colors[sensor])
        all_points.append(ego_points)

    plot_radar_visualization(all_points_dict)

    project_to_cameras(sample, all_points_dict, nusc, nuscenes_path)

    print("\nPoints per sensor:")
    for sensor in radar_sensors:
        path = os.path.join(nuscenes_path, nusc.get('sample_data', sample['data'][sensor])['filename'])
        num = RadarPointCloud.from_file(path).points.shape[1]
        print(f"{sensor}: {num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "<Path to nuScenes dataset>", help="Path to nuScenes data")
    parser.add_argument('--version', type=str, default="v1.0-mini", help="nuScenes dataset version")
    args = parser.parse_args()

    main(args.data_path, args.version)
