import os
import os.path as osp
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from glob import glob
import argparse


import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


def load_o3d_cam_pose(cam_pose_file: str) -> tuple:
    """ load open3D camera pose from depth_rendering json file

    Args:
        cam_pose_file (str): file path of open3D camera pose json file

    Returns:
        Tuple: (extrinsic matrix, intrinsic matrix)
    """
    with open(cam_pose_file, 'r') as ifs:
        trajectory = json.load(ifs)
        assert (trajectory['class_name'] == "PinholeCameraParameters")

        T_w_c = np.eye(4)
        intrinsic_matrix = np.eye(3)

        extrinsics = trajectory['extrinsic']
        T_w_c[:, 0] = extrinsics[:4]
        T_w_c[:, 1] = extrinsics[4:8]
        T_w_c[:, 2] = extrinsics[8:12]
        T_w_c[:, 3] = extrinsics[12:]
        # print(T_w_c)

        intrinsics = trajectory['intrinsic']
        img_width = intrinsics['width']
        img_height = intrinsics['height']
        intrin_mat = intrinsics['intrinsic_matrix']
        intrinsic_matrix[:, 0] = intrin_mat[:3]
        intrinsic_matrix[:, 1] = intrin_mat[3:6]
        intrinsic_matrix[:, 2] = intrin_mat[6:]
        # print(intrinsic_matrix)
        return T_w_c, intrinsic_matrix


def interp_and_convert_o3d_render_json(v_cam_pose_files: List[str], save_o3d_camera_traj_filepath: str) -> Dict:
    """ interpolate camera poses and save to open3D camera trajectory json file

    Args:
        v_cam_pose_files (List[str]): list of camera pose file paths
        save_o3d_camera_traj_filepath (str): output open3D camera trajectory json file path

    Returns:
        Dict: caemra trajectory json dict
    """
    v_rotation_in = np.zeros([0, 4])
    v_pos_x_in = []
    v_pos_y_in = []
    v_pos_z_in = []
    for i, cam_pose_file in enumerate(v_cam_pose_files):
        T_w_c, intrinsic = load_o3d_cam_pose(cam_pose_file)
        v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(T_w_c[:3, :3]).as_quat()], axis=0)
        v_pos_x_in.append(T_w_c[0, 3])
        v_pos_y_in.append(T_w_c[1, 3])
        v_pos_z_in.append(T_w_c[2, 3])

    in_times = np.arange(0, len(v_rotation_in)).tolist()
    out_times = np.linspace(0, len(v_rotation_in) - 1, len(v_rotation_in) * 12).tolist()
    print(f'in_times: {(in_times)}')
    print(f'out_times: {(out_times)}')
    v_rotation_in = Rotation.from_quat(v_rotation_in)
    slerp = Slerp(in_times, v_rotation_in)
    v_interp_rotation = slerp(out_times)

    fx = interp1d(in_times, np.array(v_pos_x_in), kind='quadratic')
    fy = interp1d(in_times, np.array(v_pos_y_in), kind='quadratic')
    fz = interp1d(in_times, np.array(v_pos_z_in), kind='quadratic')
    v_interp_xs = fx(out_times)
    v_interp_ys = fy(out_times)
    v_interp_zs = fz(out_times)

    root_node = {}
    root_node["class_name"] = "PinholeCameraTrajectory"

    intrinsic_node = {}
    intrinsic_node['width'] = 512
    intrinsic_node['height'] = 512
    cam_intrinsic = np.array([[150.0, 0., 256], [0., 150.0, 256], [0., 0., 1.]], dtype=np.float32)
    # cam_intrinsic = intrinsic
    cam_intrinsic_params = []
    cam_intrinsic_params += cam_intrinsic[:, 0].tolist()
    cam_intrinsic_params += cam_intrinsic[:, 1].tolist()
    cam_intrinsic_params += cam_intrinsic[:, 2].tolist()
    intrinsic_node['intrinsic_matrix'] = cam_intrinsic_params

    v_cam_nodes = []
    for idx in range(len(out_times)):
        cam_node = {}
        cam_node['class_name'] = "PinholeCameraParameters"
        cam_node['version_major'] = 1
        cam_node['version_minor'] = 0

        rot_matrix = v_interp_rotation[idx].as_matrix()
        trans = np.array([v_interp_xs[idx], v_interp_ys[idx], v_interp_zs[idx]])
        cam_ext = np.eye(4)
        cam_ext[:3, :3] = rot_matrix
        cam_ext[:3, 3] = trans
        cam_ext_params = []
        cam_ext_params += cam_ext[:, 0].tolist()
        cam_ext_params += cam_ext[:, 1].tolist()
        cam_ext_params += cam_ext[:, 2].tolist()
        cam_ext_params += cam_ext[:, 3].tolist()
        cam_node['extrinsic'] = cam_ext_params
        cam_node['intrinsic'] = intrinsic_node
        v_cam_nodes.append(cam_node)

    root_node['parameters'] = v_cam_nodes
    root_node["version_major"] = 1
    root_node["version_minor"] = 0

    with open(save_o3d_camera_traj_filepath, 'w') as fc:
        json.dump(root_node, fc)

    return root_node


def custom_draw_geometry_with_camera_trajectory(pcd:o3d.geometry.TriangleMesh, 
                                                render_option_path:str, 
                                                camera_trajectory_path:str,
                                                render_geometry:bool = False,
                                                render_output_path: str = './render_output'):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.frame_idx = 0
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    if render_geometry:
        image_path = os.path.join(render_output_path, 'geometry')
    else:
        image_path = os.path.join(render_output_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(render_output_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            # if glb.index % 3 == 0:
            print("Capture image {:05d}".format(glb.frame_idx))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(glb.frame_idx)), np.asarray(depth), dpi=1)
            plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.frame_idx)), np.asarray(image), dpi=1)
            glb.frame_idx += 1

        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index], allow_arbitrary=True)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=512, height=512)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    if render_geometry:
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
    vis.get_render_option().mesh_show_back_face = False
    # if render_geometry:
    #     vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.FlatShade
    vis.register_animation_callback(move_forward)
    vis.run()
    return vis
    


def merge_image_sequences_to_video(img_seq1_folder: str, img_seq2_folder: str, output_video_filepath: str):
    """ merge two image sequences to a video, must install ffmpeg first

    Args:
        img_seq1_folder (str): the first image sequence folder
        img_seq2_folder (str): the second image sequence folder
        output_video_filepath (str): output video file path
    """

    if not osp.isdir(img_seq1_folder):
        print(f'folder {img_seq1_folder} doesnt exist!')
        exit(-1)

    if img_seq2_folder is not None and not osp.isdir(img_seq2_folder):
        print(f'folder {img_seq2_folder} doesnt exist!')
        exit(-1)

    v_rgb_img_file = [img_f for img_f in os.listdir(img_seq1_folder) if img_f.endswith('.png')]
    v_rgb_img_file.sort(key=lambda x: int(x.split('.')[0]))
    # print(v_rgb_img_file)

    if img_seq2_folder is not None:
        v_sem_img_file = [img_f for img_f in os.listdir(img_seq2_folder) if img_f.endswith('.png')]
        v_sem_img_file.sort(key=lambda x: int(x.split('.')[0]))

    if img_seq2_folder is not None:
        assert len(v_rgb_img_file) == len(v_sem_img_file) and (len(v_rgb_img_file) > 0)
    # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    video_h = 512
    video_w = 512 * 2 if img_seq2_folder is not None else 512
    merge_width = video_w // 2
    size = (video_w, video_h)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # videoWrite = cv2.VideoWriter(output_video_filepath, 0x00000021, 12.0, size)
    videoWrite = cv2.VideoWriter(output_video_filepath, fourcc, 10.0, size)

    img_num = len(v_rgb_img_file)
    for img_idx in range(img_num):
        rgb_img_filepath = osp.join(img_seq1_folder, v_rgb_img_file[img_idx])
        rgb_img = cv2.imread(rgb_img_filepath)  # 读取第一张图片

        if img_seq2_folder is not None:
            sem_img_filepath = osp.join(img_seq2_folder, v_sem_img_file[img_idx])
            sem_img = cv2.imread(sem_img_filepath)  # 读取第一张图片
        img_height, img_width, _ = rgb_img.shape
        # print(f'rgb_img.shape: {rgb_img.shape}')

        merge_img = np.ones((img_height, video_w, 3), dtype=np.uint8) * 255
        if img_seq2_folder is not None:
            # merge_img = np.concatenate((rgb_img[:, 0:merge_width], sem_img[:, merge_width:]), axis=1)
            merge_img[:, 0:merge_width, :] = rgb_img
            merge_img[:, merge_width:, :] = sem_img
        else:
            merge_img = rgb_img
        # print(f'merge_img.shape: {merge_img.shape}')
        # if img_idx == 0:
        # cv2.imwrite(f'/Users/fc/Desktop/videoclip/merge_img_{img_idx}.png', merge_img)

        videoWrite.write(merge_img)  # 将图片写入所创建的视频对象
        if img_idx == len(v_rgb_img_file)-1:
            for i in range(14):
                videoWrite.write(merge_img)

    videoWrite.release()

    print('Start convert MP4 to H264...')
    os.system(f'/Users/fc/Downloads/ffmpeg -i {output_video_filepath} -vcodec libx264 -y {output_video_filepath[:-4]}_h264.mp4')
    print('end!')

def recover_pcl_from_depth(depth_img_filepath:str, rgb_img_filepath:str):
    rgb_img = cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED)
    dep_img = cv2.imread(depth_img_filepath, cv2.IMREAD_UNCHANGED)
    H,W,C = rgb_img.shape

    depth_img = dep_img.astype(np.float32)[:,:,None]
    # depth_img = 1./depth_img

    rgb = o3d.geometry.Image(rgb_img.astype(np.uint8))
    depth = o3d.geometry.Image(depth_img.astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth); 
    
    K = np.array([[250.0, 0., W/2], [0., 250.0, H/2], [0., 0., 1.]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,K[0,0],K[1,1],K[0,2],K[1,2])
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H)
    pcl = []
    pcl = o3d.geometry.PointCloud()
    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcl


def test_perspective_img_depth():


    rgb_img_filepath = '/Users/fc/Desktop/WechatIMG642.jpeg'
    depth_img_filepath = '/Users/fc/Desktop/WechatIMG642_depth_midas.png'
    rgb_img = np.asarray(cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED))
    depth_img = np.asarray(cv2.imread(depth_img_filepath, cv2.IMREAD_GRAYSCALE))
    H, W, C = rgb_img.shape

    # Get camera intrinsic
    hfov = 60. * np.pi / 180.
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])

    depth_img = 1./depth_img.astype(np.float32)
    depth_img = np.expand_dims((depth_img).astype(np.float32),axis=2)
    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
    depth = depth_img.reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0

    # chose front as reference view
    T_ref_subview = np.eye(4)

    subview_pointcloud = T_ref_subview @ xy_c0
    subview_pointcloud_T = np.transpose(subview_pointcloud)
    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(subview_pointcloud_T[:,:3])
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1,3))
    # o3d.visualization.draw_geometries([o3d_pointcloud])
    o3d_pointcloud = o3d_pointcloud.voxel_down_sample(voxel_size=0.00005)
    o3d.io.write_point_cloud('/Users/fc/Desktop/WechatIMG642_downsample.ply', o3d_pointcloud)


def visulize_textured_mesh(textured_mesh_folder:str = '/Users/fc/Desktop/papers/2024-ECCV-ctrlroom/2024eccv_experiments/ours_kitchen/',
                           scene_name_lst:List[str] = ['study_scene_03253_534182']):
    """ vislize textured mesh in the folder

    Args:
        textured_mesh_folder (str, optional): _description_. Defaults to '/Users/fc/Desktop/papers/2024-ECCV-ctrlroom/2024eccv_experiments/ours_kitchen/'.
        scene_name_lst (List[str], optional): specific scene to be visualized. Defaults to ['kitchen_scene_03406_159'].
    """
    
    # textured_mesh_folder = '/Users/fc/Desktop/papers/2024-ECCV-ctrlroom/2024eccv_experiments/ours_kitchen/'
    sub_folder_lst = [f for f in os.listdir(textured_mesh_folder) if osp.isdir(osp.join(textured_mesh_folder, f))]

    for i, sub_folder in enumerate(sub_folder_lst):
        if len(scene_name_lst):
            if sub_folder not in scene_name_lst:
                continue

        sub_folder_path = os.path.join(textured_mesh_folder, sub_folder, 'select_textured_mesh')
        mesh_path = os.path.join(sub_folder_path, 'model.obj')
        # mesh_path = os.path.join(sub_folder_path, 'model_table_move/cut_ceiling_model.obj')
        # mesh_path = os.path.join(sub_folder_path, 'cut_ceiling.ply')
        # mesh_path = os.path.join(sub_folder_path, 'model_sofa_resize', 'cut_ceiling_model.obj')
        # mesh_path = os.path.join(sub_folder_path, 'fused_final_poisson_meshlab_depth_12_quadric_10000000.ply')

        if not osp.exists(mesh_path):
            print(f'{mesh_path} doesnt exist!')
            continue

        mesh = o3d.io.read_triangle_mesh(mesh_path, True)
        # mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folderpath', 
                      type=str, 
                      help='input folder path of generated scenes',
                      default='/Users/fc/Desktop/papers/2024-3DV-CtrlRoom/bedrooms/')
    parser.add_argument('--scene_name',
                        type=list,
                        help='specific scenes to render, if not specified, render all scenes in the input folderpath',
                        default=['scene_03422_794742'])
    parser.add_argument('--vis_mesh',
                        action='store_true',
                        help='render geometry or not',)
    
    args = parser.parse_args()
    
    input_folderpath =args.input_folderpath
    scene_name_lst = args.scene_name
    input_subfolder_lst = [f for f in os.listdir(input_folderpath) if osp.isdir(osp.join(input_folderpath, f))]

    vis_mesh = args.vis_mesh
    
    # visulize textured mesh
    if vis_mesh:
        visulize_textured_mesh(textured_mesh_folder=input_folderpath, scene_name_lst=scene_name_lst)
    else:
        for i, sub_folder in enumerate(input_subfolder_lst):
            # if the scene name is specified, only render the scene with the name
            if len(scene_name_lst):
                if sub_folder not in scene_name_lst:
                    continue

            scene_folderpath = osp.join(input_folderpath, sub_folder, 'select_textured_mesh')

            mesh_filepath = osp.join(scene_folderpath, 'model.obj')
            save_o3d_camera_traj_filepath = osp.join(scene_folderpath, 'camera_trajectory.json')
            render_output_folderpath = osp.join(scene_folderpath, 'render_output')
            if not osp.exists(render_output_folderpath):
                os.makedirs(render_output_folderpath)

            render_option_path = 'render.json'
            v_cam_pose_files = glob(osp.join(scene_folderpath, 'DepthCamera*.json'))
            v_cam_pose_files.sort(key=lambda x: osp.basename(x))
            print(v_cam_pose_files)

            # interpolated_camera_trajectory  and save to json
            interp_and_convert_o3d_render_json(v_cam_pose_files, save_o3d_camera_traj_filepath)

            render_modes = ['texture', 'geometry']
            for render_mode in render_modes:
                print("6. Customized visualization playing a camera trajectory")
                
                pcd_flipped = o3d.io.read_triangle_mesh(mesh_filepath, True)
                
                if 'geometry' == render_mode:
                    pcd_flipped.compute_vertex_normals()
                o3d_vis = custom_draw_geometry_with_camera_trajectory(pcd_flipped, 
                                                            render_option_path, 
                                                            save_o3d_camera_traj_filepath,
                                                            render_geometry='geometry' == render_mode,
                                                            render_output_path=render_output_folderpath)

                if 'geometry' == render_mode:
                    video_path = osp.join(scene_folderpath, 'mesh_video.mp4')
                    img_seq_folder = osp.join(render_output_folderpath, 'geometry')
                else:
                    video_path = osp.join(scene_folderpath, 'rgb_video.mp4')
                    img_seq_folder = osp.join(render_output_folderpath, 'image')
                # save video
                merge_image_sequences_to_video(img_seq1_folder=img_seq_folder, 
                                                img_seq2_folder=None, 
                                                output_video_filepath=video_path)
                o3d_vis.destroy_window()


            # merge image sequences of ours, other approaches into a common video
            tex_render_img_folderpath = f'{scene_folderpath}/render_output/image'
            geo_render_img_folderpath = f'{scene_folderpath}/render_output/geometry'
            video_path = f'{scene_folderpath}/render_output/rgb_geometry.mp4'
            if osp.exists(tex_render_img_folderpath) and osp.exists(geo_render_img_folderpath):
                merge_image_sequences_to_video(img_seq1_folder=tex_render_img_folderpath, 
                                            img_seq2_folder=geo_render_img_folderpath, 
                                            output_video_filepath=video_path)


