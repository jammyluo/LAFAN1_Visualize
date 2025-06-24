import os
import argparse
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
import time
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import joblib

G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
   
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])


class MotionPlayer:
    def __init__(self, args):
        # init args
        self.args = args
        if self.args.robot_type == 'g1':
            urdf_path = "robot_description/g1/g1_29dof_rev_1_0.urdf"
            
            # 修复：使用正确的工作目录来加载 URDF
            current_dir = os.getcwd()
            urdf_full_path = os.path.join(current_dir, urdf_path)
            package_dir = os.path.join(current_dir, "robot_description/g1")
            
            # 确保路径存在
            if not os.path.exists(urdf_full_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_full_path}")
            if not os.path.exists(package_dir):
                raise FileNotFoundError(f"Package directory not found: {package_dir}")
                
            print(f"Loading URDF from: {urdf_full_path}")
            print(f"Package directory: {package_dir}")
            
            # 方法1：尝试使用绝对路径
            try:
                self.robot = pin.RobotWrapper.BuildFromURDF(urdf_full_path, package_dir, pin.JointModelFreeFlyer())
                print("✓ URDF 加载成功 (方法1)")
            except Exception as e:
                print(f"方法1失败: {e}")
                
                # 方法2：修改工作目录
                try:
                    print("尝试方法2：修改工作目录...")
                    original_cwd = os.getcwd()
                    os.chdir(package_dir)
                    
                    self.robot = pin.RobotWrapper.BuildFromURDF("g1_29dof_rev_1_0.urdf", ".", pin.JointModelFreeFlyer())
                    print("✓ URDF 加载成功 (方法2)")
                    
                    # 恢复工作目录
                    os.chdir(original_cwd)
                    
                except Exception as e2:
                    print(f"方法2失败: {e2}")
                    # 恢复工作目录
                    if 'original_cwd' in locals():
                        os.chdir(original_cwd)
                    raise Exception(f"无法加载 URDF: {e2}")
            
            self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                    -0.15,0,0,0.3,-0.15,0,
                                    -0.15,0,0,0.3,-0.15,0,
                                    0,0,0,
                                    0, 1.57,0,1.57,0,0,0,
                                    0,-1.57,0,1.57,0,0,0]).astype(np.float32)
        elif self.args.robot_type == 'h1_2':
            urdf_path = "robot_description/h1_2/h1_2_wo_hand.urdf"
        elif self.args.robot_type == 'h1':
            urdf_path = "robot_description/h1/h1.urdf"
        
        # inital gym
        self.gym = gymapi.acquire_gym()
        # create sim environment
        self.sim = self._create_simulation()
        # add plane
        self._add_ground_plane()
        # load urdf
        self.asset = self._load_urdf(urdf_path)
        # create and add robot
        self.env = self._create_env_with_robot()

    def _create_simulation(self):
        """create physics simulation environment"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 30.0
        sim_params.gravity = gymapi.Vec3(0.0, 0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        
        return self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def _add_ground_plane(self):
        """add plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up plane
        plane_params.distance = 0                   # the distance from plane to original
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _load_urdf(self, urdf_path):
        """load URDF"""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not existent: {urdf_path}")
            
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        
        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_env_with_robot(self):
        """create environment with robot"""
        env = self.gym.create_env(self.sim, 
                                 gymapi.Vec3(-2, 0, -2), 
                                 gymapi.Vec3(2, 2, 2), 
                                 1)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # put on the place with 1 meter high
        self.actor = self.gym.create_actor(env, self.asset, pose, "Robot", 0, 0)
        
        return env

    def set_camera(self, viewer):
        """ set the camera"""
        cam_pos = gymapi.Vec3(3, 2, 3)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        
    def run_viewer(self):
        """run visualize"""
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if not self.viewer:
            return
            
        self.set_camera(self.viewer)
        motion_data = self.load_data()
        
        root_state_tensor = torch.zeros((1, 13), dtype=torch.float32)
        dof_state_tensor = torch.zeros((29, 2), dtype=torch.float32)
        
        root_trans_all = []
        pose_aa_all = []
        dof_pos_all = []
        root_rot_all = []
        rot_vec_all = []
        
        max_motion_length = motion_data.shape[0]  # 使用实际的数据长度
        fps = 30.0
        
        # 计算开始和结束帧
        start_time = getattr(self.args, 'start_time', 0.0)  # 默认从0秒开始
        end_time = getattr(self.args, 'end_time', None)     # 默认到结束
        
        start_frame = int(start_time * fps)
        if end_time is not None:
            end_frame = int(end_time * fps)
        else:
            end_frame = max_motion_length
        
        # 确保帧数在有效范围内
        start_frame = max(0, min(start_frame, max_motion_length - 1))
        end_frame = max(start_frame + 1, min(end_frame, max_motion_length))
        
        print(f"=== 动画播放信息 ===")
        print(f"原始数据帧数: {max_motion_length}")
        print(f"指定开始时间: {start_time:.2f}s (帧 {start_frame})")
        print(f"指定结束时间: {end_time if end_time else '结束'}s (帧 {end_frame})")
        print(f"实际播放帧数: {end_frame - start_frame}")
        print(f"播放时间: {(end_frame - start_frame) / fps:.2f} 秒")
        print(f"==================")
        
        # main loop
        while not self.gym.query_viewer_has_closed(self.viewer):
            for frame_nr in range(start_frame, end_frame):
                start_time = time.time()
                configuration = torch.from_numpy(motion_data[frame_nr, :])
                
                # 打印当前帧信息（每100帧打印一次）
                if frame_nr % 100 == 0:
                    current_time = frame_nr / fps
                    print(f"当前帧: {frame_nr}/{end_frame} ({frame_nr/end_frame*100:.1f}%) - 时间: {current_time:.2f}s")
                
                # root_trans, root_rot, rot_vec = self.get_key_point(motion_data[frame_nr, :])
                
                root_trans_all.append(configuration[:3])
                root_rot_all.append(configuration[3:7])
                dof_pos_all.append(configuration[7:])
                
                rotation = R.from_quat(configuration[3:7])
                rotvec = rotation.as_rotvec()
                rotvec = torch.from_numpy(rotvec)
                
                rot_vec_all.append(rotvec)
                
                root_state_tensor[0, :7] = configuration[:7]
                dof_state_tensor[:,0] = configuration[7:]
                
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state_tensor))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
                
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / 30.0 - elapsed_time)
                time.sleep(sleep_time)

            root_trans_all = torch.cat(root_trans_all, dim=0).view(-1, 3).float()
            root_rot_all = torch.cat(root_rot_all, dim=0).view(-1, 4).float()
            dof_pos_all = torch.cat(dof_pos_all, dim=0).view(-1, 29).float()
            dof_pos_all = torch.cat((dof_pos_all[:, :19], dof_pos_all[:, 22:26]), dim=1).float()
            rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).float()
            N = rot_vec_all.shape[0]
            pose_aa = torch.cat([rot_vec_all[None, :, None], G1_ROTATION_AXIS * dof_pos_all[None,:,:,None], torch.zeros((1, N, 3, 3))], axis = 2)
            
            data_name = self.args.robot_type + '_' + self.args.file_name
            data_dump = {}
            
            data_dump[data_name]={
                "root_trans_offset": root_trans_all.cpu().detach().numpy(),
                "pose_aa": pose_aa.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_all.detach().cpu().numpy(), 
                "root_rot": root_rot_all.cpu().numpy(),
                "fps": 30
                }
            joblib.dump(data_dump, "pkl_data/" + self.args.robot_type + "/" + self.args.file_name + ".pkl")
            print("retargte data save succefully!")
            break
            
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def load_data(self):
        file_name = self.args.file_name
        robot_type = self.args.robot_type
        csv_files = robot_type + '/' + file_name + '.csv'
        
        # 检查文件是否存在
        if not os.path.exists(csv_files):
            raise FileNotFoundError(f"CSV file not found: {csv_files}")
        
        data = np.genfromtxt(csv_files, delimiter=',')
        
        # 打印文件信息
        print(f"=== CSV 文件信息 ===")
        print(f"文件路径: {csv_files}")
        print(f"数据形状: {data.shape}")
        print(f"总帧数: {data.shape[0]}")
        
        # 计算时间长度（假设 30 FPS）
        fps = 30.0
        total_time = data.shape[0] / fps
        print(f"时间长度: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"帧率: {fps} FPS")
        print(f"==================")
        
        return data
    
    # key point visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
        sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.env, sphere_pose)

    def draw_line(self, start_point, end_point, color, env_id):
        gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.env)
    
    # get key point by piconicon
    def get_key_point(self, configuration = None):
        self.clear_lines()
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        
        _rot_vec = []
        _root_trans = []
        _root_rot = []
        
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            
            ref_body_pos = joint_tf.translation 
            ref_body_rot = joint_tf.rotation
            
            rotation = R.from_matrix(ref_body_rot)
            rot_vec = rotation.as_rotvec()
        
            if frame_name == 'pelvis':
                _rot_vec = rot_vec
                _root_trans = ref_body_pos
                _root_rot = ref_body_rot

            color_inner = (0.0, 0.0, 0.545)
            color_inner = tuple(color_inner)

            # import ipdb; ipdb.set_trace()
            self.draw_sphere(ref_body_pos, 0.04, color_inner, 0)
            
        return np.array(_root_trans), np.array(_root_rot), np.array(_rot_vec)

    def get_robot_state(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.num_envs = 1
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
    def save_data(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='c')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    parser.add_argument('--start_time', type=float, default=0.0, 
                       help="开始时间（秒），默认从0秒开始")
    parser.add_argument('--end_time', type=float, default=None, 
                       help="结束时间（秒），默认到文件结束")
    args = parser.parse_args()
    
    print(f"=== 转换参数 ===")
    print(f"文件名称: {args.file_name}")
    print(f"机器人类型: {args.robot_type}")
    print(f"开始时间: {args.start_time}s")
    print(f"结束时间: {args.end_time if args.end_time else '文件结束'}s")
    print(f"================")
    
    loader = MotionPlayer(args)
    loader.run_viewer()