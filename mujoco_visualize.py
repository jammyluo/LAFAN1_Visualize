import os
import argparse
import numpy as np
import mujoco
import mujoco_viewer
import time

class MotionPlayer:
    def __init__(self, args):
        self.args = args
        # 优先使用用户指定的xml_path
        if self.args.xml_path:
            xml_path = self.args.xml_path
        else:
            if self.args.robot_type == 'g1':
                xml_path = "robot_description/g1/g1_29dof_anneal_23dof.xml"
            elif self.args.robot_type == 'h1_2':
                xml_path = "robot_description/h1_2/h1_2_wo_hand.xml"
            elif self.args.robot_type == 'h1':
                xml_path = "robot_description/h1/h1.xml"
            else:
                raise ValueError(f"未知的robot_type: {self.args.robot_type}")
        
        # 直接加载xml
        self.model = self._load_model(xml_path)
        self.data = mujoco.MjData(self.model)
        self._set_initial_state()

    def _load_model(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML model not existent: {xml_path}")
        try:
            return mujoco.MjModel.from_xml_path(xml_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _set_initial_state(self):
        """设置初始状态，与Isaac Gym保持一致"""
        # 设置初始位置（1米高，与Isaac Gym一致）
        # if self.data.qpos.size > 2:
        #     self.data.qpos[2] = 1.0  # Z轴位置
        
        # 设置重力（与Isaac Gym一致：Z轴向下）
        self.model.opt.gravity[2] = -9.81
        
        # 应用初始状态
        mujoco.mj_forward(self.model, self.data)

    def set_camera(self, viewer):
        """设置相机位置，与Isaac Gym保持一致"""
        # Isaac Gym: cam_pos = gymapi.Vec3(3, 2, 3), cam_target = gymapi.Vec3(0, 0, 1)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -45.0
        viewer.cam.lookat = np.array([0.0, -0.25, 0.824])

    def run_viewer(self):
        try:
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            if not viewer:
                print("Failed to create viewer")
                return
            
            self.set_camera(viewer)
            motion_data = self.load_data()
            print(f"Loaded motion data with {motion_data.shape[0]} frames")
            print(f"Robot has {self.model.nq} position DOFs and {self.model.nv} velocity DOFs")
            print("Press 'q' to quit the viewer")
            
            # 主循环，与Isaac Gym保持一致的结构
            while viewer.is_alive:
                for frame_nr in range(motion_data.shape[0]):
                    start_time = time.time()
                    configuration = motion_data[frame_nr, :]
                    
                    # 只在第一帧显示调试信息
                    if frame_nr == 0:
                        self.debug_joint_mapping(configuration)
                    
                    # 设置根状态（前7个值：位置+旋转四元数）
                    if len(configuration) >= 7:
                        # 设置完整根状态 (位置 + 四元数)
                        self.data.qpos[0:3] = configuration[0:3]
                        # self.data.qpos[3] = configuration[6] * -1.0
                        # self.data.qpos[4] = configuration[3]
                        # self.data.qpos[5] = configuration[4]
                        # self.data.qpos[6] = configuration[5]  
                        
                    
                    # 设置关节角度（剩余值）
                    if len(configuration) > 7 and self.model.nq > 7:
                        # 正确映射23个关节角度
                        # 前19个关节: configuration[7:26] -> qpos[7:26]
                        self.data.qpos[7:26] = configuration[7:26]
                        # 后4个关节: configuration[26:30] -> qpos[29:29+4]
                        self.data.qpos[26:30] = configuration[29:29+4]

                    
                    # 应用状态更新
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 渲染
                    viewer.render()
                    
                    # 控制帧率（30 FPS，与Isaac Gym一致）
                    elapsed_time = time.time() - start_time
                    sleep_time = max(0, 1.0 / 30.0 - elapsed_time)
                    time.sleep(sleep_time)
            
            viewer.close()
        except Exception as e:
            print(f"Error in viewer: {e}")
            raise

    def load_data(self):
        file_name = self.args.file_name
        robot_type = self.args.robot_type
        csv_files = robot_type + '/' + file_name + '.csv'
        if not os.path.exists(csv_files):
            raise FileNotFoundError(f"Motion data file not found: {csv_files}")
        try:
            data = np.genfromtxt(csv_files, delimiter=',')
            print(f"Loaded motion data: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading motion data: {e}")
            raise

    def print_robot_info(self):
        print(f"Robot model: {self.model.nq} position DOFs, {self.model.nv} velocity DOFs")
        print(f"Number of bodies: {self.model.nbody}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of actuators: {self.model.nu}")
        
        # 打印关节信息
        print(f"\n=== 关节信息 ===")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            joint_qposadr = self.model.jnt_qposadr[i]
            joint_dofadr = self.model.jnt_dofadr[i]
            print(f"Joint {i}: {joint_name}, type={joint_type}, qpos_adr={joint_qposadr}, dof_adr={joint_dofadr}")

    def debug_joint_mapping(self, configuration):
        """调试关节映射"""
        print(f"\n=== 关节映射调试 ===")
        print(f"动作数据维度: {len(configuration)}")
        print(f"模型qpos维度: {self.model.nq}")
        print(f"根状态 (前7维): {configuration[:7]}")
        print(f"关节角度 (7-30): {configuration[7:30]}")
        print(f"关节角度 (30-36): {configuration[30:36] if len(configuration) > 30 else 'N/A'}")
        
        # 检查四元数是否为单位四元数
        root_quat = configuration[3:7]
        quat_norm = np.linalg.norm(root_quat)
        print(f"根四元数模长: {quat_norm:.6f}")
        if abs(quat_norm - 1.0) > 1e-6:
            print(f"⚠️  警告: 四元数不是单位四元数")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject1')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    parser.add_argument('--xml_path', type=str, help="MuJoCo XML模型路径（可选）", default=None)
    parser.add_argument('--info', action='store_true', help="Print robot information")
    args = parser.parse_args()
    try:
        loader = MotionPlayer(args)
        if args.info:
            loader.print_robot_info()
        else:
            loader.run_viewer()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 