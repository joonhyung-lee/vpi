import os
import cv2
import mujoco_viewer
import numpy as np
import ray
import time
import sys
sys.path.append('./../')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from models.env.manipulator_agent import ManipulatorAgent
from models.utils.util import (rpy2r,r2rpy,r2quat, fill_object, get_geom_region_type1, get_geom_region_type2, get_geom_region_type3, 
                        passthrough_filter, remove_duplicates_with_threshold,sample_xyzs,downsample_pointcloud)

@ray.remote(max_restarts=-1)
class ManipulatorAgentRay(ManipulatorAgent):
    """
        MuJoCo Parser class
    """
    def __init__(self,name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=False, MODE='offscreen',env_id=0):
        """
            Initialize MuJoCo parser with ray
        """
        super().__init__(name=name,rel_xml_path=rel_xml_path,USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER,VERBOSE=VERBOSE, MODE=MODE)
        self.cnt = 0
        self.env_id = env_id
        self.DONE_FLAG = False
        # self.feasible_place_positions = []

    def get_mode(self):
        """
            Get mode
        """
        return self.MODE.copy()
    
    def get_idxs(self):
        """
            Get state of the robot
        """
        return [self.idxs_forward.copy(), self.idxs_jacobian.copy(), self.idxs_step.copy()]

    def check_done(self):
        """
            Check done flag
        """
        if self.DONE_FLAG:
            return True
        else:
            return False

    def get_done(self):
        """
            Get done flag
        """
        return self.DONE_FLAG
    
    def get_cnt(self):
        """
            Get cnt
        """
        return self.cnt.copy()
    
    def get_stability_verification_result(self, VERBOSE=True):
        """
            Get feasible place positions
        """
        if VERBOSE:
            print(f"Env ID: {self.env_id}, # of feasible place positions: {len(self.p_feasible)}")
        return self.p_feasible.copy(), self.R_feasible_range_list.copy()
        # return self.feasible_place_positions.copy(), self.R_feasible_range_list.copy()
    
    def set_viewer(self, azimuth, elevation, distance, lookat):
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation
        self.viewer.cam.distance = distance
        self.viewer.cam.lookat = lookat

    def reset_env(self, flag=False):
        """
            Reset done flag
        """
        self.DONE_FLAG = flag
        self.cnt = 0
        self.feasible_place_positions = []

    def init_set_state(self):
        """
            Initialize set environment configuration
        """
        # Move tables and robot base
        self.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        self.model.body('side_object_table').pos = np.array([-0.05,-0.80,0])
        self.model.body('ur_base').pos = np.array([0.18,0,0.79])
        self.model.body('ur_base').pos = np.array([0.18,0,0.8]) # robot base
        for body_name in ['base_table','front_object_table','side_object_table']:
            geomadr = self.model.body(body_name).geomadr[0]
            self.model.geom(geomadr).rgba[3] = 1.0

        # # Place objects
        # obj_box_names = [body_name for body_name in self.body_names
        #             if body_name is not None and (body_name.startswith("obj_box"))]
        # n_box_obj = len(obj_box_names)
        # self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=False)

        # Set target object's position
        self.model.joint(self.model.body('obj_target_01').jntadr[0]).qpos0[:3] = np.array([0.0, -0.5, 0.86])
        super().reset()

    def execute_stability_verification_ray(self, data_split, data_obj_xyzs, target_object_name='obj_target_01', quat_lower_bound=0.70, quat_upper_bound=0.79,
                                        end_tick=5000, noise_tick=2500, pos_offset=np.array([0, 0, 0.05]), noise_scale=0.01,nstep=100,inc_prefix=None, exc_prefix=None,
                                        init_pose=np.array([np.deg2rad(-90), np.deg2rad(-132.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)]), VERBOSE=True):
        # Set Objects from the dataset configuration.
        obj_names = [body_name for body_name in self.get_body_names()
                        if body_name is not None and (body_name.startswith("obj_"))]
        for obj_idx,obj_name in enumerate(obj_names):
            if obj_name == 'obj_target_01':
                continue
            jntadr = self.model.body(obj_name).jntadr[0]
            self.model.joint(jntadr).qpos0[:3] = data_obj_xyzs[obj_idx,:]
        self.reset()
        self.DONE = False
        # Feasible Position List
        positions = data_split + pos_offset.copy()
        R_feasible_range = np.zeros(2)
        # Reset Buffer
        self.p_feasible = []
        self.R_feasible_range_list = []

        # positions: [x, y, z]
        for position_idx, position in enumerate(positions):
            # Quaternion and position of the target object
            p_target_list = []
            R_target_list = []
            # Move the target object to the position
            jntadr = self.model.body(target_object_name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            self.data.qpos[qposadr:qposadr+3] = position
            # self.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))

            start = self.tick
            while (self.tick - start) < end_tick:
                if (self.tick - start) > noise_tick:
                    # Add arbitrary noise to the target object.
                    noise = np.random.normal(0, noise_scale, 4)
                    self.data.qpos[qposadr+3: qposadr+7] += noise

                # if not self.is_viewer_alive(): break
                self.forward(q=init_pose,joint_idxs=[0,1,2,3,4,5])
                self.step(ctrl=init_pose,ctrl_idxs=[0,1,2,3,4,5])
                R_target_list.append(r2quat(self.get_R_body(target_object_name)))

                p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(must_include_prefix=inc_prefix, must_exclude_prefix=exc_prefix)

                # Render
                if self.loop_every(HZ=50):
                    if self.MODE == 'window':
                        # Visualize Time and Placement position
                        self.plot_sphere(p=position, r=0.005, rgba=[0,1,0,1], label=f"{self.tick}/{end_tick}")
                        self.plot_sphere(p=position + np.array([0,0,0.1]), r=0.005, rgba=[0,1,0,0], label=f'Time: [{self.tick * 0.002:.4f}]')
                        self.plot_sphere(p=position + np.array([0,0,0.2]), r=0.005, rgba=[0,1,0,0], label=f'Quaternion: [{r2quat(self.get_R_body(target_object_name))[0]:.4f}]')
                        [self.plot_sphere(p=p___, r=0.005, rgba=[0,1,0,1]) for p___ in downsample_pointcloud(pointcloud=positions, grid_size=0.035)]
                        self.render(render_every=2500)

            # if not self.is_viewer_alive(): break
            self.reset()
            in_range = np.logical_and(np.array(R_target_list)[5:][:,0] >= quat_lower_bound, np.array(R_target_list)[5:][:,0] <= quat_upper_bound)
            all_in_range = np.all(in_range)
            if VERBOSE:
                print(f"all_in_range: {all_in_range}")
                print(f"max_qw: {np.max(np.array(R_target_list)[5:][:,0])}")
                print(f"min_qw: {np.min(np.array(R_target_list)[5:][:,0])}")

            # If the collide with other objects, it is not a feasible position.
            
            if all_in_range:
                if VERBOSE:
                    print(f"Feasible position: {position}")
                self.p_feasible.append(position)
                R_feasible_range = [np.max(np.array(R_target_list)[5:][:,0]) - np.min(np.array(R_target_list)[5:][:,0])]
                self.R_feasible_range_list.append(R_feasible_range)

            if position_idx == len(positions) - 1:
                self.DONE = True
            print(f"[Progress] [Worker Index:{self.env_id}] [Position Index: {position_idx}/{len(positions)}]")

        print(f"{self.env_id}: DONE Stability Verification")
        print(f"{self.env_id}: Feasible Positions: {len(self.p_feasible)}")