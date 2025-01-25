import mujoco.viewer
import numpy as np
import mujoco
import time

class MujocoSimBase:

    def __init__(
                self, 
                model_path, 
                headless=False,
                viewer_fps=24,
                ):
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.world_root_constraint_id = self.obj_name2id('world_root', type='equality')
        self.base_pos_nominal = np.array([0.0, 0.0, 0.55])
        self.data = mujoco.MjData(self.model)
        self.headless = headless
        self.start_time = time.time()
        self.viewer_sync_rate = 1.0 / viewer_fps
        if self.headless:
            self.step = self.step_headless
        else:
            self.viewer = mujoco.viewer.launch_passive(
                                                                    self.model, 
                                                                    self.data,
                                                                    show_left_ui=False,
                                                                    show_right_ui=False,
                                                                    key_callback=self.viewer_key_callback,
                                                                    ) 
            self.viewer_pause = False
            self.step = self.step_head


    def viewer_key_callback(self,keycode):
        if chr(keycode) == ' ':
            self.viewer_pause = not self.viewer_pause
        elif chr(keycode) == 'E':
            self.viewer.opt.frame = not self.viewer.opt.frame
        elif chr(keycode) == 'Q':
            self.set_robot_on_ground()

    def step_headless(self):
        mujoco.mj_step(self.model, self.data)

    # def reset(self):
    #     init_qp = np.array(self.model.keyframe('home').qpos)
    #     mujoco.mj_resetData(self.model,self.data) 
    #     self.data.qpos[:] = init_qp
    #     self.step()
    #     self.viewer.sync()
    #     self.viewer_pause = True

    def reset(self):
        init_qp = np.array(self.model.keyframe('home').qpos)
        mujoco.mj_resetData(self.model, self.data) 
        self.data.qpos[:] = init_qp
        self.step()
        
        # Only sync the viewer if it exists (not in headless mode)
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.sync()
        
        self.viewer_pause = True


    def step_head(self):
        if self.viewer.is_running():
            if not self.viewer_pause:
                mujoco.mj_step(self.model, self.data)
                if (time.time() - self.start_time) % self.viewer_sync_rate < 1e-3:
                    self.viewer.sync()
        else:
            exit()


    def obj_name2id(self,name,type='body'):
        type = type.upper()
        return mujoco.mj_name2id(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    name
                                )

    def obj_id2name(self,obj_id,type='body'):
        type = type.upper() 
        return mujoco.mj_id2name(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    obj_id
                                )

    def get_sensordata_from_id(self,sensor_id):
            
        start_n = self.model.sensor_adr[sensor_id]
    
        if sensor_id == self.model.nsensor -1:
            return self.data.sensordata[start_n:]
        else:
            end_n = self.model.sensor_adr[sensor_id+1]
            return self.data.sensordata[start_n:end_n]

    def set_robot_on_ground(self):
        self.model.eq_active0[self.world_root_constraint_id] = 0
        self.data.eq_active[self.world_root_constraint_id] = 0
        self.data.qpos[2] = self.base_pos_nominal[2] + 0.01
        self.data.qvel = 0.0
        # self.step()
        self.viewer.sync()  
        self.viewer_pause = True

    def set_robot_off_ground(self):
        self.data.qpos[:2] = 0.0
        self.data.qpos[2] = 1.0
        self.data.qpos[3] = 1.0
        self.data.qpos[4:] = 0.0
        self.data.qvel = 0.0
        for _ in range(5):
            self.step()
        self.model.eq_active0[self.world_root_constraint_id] = 1
        self.data.eq_active[self.world_root_constraint_id] = 1
        for _ in range(10):
            self.step()
        self.viewer.sync()