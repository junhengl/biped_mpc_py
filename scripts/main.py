import sys
sys.path.append('./')
from src import mujoco_sim_base
from src.bipedalLocomotionMPC import *
from src.transformations import *
import numpy as np
import argparse
import yaml

# TO:CHECK
# 1. joint zeros and axis direction
# 2. control desimation and simualtion dt
# 3. order of tau from MPC 
if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description='Run the simulation')
    argparser.add_argument('--conf_path', type=str, help='Path to the configuration file', default='config/default.yaml')
    argparser.add_argument('--headless', default=False,action='store_true', help='Run the simulation in headless mode')
    args = argparser.parse_args()

    # load the yaml file
    conf = yaml.load(open(args.conf_path, 'r'), Loader=yaml.FullLoader)
    conf['sim']['headless'] = args.headless

    # create the simulation object
    sim = mujoco_sim_base.MujocoSimBase(**conf['sim'])

    # initialize the simulation
    sim.reset()
    steps = 0
    max_steps = np.inf
    print("max_steps:",max_steps)
    # initialize the controller
    mpc = MPC()
    biped = Biped()

    t = 0
    gait = 0 # standing = 0; walking = 1;
    # contact sequence generation
    if gait == 1:
        contact = get_contact_sequence(t, mpc)
    elif gait == 0:
        contact = np.ones((mpc.h, 2))

    while True:
        # pretty_print_low_cmd(cmd)
        if not sim.viewer_pause:
            
            base_pos = sim.data.qpos[0:3]
            base_quat = sim.data.qpos[3:7]
            base_eul = quat_to_euler(base_quat)
            body_tvel = sim.data.qvel[0:3]
            body_avel = sim.data.qvel[3:6]

            # joint: l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle, r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ankle
            jpos = sim.data.qpos[7:]
            jvel = sim.data.qvel[6:]

            x_fb = np.concatenate([
                                    base_eul,
                                    base_pos,
                                    body_avel,
                                    body_tvel,
                                    ])     
            q = jpos
            qd = jvel   
    
            pf_w = getFootPositionWorld(x_fb, q, biped)
            foot = pf_w.reshape(-1)

            start_time = time.time()
            states, controls = solve_mpc(x_fb, t, foot, mpc, biped, contact)
            end_time = time.time()
            print(f"MPC Function execution time: {end_time - start_time} seconds")
            print("States: \n", states)
            print("Controls: \n", controls)
            u0 = controls[0, :].reshape(-1,1)
            tau = lowLevelControl(x_fb, t, pf_w, q, qd, mpc, biped, contact, u0)
            print("Torques: \n", tau)
            sim.data.ctrl[:] = tau.squeeze()

            steps += 1
            if steps > max_steps:
                break
    
        sim.step()


