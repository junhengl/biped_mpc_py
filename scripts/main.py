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
    max_steps = 6000
    print("max_steps:",max_steps)
    # initialize the controller
    mpc = MPC()
    biped = Biped()
    u0 = np.zeros([12,1])

    t = 0
    gait = 0 # standing = 0; walking = 1;
    verbose = False
    # global foot_des_i 
    global foot_l 
    global foot_r 

    # foot_des_i = np.zeros([3, 1])
    # foot_l = np.zeros([3, 1])
    # foot_r = np.zeros([3, 1])
    # print('foot l', foot_l)
    base_pos_tru = []
    base_pos_ref = []
    base_tvel_tru = []
    base_tvel_ref = []

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

            # contact sequence generation
            if gait == 1:
                contact = get_contact_sequence(steps/1000, mpc)
            elif gait == 0:
                contact = np.ones((mpc.h, 2))
            t = steps/1000
            if verbose: print('time: ', t)

            pf_w = getFootPositionWorld(x_fb, q, biped)
            foot = pf_w.reshape(-1)
            # mpc.x_cmd[3] = (foot[0] + foot[3])/2
            # mpc.x_cmd[4] = (foot[1] + foot[4])/2
            mpc.x_cmd[5] = 0.55 +0.05*np.sin(2*np.pi*0.25*t)
            if np.remainder(steps, mpc.dt*1000/1) == 0:
                start_time = time.time()
                states, controls,reference = solve_mpc(x_fb, t, foot, mpc, biped, contact)
                end_time = time.time()
                if verbose:
                    print(f"MPC Function execution time: {end_time - start_time} seconds")
                    print("States: \n", states)
                    print("Controls: \n", controls)
                u0 = controls[0, :].reshape(-1,1)
            tau = lowLevelControl(x_fb, t, pf_w, q, qd, mpc, biped, contact, u0)
            if verbose: print("Torques: \n", tau)
            sim.data.ctrl[:] = tau.squeeze()



            steps += 1
            base_pos_tru.append(sim.data.qpos[0:3].copy())
            base_tvel_tru.append(sim.data.qvel[0:3].copy())
            base_pos_ref.append(reference[3:6,-1].copy())
            base_tvel_ref.append(reference[9:12,-1].copy())
            # print('steps:', steps)
            if steps > max_steps:
                break
    
        sim.step()

    base_pos_tru = np.array(base_pos_tru)
    base_pos_ref = np.array(base_pos_ref)
    base_tvel_tru = np.array(base_tvel_tru)
    base_tvel_ref = np.array(base_tvel_ref)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # firs row position
    axs[0,0].plot(base_pos_tru[:,0], label='true')
    axs[0,0].plot(base_pos_ref[:,0], label='ref')

    axs[0,1].plot(base_pos_tru[:,1], label='true')
    axs[0,1].plot(base_pos_ref[:,1], label='ref')

    axs[0,2].plot(base_pos_tru[:,2], label='true')
    axs[0,2].plot(base_pos_ref[:,2], label='ref')

    # second row velocity
    axs[1,0].plot(base_tvel_tru[:,0], label='true')
    axs[1,0].plot(base_tvel_ref[:,0], label='ref')

    axs[1,1].plot(base_tvel_tru[:,1], label='true')
    axs[1,1].plot(base_tvel_ref[:,1], label='ref')

    axs[1,2].plot(base_tvel_tru[:,2], label='true')
    axs[1,2].plot(base_tvel_ref[:,2], label='ref')


    for ax in axs.flat:
        ax.grid()   
        ax.legend()
    
    fig.tight_layout()
    plt.show()



