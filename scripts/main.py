import sys
sys.path.append('./')
from src import mujoco_sim_base
from src.mpc import *
from src.transformations import *
import numpy as np
import argparse
import yaml
from pynput import keyboard

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Run the simulation')
    argparser.add_argument('--conf_path', type=str, help='Path to the configuration file', default='config/default.yaml')
    argparser.add_argument('--headless', default=False,action='store_true', help='Run the simulation in headless mode')
    args = argparser.parse_args()

    # load the yaml file
    SIM_DT = 0.001
    CTRL_DT = 0.02 # 50Hz
    decimation = int(CTRL_DT/SIM_DT) # number of simulation steps per control step


    conf = yaml.load(open(args.conf_path, 'r'), Loader=yaml.FullLoader)
    conf['sim']['headless'] = args.headless

    # create the simulation object
    sim = mujoco_sim_base.MujocoSimBase(**conf['sim'])
    sim.model.opt.timestep = SIM_DT  # set the simulation time step

    # initialize the simulation
    sim.reset()
    steps = 0
    max_steps = np.inf
    print("max_steps:",max_steps)
    # initialize the controller
    mpc = MPC()
    biped = Biped()
    u0 = np.zeros([12,1])

    t = 0
    gait = 1 # standing = 0; walking = 1;

    key_pressed = False

    # keyboard utils
    def on_press(key):
        global key_pressed
        key_pressed = True
        try:
            print('step:',steps,end=' ')
            if key == keyboard.Key.up:
                mpc.x_cmd[9] = 0.5
                mpc.x_cmd[3] += mpc.x_cmd[9] * SIM_DT 
                print('mpc.x_cmd[3]:', mpc.x_cmd[3], 'mpc.x_cmd[9]:', mpc.x_cmd[9])
            elif key == keyboard.Key.down:
                mpc.x_cmd[9] = -0.5
                mpc.x_cmd[3] += mpc.x_cmd[9] * SIM_DT
                print('mpc.x_cmd[3]:', mpc.x_cmd[3], 'mpc.x_cmd[9]:', mpc.x_cmd[9])
            elif key == keyboard.Key.left:
                mpc.x_cmd[10] = -0.3
                mpc.x_cmd[4] += mpc.x_cmd[10] * SIM_DT 
                print('mpc.x_cmd[4]:', mpc.x_cmd[4], 'mpc.x_cmd[10]:', mpc.x_cmd[10])
            elif key == keyboard.Key.right:
                mpc.x_cmd[10] = 0.3
                mpc.x_cmd[4] += mpc.x_cmd[10] * SIM_DT 
                print('mpc.x_cmd[4]:', mpc.x_cmd[4], 'mpc.x_cmd[10]:', mpc.x_cmd[10])

        except AttributeError:
            print(f'Special key {key} pressed')

    def on_release(key):
        global key_pressed
        key_pressed = False
        if key == keyboard.Key.esc:
            # Stop listener
            return False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print('######################## keyboard setup ########################')
    print('up    - vx = 0.5  and px += vx * dt')
    print('down  - vx = -0.5 and px += vx * dt')
    print('left  - vy = -0.3 and py += vy * dt')
    print('right - vy = 0.3  and py += vy * dt')
    print('space - pause/unpause')
    print('key relase and no key pressed - stand at current position')
    print('#################################################################')
    while True:
        # pretty_print_low_cmd(cmd)
        if not sim.viewer_pause:
            if not key_pressed:
                base_pos = sim.data.qpos[0:3]
                base_eul = quat_to_euler(sim.data.qpos[3:7])
                mpc.x_cmd[2] = base_eul[2] # yaw
                mpc.x_cmd[3] = base_pos[0] # x
                mpc.x_cmd[4] = base_pos[1] # y
                for i in range(3):
                    mpc.x_cmd[6+i] = 0 
                    mpc.x_cmd[9+i] = 0
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
            t = steps * SIM_DT
            # print('time: ', t)
            pf_w = getFootPositionWorld(x_fb, q, biped)
            foot = pf_w.reshape(-1)

            if steps % decimation == 0:
                start_time = time.time()
                states, controls = solve_mpc(x_fb, t, foot, mpc, biped, contact)
                end_time = time.time()
                # print(f"MPC Function execution time: {end_time - start_time} seconds")
                # print("States: \n", states)
                # print("Controls: \n", controls)
                u0 = controls[0, :].reshape(-1,1)
            
            tau = lowLevelControl(x_fb, t, pf_w, q, qd, mpc, biped, contact, u0)
            # print("Torques: \n", tau)
            sim.data.ctrl[:] = tau.squeeze()

            steps += 1
            if steps > max_steps:
                break
    
        sim.step()