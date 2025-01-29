import sys
sys.path.append('./')
from src import mujoco_sim_base
from src.biped_mpc_jax import *
from src.transformations import *
import numpy as np
import argparse
import yaml

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
    max_steps = 500
    print("max_steps:",max_steps)
    # initialize the controller
    u0 = np.zeros([12,1])
    gait = 0 # standing = 0; walking = 1;
    verbose = False

    # iterators
    t = 0
    steps = 0
    cmd = mpc.x_cmd.copy()
    # loggers 
    references = []
    feedbacks = []
    while True:
        if not sim.viewer_pause:
            base_pos = sim.data.qpos[0:3]
            base_quat = sim.data.qpos[3:7]
            base_eul = quat_to_euler(base_quat)
            body_tvel = sim.data.qvel[0:3]
            body_avel = sim.data.qvel[3:6]
            x_fb = np.concatenate([
                                    base_eul,
                                    base_pos,
                                    body_avel,
                                    body_tvel,
                                    ])     
            q = sim.data.qpos[7:]
            qd = sim.data.qvel[6:]

            # contact sequence generation
            if gait == 1:
                contact = get_contact_sequence(steps/1000, mpc)
            elif gait == 0:
                contact = np.ones((mpc.h, 2))
            t = steps*sim.model.opt.timestep

            pf_w = getFootPositionWorld(x_fb, q)
            cmd[5] = 0.55 +0.05*np.sin(2*np.pi*3.0*t)
            # cmd[0] = np.deg2rad(20)*np.sin(2*np.pi*5*t)
            # cmd[1] = np.deg2rad(20)*np.sin(2*np.pi*3.0*t)

            # decimation
            if np.remainder(steps, mpc.dt*int(1/sim.model.opt.timestep)) == 0:
                Q = np.array([100, 100, 100,  500, 100, 500,  1, 1, 1,   1, 1, 1, 1])
                R = np.array([1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1]) * 1e-6
                foot = pf_w.reshape(-1)
                print(steps,'#'*25)
                _, controls, x_ref = solve_mpc(
                                            # parameters
                                            Q,
                                            R,
                                            # feedback
                                            x_fb, 
                                            foot,
                                            t, 
                                            # commands
                                            contact,
                                            cmd,
                                            )
                u0 = controls[0, :].reshape(-1,1)

            # as fast as possible
            tau = lowLevelControl(x_fb, t, pf_w, q, qd, contact, u0)
            sim.data.ctrl[:] = tau.squeeze()
            
            steps += 1
            # log
            references.append(x_ref[:12,-1].copy()) # ignore  gravity, take the last refernece
            feedbacks.append(x_fb.copy())
            
            if steps > max_steps:
                break

        # pause is also inside step()
        sim.step()

    references = np.array(references)
    feedbacks = np.array(feedbacks)
    print('references:', references.shape, 'feedbacks:', feedbacks.shape)

    # save the logs
    np.savez_compressed('logs/height_control.npz', refrence = references, feedback = feedbacks)

    # plot 
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(2,6 , figsize=(20,6))
    ax = ax.flatten()
    for ref, fb, a in zip(references.T, feedbacks.T, ax):
        a.plot(ref, label='ref')
        a.plot(fb, label='fb')
        a.grid(True)
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Limits to 3 decimal places
        a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Limits to 2 decimal places
    ax[0].legend()
    fig.tight_layout()
    plt.show()


