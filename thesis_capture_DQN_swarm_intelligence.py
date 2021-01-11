import logging

import logger as logger
import time

import gym
import json
from thesis.CDF_swarm_intelligence.thesis_env_swarm_intelligence import ThesisEnvExperiment
import datetime as dt
from malmo import MalmoPython
import random
from malmopy.agent.agent import ReplayMemory
import argparse
import os
import sys
from chainer import optimizers
import gym
from gym import spaces
import numpy as np
import chainerrl
from thesis.chainer_dqn import DQN
from thesis.observer import OBSERVER
from thesis.chainerrl import experiments
from thesis.chainerrl import explorers
from thesis.chainerrl import links
from thesis.chainerrl import misc
from thesis.chainerrl import q_functions
from thesis.chainerrl import replay_buffer
from thesis.chainerrl.experiments.evaluator import Evaluator
from thesis.chainerrl.experiments.evaluator import save_agent

if sys.version_info[0] == 2:
    import Tkinter as tk
else:
    import tkinter as tk

if __name__ == '__main__':

    """ build the environment """
    env = ThesisEnvExperiment()

    """ define some useful parameters to separate the agents """
    num_01 = 1  # Agent number 1
    num_02 = 2  # Agent number 2
    num_03 = 3  # Agent number 3
    num_04 = 4  # Agent number 4

    overall_reward_agent_Tom = 0
    overall_reward_agent_Jerry = 0
    overall_reward_agent_roadrunner = 0
    overall_reward_agent_coyote = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='ThesisEnvExperiment')
    parser.add_argument('--seed', type=int, default=1423,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
    args = parser.parse_args()

    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    """ initialize clientpool and environment """
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001), ('127.0.0.1', 10002), ('127.0.0.1', 10003),
                   ('127.0.0.1', 10004)]
    env.init(start_minecraft=False, client_pool=client_pool)

    """ WIP: need to skip that """
    test = True

    """ set the seed """
    env_seed = args.seed
    env.seed(env_seed)

    """ Cast observations to float32 because the model needs float32 """
    env = chainerrl.wrappers.CastObservationToFloat32(env)

    if args.monitor:
        env = chainerrl.wrappers.Monitor(env, args.outdir)
    if isinstance(env.action_space, spaces.Box):
        misc.env_modifiers.make_action_filtered(env, env.clip_action_filter)
    if not test:
        env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
    if ((args.render_eval and test) or
            (args.render_train and not test)):
        env = chainerrl.wrappers.Render(env)

    timestep_limit = 9000
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    """ calculate everything to start the mission """
    q_func, opt, rbuf, explorer = env.dqn_q_values_and_neuronal_net(args, action_space, obs_size, obs_space)

    """ 
    initialize agents
    tom:            player, mode = survival
    jerry:          player, mode = survival
    roadrunner:     player, mode = survival
    coyote:         player, mode = survival
    skye:           observer, mode = creative 
    """
    tom = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
              explorer=explorer, replay_start_size=args.replay_start_size,
              target_update_interval=args.target_update_interval,
              update_interval=args.update_interval,
              minibatch_size=args.minibatch_size,
              target_update_method=args.target_update_method,
              soft_update_tau=args.soft_update_tau,
              )
    jerry = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_interval=args.target_update_interval,
                update_interval=args.update_interval,
                minibatch_size=args.minibatch_size,
                target_update_method=args.target_update_method,
                soft_update_tau=args.soft_update_tau,
                )
    roadrunner = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                     explorer=explorer, replay_start_size=args.replay_start_size,
                     target_update_interval=args.target_update_interval,
                     update_interval=args.update_interval,
                     minibatch_size=args.minibatch_size,
                     target_update_method=args.target_update_method,
                     soft_update_tau=args.soft_update_tau,
                     )
    coyote = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 minibatch_size=args.minibatch_size,
                 target_update_method=args.target_update_method,
                 soft_update_tau=args.soft_update_tau,
                 )
    skye = OBSERVER()

    if args.load:
        tom.load(args.load)
        jerry.load(args.load)
        roadrunner.load(args.load)
        coyote.load(args.load)

    """ 
    initialize env parameters 
    env:                    environment container
    steps:                  maximum episode number
    eval_n_steps:           ..
    eval_interval:          ..
    outdir:                 directory to save the results
    train_max_episode_len:  ..
    logger:                 to log errors and information 
    step_offset:            startingpoint, is 0 at the beginning
    eval_max_episode_len:   ..
    successful_score:       ..
    step_hooks:             ..
    save_best_so_far_agent: .. not yet              
    """
    env = env
    steps = args.steps
    eval_n_steps = None
    eval_n_episodes = args.eval_n_runs
    eval_interval = args.eval_interval
    outdir = args.outdir
    train_max_episode_len = timestep_limit
    logger = logger or logging.getLogger(__name__)

    step_offset = 0
    eval_max_episode_len = None
    successful_score = None
    step_hooks = ()
    save_best_so_far_agent = True

    os.makedirs(outdir, exist_ok=True)

    eval_max_episode_len = train_max_episode_len

    """ evaluator to save best so far agent ---  WIP
    evaluator1 = Evaluator(agent=tom,
                           n_steps=eval_n_steps,
                           n_episodes=eval_n_episodes,
                           eval_interval=eval_interval, outdir=outdir,
                           max_episode_len=eval_max_episode_len,
                           env=env,
                           step_offset=step_offset,
                           save_best_so_far_agent=save_best_so_far_agent,
                           logger=logger,
                           )

    evaluator2 = Evaluator(agent=jerry,
                           n_steps=eval_n_steps,
                           n_episodes=eval_n_episodes,
                           eval_interval=eval_interval, outdir=outdir,
                           max_episode_len=eval_max_episode_len,
                           env=env,
                           step_offset=step_offset,
                           save_best_so_far_agent=save_best_so_far_agent,
                           logger=logger,
                           )
    """
    """ 
    int r1/r2:                      initialisation of the reward per agent, starts at 0
    bool done_team01/done_team02:   mission is running as long as 'done'-flag is true 
    int t:                          current step of episode, starts at 0
    int max_episode_len:            None, set at another point
    float time_step_start:          start time of the episode, to track length for timeout
    string experiment_ID:           must be the same in every agent to combine them in one arena, must be a string
    """

    r1 = r2 = r3 = r4 = 0
    done_team01 = done_team02 = False

    t = step_offset

    if hasattr(tom, 't') and hasattr(jerry, 't') and hasattr(roadrunner, 't') and hasattr(coyote, 't'):
        tom.t = step_offset
        jerry.t = step_offset
        roadrunner.t = step_offset
        coyote.t = step_offset

    max_episode_len = None

    time_stamp_start = time.time()
    print("actual time: ", time_stamp_start)

    experiment_ID = ""

    """ initial observation """
    obs1, obs2, obs3, obs4 = env.reset_world(experiment_ID)

    """ save new episode start into result.txt """
    env.save_new_round(t)

    """ start training episodes """
    while t < steps:
        try:
            while not done_team01 and not done_team02:

                """ current time to calculate the elapsed time """
                time_stamp_actual = time.time()
                time_step = time_stamp_actual - time_stamp_start
                print("mission time up: %i sec" % (time_step))


                """ calculate the next steps for the agents """
                action1 = tom.act_and_train(obs1, r1)
                action2 = jerry.act_and_train(obs2, r2)
                action3 = roadrunner.act_and_train(obs3, r3)
                action4 = coyote.act_and_train(obs4, r4)
                #time.sleep(1)
                """ check if agents would run into each other when they do the calculated step """
                action1, action2, action3, action4 = env.approve_distance(tom, jerry, roadrunner, coyote, obs1, obs2,
                                                                          obs3, obs4, r1, r2, r3, r4, action1, action2,
                                                                          action3, action4, time_step)
                #time.sleep(1)
                """ do calculated steps and pass information of the time_step """
                obs1, r1, done1, info1 = env.step_generating(action1, 1)
                obs2, r2, done2, info2 = env.step_generating(action2, 2)
                obs3, r3, done3, info3 = env.step_generating(action3, 3)
                obs4, r4, done4, info4 = env.step_generating(action4, 4)
                #time.sleep(1)

                if done1 or done3:
                    done_team01 = True
                if done2 or done4:
                    done_team02 = True

                """ sum up the reward for every episode """
                overall_reward_agent_Tom += r1
                overall_reward_agent_Jerry += r2
                overall_reward_agent_roadrunner += r3
                overall_reward_agent_coyote += r4
                print("Current Reward Tom:   ", overall_reward_agent_Tom)
                print("Current Reward Jerry: ", overall_reward_agent_Jerry)
                print("Current Reward Roadrunner:   ", overall_reward_agent_Tom)
                print("Current Reward Coyote: ", overall_reward_agent_Jerry)

                """ hook """
                for hook in step_hooks:
                    hook(env, tom, t)
                    hook(env, jerry, t)
                    hook(env, roadrunner, t)
                    hook(env, coyote, t)

                """ check, if an agent captured the flag """
                env.check_inventory(time_step)

                print("--------------------------------------------------------------------")

                """ end mission when one agent finishes, the agents crash or the time is over, start over again """
                if env.mission_end or done_team01 or done_team02 or (time_step > 1920):  # 960 = 16 min | 1920 = 32 min

                    """ send mission QuitCommands to tell Malmo that the mission has ended,save and reset everything """
                    t, obs1, obs2, r1, r2, obs3, obs4, r3, r4, done_team01, done_team02, overall_reward_agent_Jerry, \
                    overall_reward_agent_Tom, overall_reward_agent_roadrunner, overall_reward_agent_coyote = \
                        env.sending_mission_quit_commands(overall_reward_agent_Tom, overall_reward_agent_Jerry,
                                                          overall_reward_agent_roadrunner, overall_reward_agent_coyote,
                                                          time_step, obs1, r1, obs2, r2, obs3, r3, obs4, r4, outdir, t,
                                                          tom, jerry, roadrunner, coyote, experiment_ID)

                    time_stamp_start = time.time()

                    """ recover """
                    time.sleep(5)

                if t == 1001:
                    print("Mission-Set finished. Congratulations! Check results and parameters. Start over.")
                    break

        except (Exception, KeyboardInterrupt):
            # Save the current model before being killed
            save_agent(tom, t, outdir, logger, suffix='_except01')
            save_agent(jerry, t, outdir, logger, suffix='_except02')
            save_agent(roadrunner, t, outdir, logger, suffix='_except03')
            save_agent(coyote, t, outdir, logger, suffix='_except04')
            raise
