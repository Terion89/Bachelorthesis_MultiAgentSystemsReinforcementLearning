import logging

import logger as logger
import time

import gym
import json
from thesis.thesis_env_experiment import ThesisEnvExperiment
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
from chainerrl.agents.dqn import DQN
from thesis.observer import OBSERVER
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer
from chainerrl.experiments.evaluator import Evaluator
from chainerrl.experiments.evaluator import save_agent

# main
if __name__ == '__main__':

    """ build the environment """
    env = ThesisEnvExperiment()

    """ define some useful parameters """
    num_01 = 1  # Agentnumber 1
    num_02 = 2  # Agentnumber 2

    overall_reward_agent_Tom = 0
    overall_reward_agent_Jerry = 0

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
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001), ('127.0.0.1', 10002)]
    env.init(start_minecraft=False, client_pool=client_pool)

    test = True

    """ set the seed """
    env_seed = args.seed
    env.seed(env_seed)

    """Cast observations to float32 because the model needs float32"""
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

    """ mission is running as long as 'done'-flag is true """
    # print(args)
    q_func, opt, rbuf, explorer = env.dqn_q_values_and_neuronal_net(args, action_space, obs_size, obs_space)

    """ 
    initialize agents
    tom:        player, mode = survival
    jerry:      player, mode = survival
    skye:       observer, mode = creative 
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
    skye = OBSERVER()

    if args.load:
        tom.load(args.load)
        jerry.load(args.load)

    """ 
    initialize env parameters 
    env:                    environment container
    steps:                  maximum episode number
    eval_n_steps:           
    eval_interval:          
    outdir:                 directory to save the results
    train_max_episode_len:  
    logger:                 to log errors and information 
    step_offset:            startingpoint, is 0 at the beginning
    eval_max_episode_len: 
    successful_score:   
    step_hooks:      
    save_best_so_far_agent:               
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

    """ evaluator to save best so far agent """
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
    int r1/r2:                  initialisation of the reward per agent, starts at 0
    bool done1/done2:           mission is running as long as 'done'-flag is true 
    int t:                      current step of episode, starts at 0
    int max_episode_len:        None, set at another point
    float time_step_start:      start time of the episode, to track length for timeout
    string experiment_ID:       must be the same in every agent to combine them in one arena, must be a string
    """

    r1 = r2 = 0
    done1 = done2 = False

    t = step_offset

    if hasattr(tom, 't') and hasattr(jerry, 't'):
        tom.t = step_offset
        jerry.t = step_offset

    max_episode_len = None

    time_stamp_start = time.time()
    print("actual time: ", time_stamp_start)

    experiment_ID = "a"

    """ initial observation """
    obs1, obs2 = env.reset_world(experiment_ID)

    """ save new episode into result.txt """
    env.save_new_round(t)

    """ start training episodes """
    while t < steps:
        try:
            while not done1 and not done2:

                """ current time to calculate the elapsed time """
                time_stamp_actual = time.time()

                """ calculate the next steps for the agents """
                action1 = tom.act_and_train(obs1, r1)
                action2 = jerry.act_and_train(obs2, r2)

                """ do calculated steps and pass information of the time_step """
                obs1, r1, done1, info1 = env.step_generating(action1, 1)
                obs2, r2, done2, info2 = env.step_generating(action2, 2)

                """ sum up the reward for every episode """
                overall_reward_agent_Tom += r1
                overall_reward_agent_Jerry += r2
                print("Actual Reward Tom:   ", overall_reward_agent_Tom)
                print("Actual Reward Jerry: ", overall_reward_agent_Jerry)
                print("--------------------------------------------------------------------")

                """ check if agents run too close """
                env.distance()

                """ hook """
                for hook in step_hooks:
                    hook(env, tom, t)
                    hook(env, jerry, t)

                time_step = time_stamp_actual - time_stamp_start

                """ check, if an agent captured the flag """
                env.check_inventory(time_step)

                print("mission time up: %i sec" % (time_step))

                """ end mission when both agents finished or time is over, start over again """
                if (done1 and done2) or (time_step > 1920) or (env.mission_end == True):  # 960 = 16 min | 1920 = 32 min

                    """ send mission QuitCommands to tell Malmo that the mission has ended """
                    env.agent_host1.sendCommand("quit")
                    env.agent_host2.sendCommand("quit")
                    env.agent_host3.sendCommand("quit")

                    """ save and show results of reward calculations """
                    env.save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step)
                    print("Final Reward Tom:   ", overall_reward_agent_Tom)
                    print("Final Reward Jerry: ", overall_reward_agent_Jerry)

                    """ end episode, save results """
                    tom.stop_episode_and_train(obs1, r1, done=True)
                    jerry.stop_episode_and_train(obs2, r2, done=True)
                    print("outdir: %s step: %s " % (outdir, t))
                    print("Tom's statistics:   ", tom.get_statistics())
                    print("Jerry's statistics: ", jerry.get_statistics())

                    """ save the final model and results """
                    save_agent(tom, t, outdir, logger, suffix='_finish_01')
                    save_agent(jerry, t, outdir, logger, suffix='_finish_02')



                    """ initialisation for the next episode, reset parameters, build new world """
                    t += 1
                    r1 = r2 = 0
                    done1 = done2 = False
                    overall_reward_agent_Jerry = overall_reward_agent_Tom = 0
                    time_stamp_start = time.time()
                    env.save_new_round(t)
                    obs1, obs2 = env.reset_world(experiment_ID)
                    # print("obs1: ", obs1)
                    # print("\nobs2: ", obs2)

                    """ recover """
                    time.sleep(5)
                    """if evaluator1 and evaluator2 is not None:
                        evaluator1.evaluate_if_necessary(
                            t=t, episodes=episode_idx + 1)
                        evaluator2.evaluate_if_necessary(
                            t=t, episodes=episode_idx + 1)
                        if (successful_score is not None and
                                evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                            break"""
                if t == 1000:
                    print("Mission-Set finished. Congratulations! Check results and parameters. Start over.")
                    break

        except (Exception, KeyboardInterrupt):
            # Save the current model before being killed
            save_agent(tom, t, outdir, logger, suffix='_except01')
            save_agent(jerry, t, outdir, logger, suffix='_except02')
            raise
