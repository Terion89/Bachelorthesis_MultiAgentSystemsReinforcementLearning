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
    num_01 = 1 # Agentnumber 1
    num_02 = 2 # Agentnumber 2

    overall_reward_agent_Tom = 0
    overall_reward_agent_Jerry = 0

    #max_size = 1000
    #state_shape = (84, 84, 3)

    #buf_01 = ReplayMemory(max_size, state_shape)
    #buf_02 = ReplayMemory(max_size, state_shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='ThesisEnvExperiment')
    parser.add_argument('--seed', type=int, default=0,
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
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    env.init(start_minecraft=False, client_pool=client_pool)

    test = True

    # Use different random seeds for train and test envs
    env_seed = args.seed
    env.seed(env_seed)
    # Cast observations to float32 because our model uses float32
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
    done = False
    #print(args)
    q_func, opt, rbuf, explorer = env.dqn_q_values_and_neuronal_net(args, action_space, obs_size, obs_space)

    #print("q_func: ", q_func)
    #print("opt: ", opt)
    #print("rbuf: ", rbuf)
    #print("explorer: ", explorer)

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

    if args.load:
        tom.load(args.load)
        jerry.load(args.load)

    #tom = env.agent_host1
    #jerry = env.agent_host2
    env = env
    steps = args.steps
    eval_n_steps = None
    eval_n_episodes = args.eval_n_runs
    eval_interval = args.eval_interval
    outdir = args.outdir
    eval_env = env
    train_max_episode_len = timestep_limit
    logger = logger or logging.getLogger(__name__)

    checkpoint_freq = None
    train_max_episode_len = None
    step_offset = 0
    eval_max_episode_len = None
    successful_score = None
    step_hooks = ()
    save_best_so_far_agent = True

    os.makedirs(outdir, exist_ok=True)

    eval_max_episode_len = train_max_episode_len

    evaluator1 = Evaluator(agent=tom,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    evaluator2 = Evaluator(agent=jerry,
                           n_steps=eval_n_steps,
                           n_episodes=eval_n_episodes,
                           eval_interval=eval_interval, outdir=outdir,
                           max_episode_len=eval_max_episode_len,
                           env=eval_env,
                           step_offset=step_offset,
                           save_best_so_far_agent=save_best_so_far_agent,
                           logger=logger,
                           )

    # train agent


    """ initial observation """
    obs1, obs2 = env.reset_world()

    episode_r = 0
    episode_idx = 0

    r1 = 0
    r2 = 0
    done1 = False
    done2 = False
    done1_fin = False
    done2_fin = False

    t = step_offset
    if hasattr(tom, 't') and hasattr(jerry, 't'):
        tom.t = step_offset
        jerry.t = step_offset

    episode_len = 0
    max_episode_len = None


    while t < steps:
        try:
            while not done1 and not done2:
                env.mission_spec.drawBlock(1, 47, 13, "log")
                env.mission_spec.drawBlock(14, 47, 1, "coal_ore")

                #env.render_first_agent()
                #env.render_second_agent()


                print("calculating next steps...")
                action1 = tom.act_and_train(obs1, r1)
                action2 = jerry.act_and_train(obs2, r2)

                obs1, r1, done1, info1 = env.step_generating(action1, 1)
                obs2, r2, done2, info2 = env.step_generating(action2, 2)

                overall_reward_agent_Tom += r1
                overall_reward_agent_Jerry += r2


                print("Actual Reward Tom:   ", overall_reward_agent_Tom)
                print("Actual Reward Jerry: ", overall_reward_agent_Jerry)
                print("--------------------------------------------------------------------")

                for hook in step_hooks:
                    hook(env, tom, t)
                    hook(env, jerry, t)

                if done1:
                    pass

                if done2:
                    pass
                if done1 and done2:
                    tom.stop_episode_and_train(obs1, r1, done=done)
                    jerry.stop_episode_and_train(obs2, r2, done=done)
                    print("outdir: %s step: %s episode: %s R: %s" % (outdir, t, episode_idx, episode_r))
                    print("statistics: ", tom.get_statistics())
                    print("statistics: ", jerry.get_statistics())
                    t += 1
                    # Save the final model
                    save_agent(tom, t, outdir, logger, suffix='_finish_01')
                    save_agent(jerry, t, outdir, logger, suffix='_finish_02')
                    """
                    show results of calculations
                    """
                    print("Final Reward Tom:   ", overall_reward_agent_Tom)
                    print("Final Reward Jerry: ", overall_reward_agent_Jerry)

                    if evaluator1 and evaluator2 is not None:
                        evaluator1.evaluate_if_necessary(
                            t=t, episodes=episode_idx + 1)
                        evaluator2.evaluate_if_necessary(
                            t=t, episodes=episode_idx + 1)
                        if (successful_score is not None and
                                evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                            break
                    if t == steps:
                        print("Mission-Set finished. Check results and parameters. Start over.")
                        break
                    # Start a new episode
                    episode_r = 0
                    episode_idx += 1
                    episode_len = 0
                    obs1, obs2 = env.reset_world()
                    r1 = 0
                    overall_reward_agent_Tom =0
                    r2 = 0
                    overall_reward_agent_Jerry =0
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(tom, t, outdir, logger, suffix='_checkpoint_01')
                    save_agent(jerry, t, outdir, logger, suffix='_checkpoint_01')

            """
            choose random actions for testing the functions
            """
            #action1 = env.action_names[0][random.randint(0, len(env.action_names[0])-1)]
            #time.sleep(0.5)
            #action2 = env.action_names[0][random.randint(0, len(env.action_names[0])-1)]



        except (Exception, KeyboardInterrupt):
            # Save the current model before being killed
            save_agent(tom, t, outdir, logger, suffix='_except01')
            save_agent(jerry, t, outdir, logger, suffix='_except02')
            raise







