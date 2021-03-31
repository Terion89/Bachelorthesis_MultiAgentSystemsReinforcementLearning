import logging

import logger as logger
import time

from build.install.Python_Examples.chainer_dqn import DQN
from build.install.Python_Examples.thesis_env_swarm_intelligence import ThesisEnvExperiment
import argparse
import os
import sys
from gym import spaces
import chainerrl

from observer import OBSERVER
from thesis.chainerrl.chainerrl import experiments
from thesis.chainerrl.chainerrl import misc
from thesis.chainerrl.chainerrl.experiments.evaluator import save_agent

if sys.version_info[0] == 2:
    pass
else:
    pass

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

    # misc.set_random_seed(args.seed, gpus=(args.gpu,))
    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    """ initialize clientpool and environment """
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001), ('127.0.0.1', 10002), ('127.0.0.1', 10003),
                   ('127.0.0.1', 10004), ('127.0.0.1', 10005), ('127.0.0.1', 10006)]
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
    print("action_space: ", action_space)

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
    skye = OBSERVER()

    jerry = DQN(q_func, opt, rbuf, gpu=None, gamma=args.gamma,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_interval=args.target_update_interval,
                update_interval=args.update_interval,
                minibatch_size=args.minibatch_size,
                target_update_method=args.target_update_method,
                soft_update_tau=args.soft_update_tau,
                )
    roadrunner = DQN(q_func, opt, rbuf, gpu=None, gamma=args.gamma,
                     explorer=explorer, replay_start_size=args.replay_start_size,
                     target_update_interval=args.target_update_interval,
                     update_interval=args.update_interval,
                     minibatch_size=args.minibatch_size,
                     target_update_method=args.target_update_method,
                     soft_update_tau=args.soft_update_tau,
                     )
    coyote = DQN(q_func, opt, rbuf, gpu=None, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 minibatch_size=args.minibatch_size,
                 target_update_method=args.target_update_method,
                 soft_update_tau=args.soft_update_tau,
                 )
    tom = DQN(q_func, opt, rbuf, gpu=None, gamma=args.gamma,
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
    step_offset:            startingpoint, is 1 at the beginning
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

    step_offset = 1
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
    # int r<>:                        initialisation of the reward per agent, starts at 0
    bool done<>:                    mission is running as long as 'done'-flag is false 
    int t:                          current step of episode, starts at 0
    int max_episode_len:            None, set at another point
    float time_step_start:          start time of the episode, to track length for timeout
    string experiment_ID:           must be the same in every agent to combine them in one arena, must be a string
    """

    t = step_offset

    if hasattr(tom, 't') and hasattr(jerry, 't') and hasattr(roadrunner, 't') and hasattr(coyote, 't'):
        tom.t = step_offset
        jerry.t = step_offset
        roadrunner.t = step_offset
        coyote.t = step_offset

    max_episode_len = None

    time_stamp_start = time.time()
    print("current time: ", time_stamp_start)

    experiment_ID = ""

    """ initial observation """
    obs1, obs2, obs3, obs4 = env.reset_world(experiment_ID)
    r1 = r2 = r3 = r4 = 0

    """ save new episode start into result.txt """
    env.save_new_round(t)

    """ start training episodes """
    while t < steps:
        try:
            while not env.done_team01 and not env.done_team02:

                """ current time to calculate the elapsed time """
                time_stamp_current = time.time()
                time_step = time_stamp_current - time_stamp_start
                print("mission time up: %i sec" % (time_step))

                """ calculate the next steps for the agents """
                action1 = tom.act_and_train(obs1, r1)
                action2 = jerry.act_and_train(obs2, r2)
                action3 = roadrunner.act_and_train(obs3, r3)
                action4 = coyote.act_and_train(obs4, r4)
                # time.sleep(1)

                """ check if agents would run into each other when they do the calculated step """
                action1, action2, action3, action4 = env.approve_distance(tom, jerry, roadrunner, coyote, env.obs1,
                                                                          obs2, obs3, obs4, r1, r2,
                                                                          r3, r4, action1, action2,
                                                                          action3, action4, time_step)

                # time.sleep(1)
                """ do calculated steps, check if they were executed and pass information of the time_step """
                while not env.command_executed_tom:
                    world_state1 = env.agent_host1.peekWorldState()
                    """ checks, if world_state is read correctly, if not, trys again"""
                    while world_state1.number_of_observations_since_last_state == 0:
                        world_state1 = env.agent_host1.peekWorldState()
                        print(".")
                        time.sleep(0.01)
                    env.x1_prev, y1, env.z1_prev = env.get_position_in_arena(world_state1, time_step, 1)
                    obs1, r1, done1, info1 = env.step_generating(action1, 1)
                    env.command_executed_tom = env.check_if_command_was_executed(1, time_step)
                    if done1:
                        env.done_team01 = True
                        
                    env.overall_reward_agent_Tom += r1
                    print("Current Reward Tom:   ", env.overall_reward_agent_Tom)
                    
                while not env.command_executed_jerry:
                    world_state2 = env.agent_host2.peekWorldState()
                    """ checks, if world_state is read correctly, if not, trys again"""
                    while world_state2.number_of_observations_since_last_state == 0:
                        world_state2 = env.agent_host2.peekWorldState()
                        print(".")
                        time.sleep(0.01)
                    env.x2_prev, y2, env.z2_prev = env.get_position_in_arena(world_state2, time_step, 2)
                    obs2, r2, done2, info2 = env.step_generating(action2, 2)
                    env.command_executed_jerry = env.check_if_command_was_executed(2, time_step)
                    
                    if done2:
                        env.done_team02 = True                
                    
                    env.overall_reward_agent_Jerry += r2
                    print("Current Reward Jerry: ", env.overall_reward_agent_Jerry)

                while not env.command_executed_roadrunner:
                    world_state3 = env.agent_host3.peekWorldState()
                    """ checks, if world_state is read correctly, if not, trys again"""
                    while world_state3.number_of_observations_since_last_state == 0:
                        world_state3 = env.agent_host3.peekWorldState()
                        print(".")
                        time.sleep(0.01)
                    env.x3_prev, y3, env.z3_prev = env.get_position_in_arena(world_state3, time_step, 3)
                    obs3, r3, done3, info3 = env.step_generating(action3, 3)
                    print(env.r3)
                    env.command_executed_roadrunner = env.check_if_command_was_executed(3, time_step)
                    if done3:
                        env.done_team01 = True
                        
                    env.overall_reward_agent_roadrunner += r3
                
                    print("Current Reward Roadrunner:   ", env.overall_reward_agent_Tom)
                
                while not env.command_executed_coyote:
                    world_state4 = env.agent_host4.peekWorldState()
                    """ checks, if world_state is read correctly, if not, trys again"""
                    while world_state4.number_of_observations_since_last_state == 0:
                        world_state4 = env.agent_host4.peekWorldState()
                        print(".")
                        time.sleep(0.01)
                    env.x4_prev, y4, env.z4_prev = env.get_position_in_arena(world_state4, time_step, 4)
                    obs4, r4, done4,  info4 = env.step_generating(action4, 4)
                    env.command_executed_coyote = env.check_if_command_was_executed(4, time_step)

                    if done4:
                        env.done_team02 = True
                
                    env.overall_reward_agent_coyote += r4

                    print("Current Reward Coyote: ", env.overall_reward_agent_Jerry)
                    
                print("Current Reward Tom:   ", env.overall_reward_agent_Tom)
                print("Current Reward Jerry: ", env.overall_reward_agent_Jerry)
                print("Current Reward Roadrunner:   ", env.overall_reward_agent_Tom)
                print("Current Reward Coyote: ", env.overall_reward_agent_Jerry)

                """ hook """
                for hook in step_hooks:
                    hook(env, tom, t)
                    hook(env, jerry, t)
                    hook(env, roadrunner, t)
                    hook(env, coyote, t)

                """ check, if an agent captured the flag """
                env.check_inventory(time_step)

                print("--------------------------------------------------------------------")

                """ reset the command_executed flags for the next step """
                env.command_executed_tom = env.command_executed_jerry = False
                env.command_executed_roadrunner = env.command_executed_coyote = False

                """ end mission when one agent finishes, the agents crash or the time is over, start over again """
                if env.mission_end or env.done_team01 or env.done_team02 or (time_step > 1920):
                    # 960 = 16 min | 1920 = 32 min

                    """ send mission QuitCommands to tell Malmo that the mission has ended,save and reset everything """
                    t, obs1, obs2, r1, r2, obs3, obs4, r3, r4, env.done_team01, \
                        env.done_team02, env.overall_reward_agent_Jerry, env.overall_reward_agent_Tom, \
                        env.overall_reward_agent_roadrunner, env.overall_reward_agent_coyote = \
                        env.sending_mission_quit_commands(env.overall_reward_agent_Tom, env.overall_reward_agent_Jerry,
                                                          env.overall_reward_agent_roadrunner,
                                                          env.overall_reward_agent_coyote,
                                                          time_step, obs1, r1, obs2, r2, obs3,
                                                          r3, obs4, r4, outdir, t,
                                                          tom, jerry, roadrunner, coyote, experiment_ID)

                    time_stamp_start = time.time()

                    """ recover """
                    time.sleep(2)

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
