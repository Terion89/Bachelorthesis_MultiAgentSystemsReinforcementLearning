import random
import json
import time
from datetime import datetime

import gym
from chainerrl import replay_buffer

from build.install.Python_Examples import malmoutils
from thesis import minecraft_py
from gym import spaces
import logging
import MalmoPython
import numpy as np
import os
from chainer import optimizers
from thesis.chainerrl.chainerrl import explorers
from thesis.chainerrl.chainerrl import links
from thesis.chainerrl.chainerrl import q_functions
from thesis.chainerrl.chainerrl.experiments.evaluator import save_agent
import thesis_evaluation_swarm_intelligence
from thesis import chainerrl

"""
Logging for easier Debugging
options: DEBUG, ALL, INFO
"""
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Bachelor Thesis: Multiagent Systems and Reinforcement Learning in Minecraft
Author: Alexandra Petric
Matr.Nr.: 1361271

Environment Arena 'ThesisEnvExperiment' for capture-the-flag problem
available functions:
    init()
    create_action_space()
    load_mission_file(mission_file)
    load_mission_xml(mission_xml)
    clip_action_filter(a)
    dqn_q_values_and_neuronal_net(args, action_space, obs_size, obs_space)
    step_generating(action, agent_num)
    reset_world(experiment_ID, time_stamp_start, t)
    do_action(actions, agent_num)
    get_video_frame(world_state, agent_num)
    get_observation(world_state)
    save_new_round(t)
    append_save_file_with_flag(time_step, name)
    append_save_file_with_agents_fail(text)
    append_save_file_with_finish(time_step, name)
    save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
        overall_reward_agent_coyote,time_step)
    get_cell_agents()
    get_current_cell_agents()
    get_world_state_observations(world_state)
    get_position_in_arena(world_state, time_step, agent_num)
    movenorth1_function(x, z)
    movesouth1_function(x, z)
    moveeast1_function(x, z)
    movewest1_function(x, z)
    get_new_position(action, x, z)
    approve_distance(tom, jerry, roadrunner, coyote, obs1, obs2, obs3, obs4, r1, r2, r3, r4,
        action1, action2, action3, action4, time_step)
    winner_behaviour(agent_host, time_step, name)
    check_inventory(time_step)
    sending_mission_quit_commands(overall_reward_agent_Tom, overall_reward_agent_Jerry,
    overall_reward_agent_roadrunner, overall_reward_agent_coyote, time_step, obs1, r1, obs2, r2, obs3, r3, 
        obs4, r4, outdir, t, tom, jerry, roadrunner, coyote, experiment_ID)
    save_data()
    save_data_for_evaluation_plots(t, time_step, overall_reward_agent_Tom,
        overall_reward_agent_Jerry, overall_reward_agent_roadrunner, overall_reward_agent_coyote)
"""


class ThesisEnvExperiment(gym.Env):
    """
    initialize agents and give commandline permissions
    """
    metadata = {'render.modes': ['human']}

    """ Agent 00: Skye """
    agent_host0 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host0)
    """ Agent 01: Tom """
    agent_host1 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host1)
    """ Agent 02: Jerry """
    agent_host2 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host2)
    """ Agent 03: Roadrunner """
    agent_host3 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host3)
    """ Agent 04: Coyote """
    agent_host4 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host4)

    """global variables to remember in game and for evaluation"""
    flag_captured_tom = flag_captured_jerry = flag_captured_roadrunner = flag_captured_coyote = False
    fetched_cell_tom = fetched_cell_jerry = cell_now_tom = cell_now_jerry = 0
    fetched_cell_roadrunner = fetched_cell_coyote = cell_now_roadrunner = cell_now_coyote = 0
    time_stamp_start_for_distance = 0
    too_close_counter = 0
    x1_prev = x2_prev = x3_prev = x4_prev = 0
    z1_prev = z2_prev = z3_prev = z4_prev = 0
    x1_exp = x2_exp = x3_exp = x4_exp = 0
    z1_exp = z2_exp = z3_exp = z4_exp = 0
    command_executed_tom = command_executed_jerry = command_executed_roadrunner = command_executed_coyote = False
    block_yaw_tom = block_yaw_jerry = block_yaw_roadrunner = block_yaw_coyote = 0
    time_step_tom_won = None
    time_step_jerry_won = None
    time_step_roadrunner_won = None
    time_step_coyote_won = None
    time_step_tom_captured_the_flag = None
    time_step_jerry_captured_the_flag = None
    time_step_roadrunner_captured_the_flag = None
    time_step_coyote_captured_the_flag = None
    winner_agent = "-"
    time_step_agents_ran_into_each_other = None
    steps_tom = 0
    steps_jerry = 0
    steps_roadrunner = 0
    steps_coyote = 0
    episode_counter = 0
    obs1 = obs2 = obs3 = obs4 = 0
    r1 = r2 = r3 = r4 = reward = 0
    done1 = done2 = done3 = done4 = done_team01 = done_team02 = False
    info1 = info2 = info3 = info4 = 0
    overall_reward_agent_Jerry = overall_reward_agent_Tom = 0
    overall_reward_agent_roadrunner = overall_reward_agent_coyote = 0

    """ arrays for collected data for evaluation """
    dirname = ""
    evaluation_episode_counter = []
    evaluation_too_close_counter = []
    evaluation_episode_time = []
    evaluation_flag_captured_tom = []
    evaluation_flag_captured_jerry = []
    evaluation_flag_captured_roadrunner = []
    evaluation_flag_captured_coyote = []
    evaluation_agents_ran_into_each_other = []
    evaluation_game_won_timestamp = []
    evaluation_winner_agent = []
    evaluation_reward_tom = []
    evaluation_reward_jerry = []
    evaluation_reward_roadrunner = []
    evaluation_reward_coyote = []
    evaluation_steps_tom = []
    evaluation_steps_jerry = []
    evaluation_steps_roadrunner = []
    evaluation_steps_coyote = []

    def __init__(self):
        super(ThesisEnvExperiment, self).__init__()
        """
        load the mission file
        format: XML
        """
        mission_file = 'capture_the_flag_xml_mission_DQL_swarm_intelligence.xml'
        self.load_mission_file(mission_file)
        print("Mission loaded: Capture the Flag")

        self.client_pool = None
        self.mc_process = None
        self.mission_end = False

    def init(self, client_pool=None, start_minecraft=None,
             continuous_discrete=True, add_noop_command=None,
             max_retries=150, retry_sleep=10, step_sleep=0.001, skip_steps=0,
             videoResolution=None, videoWithDepth=None,
             observeRecentCommands=None, observeHotBar=None,
             observeFullInventory=None, observeGrid=None,
             observeDistance=None, observeChat=None,
             allowDiscreteMovement=None,
             recordDestination=None,
             recordRewards=None,
             recordCommands=None, recordMP4=None,
             gameMode=None, forceWorldReset=None):

        """ initialize the environment with the standard parameters """

        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.step_sleep = step_sleep
        self.skip_steps = skip_steps
        self.forceWorldReset = forceWorldReset
        self.continuous_discrete = continuous_discrete
        self.add_noop_command = add_noop_command
        self.client_pool = client_pool

        if videoResolution:
            if videoWithDepth:
                self.mission_spec.requestVideoWithDepth(*videoResolution)
            else:
                self.mission_spec.requestVideo(*videoResolution)

        if observeRecentCommands:
            self.mission_spec.observeRecentCommands()
        if observeHotBar:
            self.mission_spec.observeHotBar()
        if observeFullInventory:
            self.mission_spec.observeFullInventory()
        if observeGrid:
            self.mission_spec.observeGrid(*(observeGrid + ["grid"]))
        if observeDistance:
            self.mission_spec.observeDistance(*(observeDistance + ["dist"]))
        if observeChat:
            self.mission_spec.observeChat()

        if allowDiscreteMovement:
            # we just use discrete movement, not continuous
            self.mission_spec.removeAllCommandHandlers()

            if allowDiscreteMovement is True:
                self.mission_spec.allowAllDiscreteMovementCommands()
            elif isinstance(allowDiscreteMovement, list):
                for cmd in allowDiscreteMovement:
                    self.mission_spec.allowDiscreteMovementCommand(cmd)

        if start_minecraft:
            # start Minecraft process assigning port dynamically
            self.mc_process, port = minecraft_py.start()
            logger.info("Started Minecraft on port %d, overriding client_pool.", port)
            client_pool = [('127.0.0.1', port)]

        """ 
        make client_pool usable for Malmo: change format of the client_pool to struct 
        """
        if client_pool:
            if not isinstance(client_pool, list):
                raise ValueError("client_pool must be list of tuples of (IP-address, port)")
            self.client_pool = MalmoPython.ClientPool()
            for client in client_pool:
                self.client_pool.add(MalmoPython.ClientInfo(*client))

        """
        initialize video parameters for video processing
        """
        self.video_height = self.mission_spec.getVideoHeight(0)
        self.video_width = self.mission_spec.getVideoWidth(0)
        self.video_depth = self.mission_spec.getVideoChannels(0)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.video_height, self.video_width, self.video_depth))
        """
        dummy image for the first observation
        """
        self.last_image1 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.last_image2 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.last_image3 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.last_image4 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.create_action_space()

        """ 
        mission recording spec
        """
        self.mission_record_spec = MalmoPython.MissionRecordSpec()  # record nothing
        if recordDestination:
            self.mission_record_spec.setDestination(recordDestination)
        if recordRewards:
            self.mission_record_spec.recordRewards()
        if recordCommands:
            self.mission_record_spec.recordCommands()
        if recordMP4:
            self.mission_record_spec.recordMP4(*recordMP4)

        """ 
        game mode, default: survival
        """
        if gameMode:
            if gameMode == "spectator":
                self.mission_spec.setModeToSpectator()
            elif gameMode == "creative":
                self.mission_spec.setModeToCreative()
            elif gameMode == "survival":
                logger.warning("Cannot force survival mode, assuming it is the default.")
            else:
                assert False, "Unknown game mode: " + gameMode

    def create_action_space(self):
        """
        create action_space from action_names to dynamically generate the needed movement
        format:             Discrete
        possible actions:   "move", "jumpmove", "strafe", "jumpstrafe", "turn", "jumpnorth", "jumpsouth", "jumpwest",
                            "jumpeast","look", "use", "jumpuse", "sleep", "movenorth", "movesouth", "moveeast",
                            "movewest", "jump", "attack"
        discrete_actions:   wanted actions
        """

        discrete_actions = []
        discrete_actions.append("movenorth 1")
        discrete_actions.append("movesouth 1")
        discrete_actions.append("movewest 1")
        discrete_actions.append("moveeast 1")
        discrete_actions.append("attack 1")
        discrete_actions.append("turn 1")
        discrete_actions.append("turn -1")

        """ turn action lists into action spaces """
        self.action_names = []
        self.action_spaces = []
        if len(discrete_actions) > 0:
            self.action_spaces.append(spaces.Discrete(len(discrete_actions)))
            self.action_names.append(discrete_actions)

        if len(self.action_spaces) == 1:
            self.action_space = self.action_spaces[0]
        else:
            self.action_space = spaces.Tuple(self.action_spaces)
        logger.debug(self.action_space)

    def load_mission_file(self, mission_file):
        """
        load XML mission from folder
        """
        logger.info("Loading mission from " + mission_file)
        mission_xml = open(mission_file, 'r').read()
        self.load_mission_xml(mission_xml)

    def load_mission_xml(self, mission_xml):
        """
        load mission file into game
        """
        self.mission_spec = MalmoPython.MissionSpec(mission_xml, True)
        logger.info("Loaded mission: " + self.mission_spec.getSummary())

    def clip_action_filter(self, a):
        return np.clip(a, self.action_space.low, self.action_space.high)

    def dqn_q_values_and_neuronal_net(self, args, action_space, obs_size, obs_space):
        """
        WIP! dqn net
        """

        if isinstance(action_space, spaces.Box):
            action_size = action_space.low.size
            # Use NAF to apply DQN to continuous action spaces
            q_func = q_functions.FCQuadraticStateQFunction(
                obs_size, action_size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
                action_space=action_space)
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (action_space.high - action_space.low) * 0.2
            explorer = explorers.AdditiveOU(sigma=ou_sigma)

        else:
            # discrete movement
            n_actions = action_space.n
            print("n_actions: ", n_actions)
            q_func = q_functions.FCStateQFunctionWithDiscreteAction(
                obs_size, n_actions,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers)
            print("q_func ", q_func)
            # Use epsilon-greedy for exploration
            explorer = explorers.LinearDecayEpsilonGreedy(
                args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
                action_space.sample)
            print("explorer: ", explorer)

        if args.noisy_net_sigma is not None:
            links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
            # Turn off explorer
            explorer = explorers.Greedy()
            # print("obs_space.low : ", obs_space.shape)
            chainerrl.misc.draw_computational_graph(
                [q_func(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
                os.path.join(args.outdir, 'model'))

        opt = optimizers.Adam()
        opt.setup(q_func)

        # Replay Buffer
        rbuf_capacity = 5 * 10 ** 5
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                        // args.update_interval
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

        return q_func, opt, rbuf, explorer

    def get_safe_worldstate(self, agent_num):

        if agent_num == 1:
            world_state1 = 0
            """ loop to minimize errors if there is a broken world_state"""
            while world_state1 == 0:
                world_state1 = self.agent_host1.peekWorldState()

            while world_state1.number_of_observations_since_last_state == 0:
                world_state1 = self.agent_host1.peekWorldState()
                time.sleep(0.1)

            return world_state1

        if agent_num == 2:
            world_state2 = 0
            """ loop to minimize errors if there is a broken world_state"""
            while world_state2 == 0:
                world_state2 = self.agent_host2.peekWorldState()

            while world_state2.number_of_observations_since_last_state == 0:
                world_state2 = self.agent_host2.peekWorldState()
                time.sleep(0.1)

            return world_state2

        if agent_num == 3:
            world_state3 = 0
            """ loop to minimize errors if there is a broken world_state"""
            while world_state3 == 0:
                world_state3 = self.agent_host3.peekWorldState()

            while world_state3.number_of_observations_since_last_state == 0:
                world_state3 = self.agent_host3.peekWorldState()
                time.sleep(0.1)

            return world_state3

        if agent_num == 4:
            world_state4 = 0
            """ loop to minimize errors if there is a broken world_state"""
            while world_state4 == 0:
                world_state4 = self.agent_host4.peekWorldState()

            while world_state4.number_of_observations_since_last_state == 0:
                world_state4 = self.agent_host4.peekWorldState()
                time.sleep(0.1)

            return world_state4

    def step_generating(self, action, agent_num):
        """
        time step in arena
        next action is executed
        reward of current state is calculated and summed up with the overall reward
        PARAMETERS: action, agent_num
        RETURN: image, reward, done, info
        """

        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        obs = 0
        done = False
        info = {}

        if agent_num == 1:
            steps = self.steps_tom
            if world_state1.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)

            """ wait until the step was made """
            while steps == self.steps_tom:
                time.sleep(0.1)

            """ fetch the new wold_state after the step """
            world_state1 = self.get_safe_worldstate(1)

            """ take the last frame from world state | 'done'-flag indicates, if mission is still running """
            obs1 = self.get_video_frame(world_state1, 1)
            self.done_team01 = not world_state1.is_mission_running

            """ collected information during the run """
            info1 = {}
            info1['has_mission_begun'] = world_state1.has_mission_begun
            info1['is_mission_running'] = world_state1.is_mission_running
            info1['number_of_video_frames_since_last_state'] = world_state1.number_of_video_frames_since_last_state
            info1['number_of_rewards_since_last_state'] = world_state1.number_of_rewards_since_last_state
            info1['number_of_observations_since_last_state'] = world_state1.number_of_observations_since_last_state
            info1['mission_control_messages'] = [msg.text for msg in world_state1.mission_control_messages]
            info1['observation'] = self.get_observation(world_state1)

            done1 = False

            """ calculate reward of current state """
            for r in world_state1.rewards:
                reward1 = r.getValue()
                # print("%i, %f" % (reward1, reward1))
                self.reward = reward1

            obs = obs1
            done = done1
            info = info1

        if agent_num == 2:
            steps = self.steps_jerry
            if world_state2.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)

            """ wait until the step was made """
            while steps == self.steps_jerry:
                time.sleep(0.1)

            """ fetch the new wold_state after the step """
            world_state2 = self.get_safe_worldstate(2)

            """ take the last frame from world state | 'done'-flag indicates, if mission is still running """
            obs2 = self.get_video_frame(world_state2, 1)
            self.done_team02 = not world_state2.is_mission_running

            """ collected information during the run """
            info2 = {}
            info2['has_mission_begun'] = world_state2.has_mission_begun
            info2['is_mission_running'] = world_state2.is_mission_running
            info2['number_of_video_frames_since_last_state'] = world_state2.number_of_video_frames_since_last_state
            info2['number_of_rewards_since_last_state'] = world_state2.number_of_rewards_since_last_state
            info2['number_of_observations_since_last_state'] = world_state2.number_of_observations_since_last_state
            info2['mission_control_messages'] = [msg.text for msg in world_state2.mission_control_messages]
            info2['observation'] = self.get_observation(world_state2)

            done2 = False

            """ calculate reward of current state """
            for r in world_state2.rewards:
                reward2 = r.getValue()
                # print("%i, %f" % (reward2, reward2))
                self.reward = reward2

            obs = obs2
            done = done2
            info = info2

        if agent_num == 3:
            steps = self.steps_roadrunner
            if world_state3.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)

            """ wait until the step was made """
            while steps == self.steps_roadrunner:
                time.sleep(0.1)

            """" wait for the new state """
            """ fetch the new wold_state after the step """
            world_state3 = self.get_safe_worldstate(3)

            """ take the last frame from world state | 'done'-flag indicates, if mission is still running """
            obs3 = self.get_video_frame(world_state3, 1)
            self.done_team01 = not world_state3.is_mission_running

            """ collected information during the run """
            info3 = {}
            info3['has_mission_begun'] = world_state3.has_mission_begun
            info3['is_mission_running'] = world_state3.is_mission_running
            info3['number_of_video_frames_since_last_state'] = world_state3.number_of_video_frames_since_last_state
            info3['number_of_rewards_since_last_state'] = world_state3.number_of_rewards_since_last_state
            info3['number_of_observations_since_last_state'] = world_state3.number_of_observations_since_last_state
            info3['mission_control_messages'] = [msg.text for msg in world_state3.mission_control_messages]
            info3['observation'] = self.get_observation(world_state3)

            done3 = False

            """ calculate reward of current state """
            for r in world_state3.rewards:
                reward3 = r.getValue()
                # print("%i, %f" % (reward3, reward3))
                self.reward = reward3

            obs = obs3
            done = done3
            info = info3

        if agent_num == 4:
            steps = self.steps_coyote
            if world_state4.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)

            """ wait until the step was made """
            while steps == self.steps_coyote:
                time.sleep(0.1)

            """ wait for the new state """
            """ fetch the new wold_state after the step """
            world_state4 = self.get_safe_worldstate(4)

            """ take the last frame from world state | 'done'-flag indicates, if mission is still running """
            obs4 = self.get_video_frame(world_state4, 1)
            self.done_team02 = not world_state4.is_mission_running

            """ collected information during the run """
            info4 = {}
            info4['has_mission_begun'] = world_state4.has_mission_begun
            info4['is_mission_running'] = world_state4.is_mission_running
            info4['number_of_video_frames_since_last_state'] = world_state4.number_of_video_frames_since_last_state
            info4['number_of_rewards_since_last_state'] = world_state4.number_of_rewards_since_last_state
            info4['number_of_observations_since_last_state'] = world_state4.number_of_observations_since_last_state
            info4['mission_control_messages'] = [msg.text for msg in world_state4.mission_control_messages]
            info4['observation'] = self.get_observation(world_state4)

            done4 = False

            """ calculate reward of current state """
            for r in world_state4.rewards:
                reward4 = r.getValue()
                # print("%i, %f" % (reward4, reward4))
                self.reward = reward4

            obs = obs4
            done = done4
            info = info4

        print("Agent: %i - reward: %i - done: %s " % (agent_num, self.reward, done))
        reward_back = self.reward
        return obs, reward_back, done, info

    def random_timer(self):
        """
        A random time-step generator is needed, because the clients must not start after the exact same time difference.
        This would lead to fails, so a random added float-number solves the starting problem.
        RETURN: random_number
        """
        random_number = random.random() * 10.0 + 1.0
        print("random number: ", random_number)
        return random_number

    def reset_world(self, experiment_ID, time_stamp_start, t):
        """
        reset the arena and start the missions per agent
        The sleep-timer of 10sec is required, because the client needs far too much time to set up the mission
        for the first time.
        All following missions start faster (sometimes).
        PARAMETERS: experiment_ID
        """

        """ Quit every unfinished open mission """
        self.quit()

        print("force world reset........")
        self.flag_captured_tom = False
        self.flag_captured_jerry = False
        self.flag_captured_roadrunner = False
        self.flag_captured_coyote = False

        # for better performance
        random_wait_time_1 = 6 + self.random_timer()
        time.sleep(random_wait_time_1)

        for retry in range(self.max_retries + 1):
            try:
                """ start missions for every client """
                random_wait_time_2 = 6 + self.random_timer()
                time.sleep(random_wait_time_2)
                print("starting mission for Skye")
                self.agent_host0.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              0, experiment_ID)
                random_wait_time_3 = 6 + self.random_timer()
                time.sleep(random_wait_time_3)
                print("starting mission for Tom")
                self.agent_host1.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              1, experiment_ID)

                random_wait_time_4 = 6 + self.random_timer()
                time.sleep(random_wait_time_4)
                print("starting mission for Jerry")
                self.agent_host2.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              2, experiment_ID)

                random_wait_time_5 = 6 + self.random_timer()
                time.sleep(random_wait_time_5)
                print("starting mission for Roadrunner")
                self.agent_host3.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              3, experiment_ID)

                random_wait_time_6 = 6 + self.random_timer()
                time.sleep(random_wait_time_6)
                print("starting mission for Coyote")
                self.agent_host4.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              4, experiment_ID)
                random_wait_time_7 = 6 + self.random_timer()
                time.sleep(random_wait_time_7)
                print("\nmissions successfully started.....\n")

                del random_wait_time_1
                del random_wait_time_2
                del random_wait_time_3
                del random_wait_time_4
                del random_wait_time_5
                del random_wait_time_6
                del random_wait_time_7

                break
            except RuntimeError as e:
                if retry == self.max_retries:
                    logger.error("Max_retries reached - error starting mission: " + str(e))
                    raise
                else:
                    logger.warning("Error starting mission: " + str(e))
                    random_wait_time_startup = self.retry_sleep + self.random_timer()
                    logger.info("Sleeping for %d seconds...", random_wait_time_startup)
                    time.sleep(random_wait_time_startup)

        logger.info("Waiting for the agents to send worldstates.")
        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        breaker = 0

        while not world_state1.has_mission_begun and not world_state2.has_mission_begun and \
                not world_state3.has_mission_begun and not world_state4.has_mission_begun:

            world_state1 = self.get_safe_worldstate(1)
            world_state2 = self.get_safe_worldstate(2)
            world_state3 = self.get_safe_worldstate(3)
            world_state4 = self.get_safe_worldstate(4)
            print("Mission hast not begun yet. Attempt: ", breaker)
            breaker += 1

            if breaker == 101:
                """ if the mission is broken from the beginning, just throw the episode away and restart """
                time_stamp_current = time.time()
                time_step = time_stamp_current - time_stamp_start
                self.time_step_agents_ran_into_each_other = int(time_step)
                self.mission_end = True
                print("The mission is broken due to insufficient client connections.")

                t, obs1, obs2, r1, r2, obs3, obs4, r3, r4, self.done_team01, \
                self.done_team02, self.overall_reward_agent_Jerry, self.overall_reward_agent_Tom, \
                self.overall_reward_agent_roadrunner, self.overall_reward_agent_coyote = \
                    self.quit_after_crash(t, experiment_ID)
                break

            for error in world_state1.errors and world_state2.errors and world_state3.errors and world_state4.errors:
                logger.warning(error.text)

        logger.info("Mission running.")

        return self.get_video_frame(world_state1, 1), self.get_video_frame(world_state2, 2), \
               self.get_video_frame(world_state3, 3), self.get_video_frame(world_state4, 4)

    def do_action(self, actions, agent_num):
        """
        get next action from action_space
        execute action in environment for the agent
        PARAMETERS: actions, agent_num
        """
        if len(self.action_spaces) == 1:
            actions = [actions]

        for spc, cmds, acts in zip(self.action_spaces, self.action_names, actions):
            if isinstance(spc, spaces.Discrete):
                logger.debug(cmds[acts])
                if agent_num == 1:
                    print("Tom's next action: ", cmds[acts])
                    self.agent_host1.sendCommand(cmds[acts])
                if agent_num == 2:
                    print("Jerry's next action: ", cmds[acts])
                    self.agent_host2.sendCommand(cmds[acts])
                if agent_num == 3:
                    print("Roadrunner's next action: ", cmds[acts])
                    self.agent_host3.sendCommand(cmds[acts])
                if agent_num == 4:
                    print("Coyote's next action: ", cmds[acts])
                    self.agent_host4.sendCommand(cmds[acts])
            elif isinstance(spc, spaces.Box):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    if agent_num == 1:
                        self.agent_host1.sendCommand(cmd + " " + str(val))
                    if agent_num == 2:
                        self.agent_host2.sendCommand(cmd + " " + str(val))
                    if agent_num == 3:
                        self.agent_host3.sendCommand(cmd + " " + str(val))
                    if agent_num == 4:
                        self.agent_host4.sendCommand(cmd + " " + str(val))
            elif isinstance(spc, spaces.MultiDiscrete):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    if agent_num == 1:
                        self.agent_host1.sendCommand(cmd + " " + str(val))
                    if agent_num == 2:
                        self.agent_host2.sendCommand(cmd + " " + str(val))
                    if agent_num == 3:
                        self.agent_host3.sendCommand(cmd + " " + str(val))
                    if agent_num == 4:
                        self.agent_host4.sendCommand(cmd + " " + str(val))
            else:
                logger.warning("Unknown action space for %s, ignoring." % cmds)

        """ count the steps for the individual agent """
        if agent_num == 1:
            self.steps_tom += 1
        if agent_num == 2:
            self.steps_jerry += 1
        if agent_num == 3:
            self.steps_roadrunner += 1
        if agent_num == 4:
            self.steps_coyote += 1

    def get_video_frame(self, world_state, agent_num):
        """
        process video frame for called agent
        PARAMETERS: world_state, agent_num
        RETURN: image for called agent
        """
        image = 0
        if world_state.number_of_video_frames_since_last_state > 0:
            assert len(world_state.video_frames) == 1
            frame = world_state.video_frames[0]
            reshaped = np.zeros((self.video_height * self.video_width * self.video_depth), dtype=np.float32)
            image = np.frombuffer(frame.pixels, dtype=np.int8)
            for i in range(frame.height * frame.width * frame.channels):
                reshaped[i] = image[i]

            # image = np.frombuffer(frame.pixels, dtype=np.float32) #300x400 = 120000 Werte // np.float32
            image = reshaped.reshape((frame.height, frame.width, frame.channels))  # 300x400x3 = 360000

            if agent_num == 1:
                self.last_image1 = image
            if agent_num == 2:
                self.last_image2 = image
            if agent_num == 3:
                self.last_image3 = image
            if agent_num == 4:
                self.last_image4 = image
        else:
            """ if mission ends before we got a frame, just take the last frame to reduce exceptions """
            if agent_num == 1:
                image = self.last_image1
            if agent_num == 2:
                image = self.last_image2
            if agent_num == 3:
                image = self.last_image3
            if agent_num == 4:
                image = self.last_image4

        return image

    def get_observation(self, world_state):
        """
        check observations during mission run
        PARAMETERS: world_state
        RETURN: number of missed observations - if there are any
        """
        if world_state.number_of_observations_since_last_state > 0:
            missed = world_state.number_of_observations_since_last_state - len(
                world_state.observations) - self.skip_steps
            if missed > 0:
                logger.warning("Agent missed %d observation(s).", missed)
            assert len(world_state.observations) == 1
            return json.loads(world_state.observations[0].text)
        else:
            return None

    def save_new_round(self, t):
        """
        saves the round number in results.txt
        """
        datei = open(self.dirname + "/" + 'results.txt', 'a')
        datei.write("\n-------------- ROUND %i --------------\n" % t)
        time_start = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        datei.write("starts at: %s" % time_start)
        datei.close()

    def append_save_file_with_flag(self, time_step, name):
        """
        saves the flagholder in results.txt
        """
        datei = open(self.dirname + "/" + 'results.txt', 'a')
        datei.write("%s captured the flag after %i seconds.\n" % (name, time_step))
        datei.close()

    def append_save_file_with_agents_fail(self, text):
        """
        saves the explicit agent-failes in results.txt
        """
        datei = open(self.dirname + "/" + 'results.txt', 'a')
        datei.write(text + "\n")
        datei.close()

    def append_save_file_with_finish(self, time_step, name):
        """
        saves the winner in results.txt
        """
        datei = open(self.dirname + "/" + 'results.txt', 'a')
        datei.write("%s won the game after %i seconds.\n" % (name, time_step))
        datei.close()

    def save_results(self, overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
                     overall_reward_agent_coyote, time_step):
        """
        saves the results in results.txt
        """
        datei = open(self.dirname + "/" + 'results.txt', 'a')
        if self.too_close_counter == 1:
            datei.write("The agents were 1 time very close to each other.\n")

        else:
            datei.write("The agents were %i times very close to each other.\n" % self.too_close_counter)

        datei.write("Reward Tom: %i, Reward Jerry: %i , Reward Roadrunner: %i, Reward Coyote: %i , Time: %f \n\n" % (
            overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
            overall_reward_agent_coyote, time_step))
        datei.close()

    def get_cell_agents(self):
        """
        gets the cell coordinates for the agents to compare with
        """
        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        ob1 = self.get_worldstate_observations(world_state1)
        ob2 = self.get_worldstate_observations(world_state2)
        ob3 = self.get_worldstate_observations(world_state3)
        ob4 = self.get_worldstate_observations(world_state4)

        if "cell" in ob1 and "cell" in ob2 and "cell" in ob3 and "cell" in ob4:
            self.fetched_cell_tom = ob1.get(u'cell', 0)
            self.fetched_cell_jerry = ob2.get(u'cell', 0)
            self.fetched_cell_roadrunner = ob3.get(u'cell', 0)
            self.fetched_cell_coyote = ob4.get(u'cell', 0)
            print("fetched cell tom: ", self.fetched_cell_tom)
            print("fetched cell jerry: ", self.fetched_cell_jerry)
            print("fetched cell roadrunner: ", self.fetched_cell_roadrunner)
            print("fetched cell coyote: ", self.fetched_cell_coyote)

    def get_current_cell_agents(self):
        """
        gets the cell coordinates for the agents at a state
        same as get_cell_agents ---> redundant, need to clear this
        """
        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        ob1 = self.get_worldstate_observations(world_state1)
        ob2 = self.get_worldstate_observations(world_state2)
        ob3 = self.get_worldstate_observations(world_state3)
        ob4 = self.get_worldstate_observations(world_state4)

        if "cell" in ob1 and "cell" in ob2 and "cell" in ob3 and "cell" in ob4:
            self.cell_now_tom = ob1.get(u'cell', 0)
            self.cell_now_jerry = ob2.get(u'cell', 0)
            self.cell_now_roadrunner = ob3.get(u'cell', 0)
            self.cell_now_coyote = ob4.get(u'cell', 0)
            print("current cell tom: ", self.cell_now_tom)
            print("current cell jerry: ", self.cell_now_jerry)
            print("current cell roadrunner: ", self.cell_now_roadrunner)
            print("current cell coyote: ", self.cell_now_coyote)

    def get_worldstate_observations(self, world_state):
        """
        get current world_state observations of called agent
        """
        msg = world_state.observations[-1].text
        ob = json.loads(msg)
        return ob

    def get_position_in_arena(self, world_state, time_step, agent_num):
        """
        get (x,y,z) Positioncoordinates of agent
        fetch the cell coordinates every 30 seconds
        check with current coordinates -> if they are the same more than 28 seconds, it is nearly safe, that the agents
        crashed into each other -> declare mission as failed and end it
        PARAMETERS: world_state, time_step, agent_num
        RETURN: x,y,z
        """

        x = y = z = t = 0

        world_state = self.get_safe_worldstate(agent_num)

        while world_state:
            if len(world_state.observations) >= 1:
                ob = self.get_worldstate_observations(world_state)
                time_now = time.time()

                if time_now - self.time_stamp_start_for_distance > 120:
                    """ fetch cell every 120 seconds """
                    self.get_cell_agents()
                    self.time_stamp_start_for_distance = time.time()

                if "XPos" in ob and "ZPos" in ob and "YPos" in ob:
                    x = ob[u'XPos']
                    y = ob[u'YPos']
                    z = ob[u'ZPos']

                seconds = time_now - self.time_stamp_start_for_distance
                # print("seconds: ", int(seconds))

                if int(seconds) == 110:
                    self.get_current_cell_agents()
                    if self.fetched_cell_tom == self.cell_now_tom and self.fetched_cell_jerry == self.cell_now_jerry \
                            and self.fetched_cell_roadrunner == self.cell_now_roadrunner and \
                            self.fetched_cell_coyote == self.cell_now_coyote:
                        print("The agents stood more than 110 seconds in one place.")
                        # self.time_step_agents_ran_into_each_other = int(time_step)
                        self.append_save_file_with_agents_fail("The agents stood more than 110 seconds in one place.")
                        # self.mission_end = True

                return x, y, z
            else:
                if t == 10:
                    self.append_save_file_with_agents_fail(
                        "The 10th attempt failed - the agents got stuck in one place.")
                    self.time_step_agents_ran_into_each_other = int(time_step)
                    self.mission_end = True
                    return x, y, z
                else:
                    time.sleep(0.1)
                    t += 1
                    world_state = self.get_safe_worldstate(agent_num)
                    print("Agents are stuck - Attempt: ", t)
                    self.append_save_file_with_agents_fail("The agents are stuck - Attempt: " + t)

    def movenorth1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step north """
        x = int(x)
        z = int(z)
        x_ziel = x

        if (z == 2 and 15 <= x <= 16) or z == 0:
            z_ziel = z
            return x_ziel, z_ziel
        else:
            z_ziel = z - 1
            return x_ziel, z_ziel

    def movesouth1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step south """
        x = int(x)
        z = int(z)
        x_ziel = x

        if (z == 13 and 0 <= x <= 2) or z == 15:
            z_ziel = z
            return x_ziel, z_ziel
        else:
            z_ziel = z + 1
            return x_ziel, z_ziel

    def moveeast1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step east """
        x = int(x)
        z = int(z)
        z_ziel = z

        if (x == 13 and 0 <= z <= 1) or x == 15:
            x_ziel = x
            return x_ziel, z_ziel
        else:
            x_ziel = x + 1
            return x_ziel, z_ziel

    def movewest1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step west """
        x = int(x)
        z = int(z)
        z_ziel = z

        if (x == 2 and 14 <= z <= 15) or x == 0:
            x_ziel = x
            return x_ziel, z_ziel
        else:
            x_ziel = x - 1
            return x_ziel, z_ziel

    def get_new_position(self, action, x, z):
        """ calculates the new position of the agent """

        action_name = ""

        for spc, cmds in zip(self.action_spaces, self.action_names):
            if isinstance(spc, spaces.Discrete):
                action_name = cmds[action]

        if action_name == "movenorth 1":
            x_new, z_new = self.movenorth1_function(x, z)
            block_yaw = 180.0
        elif action_name == "movesouth 1":
            x_new, z_new = self.movesouth1_function(x, z)
            block_yaw = 0.0
        elif action_name == "moveeast 1":
            x_new, z_new = self.moveeast1_function(x, z)
            block_yaw = 270.0
        elif action_name == "movewest 1":
            x_new, z_new = self.movewest1_function(x, z)
            block_yaw = 90.0
        else:  # turn 1, turn -1, attack 1
            x_new = x
            z_new = z
            block_yaw = 0

        return x_new, z_new, block_yaw

    def approve_distance(self, tom, jerry, roadrunner, coyote, obs1, obs2, obs3, obs4, r1, r2, r3, r4,
                         action1, action2, action3, action4, time_step, t, experiment_ID):
        """
        check if agents are too close to eachother
        if so, the next actions are calculated new
        PARAMETERS: tom, jerry, roadrunner, coyote, obs1, obs2, obs3, obs4, r1, r2, r3, r4,
                         action1, action2, action3, action4, time_step, t, experiment_ID
        RETURN: action1, action2, action3, action4
        """
        steps_approved = needed_new_calculation = False
        # world_state1 = world_state2 = world_state3 = world_state4 = 0

        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        """ if the mission doesn't get any worldstates in a minute, throw the episode away 
        because it is broken anyway """
        """if breaker == 121:
            self.time_step_agents_ran_into_each_other = int(time_step)
            self.mission_end = True
            t, obs1, obs2, r1, r2, obs3, obs4, r3, r4, self.done_team01, \
            self.done_team02, self.overall_reward_agent_Jerry, self.overall_reward_agent_Tom, \
            self.overall_reward_agent_roadrunner, self.overall_reward_agent_coyote = \
                self.quit_after_crash(t, experiment_ID)
            break
        time.sleep(0.1)"""

        self.x1_prev, y1, self.z1_prev = self.get_position_in_arena(world_state1, time_step, 1)
        self.x2_prev, y2, self.z2_prev = self.get_position_in_arena(world_state2, time_step, 2)
        self.x3_prev, y3, self.z3_prev = self.get_position_in_arena(world_state3, time_step, 3)
        self.x4_prev, y4, self.z4_prev = self.get_position_in_arena(world_state4, time_step, 4)

        time.sleep(0.1)
        while not steps_approved:
            """new position for agent tom"""
            self.x1_exp, self.z1_exp, self.block_yaw_tom = self.get_new_position(action1, self.x1_prev, self.z1_prev)

            """new position for agent jerry"""
            self.x2_exp, self.z2_exp, self.block_yaw_jerry = self.get_new_position(action2, self.x2_prev, self.z2_prev)

            """new position for agent roadrunner"""
            self.x3_exp, self.z3_exp, self.block_yaw_roadrunner = self.get_new_position(action3, self.x3_prev,
                                                                                        self.z3_prev)

            """new position for agent coyote"""
            self.x4_exp, self.z4_exp, self.block_yaw_coyote = self.get_new_position(action4, self.x4_prev, self.z4_prev)

            """
            checks, if the agents would run into each other if they took the step
            first block: checks for new positions
            second block: checks for old positions, 
                because the steps are not perfectly synchronous
            """
            # WIP !!!
            # time.sleep(0.5)
            if (int(self.x1_exp) == int(self.x2_exp) and int(self.z1_exp) == int(self.z2_exp)) or \
                    (int(self.x1_exp) == int(self.x3_exp) and int(self.z1_exp) == int(self.z3_exp)) or \
                    (int(self.x1_exp) == int(self.x4_exp) and int(self.z1_exp) == int(self.z4_exp)) or \
                    (int(self.x2_exp) == int(self.x3_exp) and int(self.z2_exp) == int(self.z3_exp)) or \
                    (int(self.x2_exp) == int(self.x4_exp) and int(self.z2_exp) == int(self.z4_exp)) or \
                    (int(self.x1_exp) == int(self.x2_prev) and int(self.z1_exp) == int(self.z2_prev)) or \
                    (int(self.x1_exp) == int(self.x3_prev) and int(self.z1_exp) == int(self.z3_prev)) or \
                    (int(self.x1_exp) == int(self.x4_prev) and int(self.z1_exp) == int(self.z4_prev)) or \
                    (int(self.x2_exp) == int(self.x1_prev) and int(self.z2_exp) == int(self.z1_prev)) or \
                    (int(self.x2_exp) == int(self.x3_prev) and int(self.z2_exp) == int(self.z3_prev)) or \
                    (int(self.x2_exp) == int(self.x4_prev) and int(self.z2_exp) == int(self.z4_prev)) or \
                    (int(self.x3_exp) == int(self.x1_prev) and int(self.z3_exp) == int(self.z1_prev)) or \
                    (int(self.x3_exp) == int(self.x2_prev) and int(self.z3_exp) == int(self.z2_prev)) or \
                    (int(self.x3_exp) == int(self.x4_prev) and int(self.z3_exp) == int(self.z4_prev)) or \
                    (int(self.x4_exp) == int(self.x1_prev) and int(self.z4_exp) == int(self.z1_prev)) or \
                    (int(self.x4_exp) == int(self.x2_prev) and int(self.z4_exp) == int(self.z2_prev)) or \
                    (int(self.x4_exp) == int(self.x3_prev) and int(self.z4_exp) == int(self.z3_prev)):

                needed_new_calculation = True

                world_state1 = self.get_safe_worldstate(1)
                world_state2 = self.get_safe_worldstate(2)
                world_state3 = self.get_safe_worldstate(3)
                world_state4 = self.get_safe_worldstate(4)

                if (len(world_state1.observations) >= 1 and len(world_state2.observations) >= 1 and
                        len(world_state3.observations) >= 1 and len(world_state3.observations) >= 1):
                    obs1 = self.get_video_frame(world_state1, 1)
                    obs2 = self.get_video_frame(world_state2, 2)
                    obs3 = self.get_video_frame(world_state3, 3)
                    obs4 = self.get_video_frame(world_state4, 4)

                if (len(world_state1.observations) >= 1 and len(world_state2.observations) >= 1 and
                        len(world_state3.observations) >= 1 and len(world_state3.observations) >= 1):
                    action1 = tom.act_and_train(obs1, r1)
                    action2 = jerry.act_and_train(obs2, r2)
                    action3 = roadrunner.act_and_train(obs3, r3)
                    action4 = coyote.act_and_train(obs4, r4)

            else:
                steps_approved = True
                if needed_new_calculation:
                    self.too_close_counter += 1

        print("calculated actions: %s, %s, %s, %s" % (action1, action2, action3, action4))
        return action1, action2, action3, action4

    def check_if_command_was_executed(self, agent_num, time_step):
        """
        checks, if the agent received and executed the calculated command, then returns True,
        if not, then returns False,
        executes it again in the main script and checks again until it was executed sucessfully.
        To check, if the command was executed correctly, it compares the current position with the expected one.
        PARAMETERS: agent_num, time_step
        RETURN: True/ False
        """
        waiting_for_execution = True
        t = 0
        while waiting_for_execution:
            if agent_num == 1:
                world_state1 = self.get_safe_worldstate(1)

                x1_current, y1, z1_current = self.get_position_in_arena(world_state1, time_step, 1)
                print("Tom's current (x,z): (%i, %i) | expected (x,z): (%i, %i) " % (
                    x1_current, z1_current, self.x1_exp, self.z1_exp))
                if int(x1_current) == 2 and int(z1_current) == 10 or int(x1_current) == 2 and int(z1_current) == 9 or \
                    int(x1_current) == 1 and int(z1_current) == 8 or int(x1_current) == 0 and int(z1_current) == 8 or \
                    int(x1_current) == 8 and int(z1_current) == 1 or int(x1_current) == 8 and int(z1_current) == 0 or \
                    int(x1_current) == 10 and int(z1_current) == 2 or int(x1_current) == 9 and int(z1_current) == 2 or \
                    int(self.x1_exp) == int(x1_current) and int(self.z1_exp) == int(z1_current):

                    print("Tom executed sucessfully.")
                    return True
                else:
                    print("Tom has not executed the command by now.")
                    t += 1
                    time.sleep(0.1)
                    if t == 5:
                        executed_anyway = self.check_if_something_blocks_the_way(1)
                        if executed_anyway:
                            return True
                    print("Tom's try Nr. ", t)
                    if t == 6:
                        print("Tom needs to resend the command.")
                        return False

            elif agent_num == 2:
                world_state2 = self.get_safe_worldstate(2)

                x2_current, y2, z2_current = self.get_position_in_arena(world_state2, time_step, 2)
                print("Jerry's current (x,z): (%i, %i) | expected (x,z): (%i, %i) " % (
                    x2_current, z2_current, self.x2_exp, self.z2_exp))
                if int(x2_current) == 2 and int(z2_current) == 10 or int(x2_current) == 2 and int(z2_current) == 9 or \
                    int(x2_current) == 1 and int(z2_current) == 8 or int(x2_current) == 0 and int(z2_current) == 8 or \
                    int(x2_current) == 8 and int(z2_current) == 1 or int(x2_current) == 8 and int(z2_current) == 0 or \
                    int(x2_current) == 10 and int(z2_current) == 2 or int(x2_current) == 9 and int(z2_current) == 2 or \
                    int(self.x2_exp) == int(x2_current) and int(self.z2_exp) == int(z2_current):
                    print("Jerry executed sucessfully.")
                    return True
                else:
                    print("Jerry has not executed the command by now.")
                    t += 1
                    time.sleep(0.1)
                    if t == 5:
                        executed_anyway = self.check_if_something_blocks_the_way(2)
                        if executed_anyway:
                            return True
                    print("Jerry's try Nr. ", t)
                    if t == 6:
                        print("Jerry needs to resend the command.")
                        return False

            elif agent_num == 3:
                world_state3 = self.get_safe_worldstate(3)

                x3_current, y3, z3_current = self.get_position_in_arena(world_state3, time_step, 3)
                print("Roadrunner's current (x,z): (%i, %i) | expected (x,z): (%i, %i) " % (
                    x3_current, z3_current, self.x3_exp, self.z3_exp))
                if int(x3_current) == 2 and int(z3_current) == 10 or int(x3_current) == 2 and int(z3_current) == 9 or \
                    int(x3_current) == 1 and int(z3_current) == 8 or int(x3_current) == 0 and int(z3_current) == 8 or \
                    int(x3_current) == 8 and int(z3_current) == 1 or int(x3_current) == 8 and int(z3_current) == 0 or \
                    int(x3_current) == 10 and int(z3_current) == 2 or int(x3_current) == 9 and int(z3_current) == 2 or \
                    int(self.x3_exp) == int(x3_current) and int(self.z3_exp) == int(z3_current):
                    print("Roadrunner executed sucessfully.")
                    return True
                else:
                    print("Roadrunner has not executed the command by now.")
                    t += 1
                    time.sleep(0.1)
                    if t == 5:
                        executed_anyway = self.check_if_something_blocks_the_way(3)
                        if executed_anyway:
                            return True
                    print("Roadrunner's try Nr. ", t)
                    if t == 6:
                        print("Roadrunner needs to resend the command.")
                        return False

            elif agent_num == 4:
                world_state4 = self.get_safe_worldstate(4)

                x4_current, y4, z4_current = self.get_position_in_arena(world_state4, time_step, 4)
                print("Coyote's current (x,z): (%i, %i) | expected (x,z): (%i, %i) " % (
                    x4_current, z4_current, self.x4_exp, self.z4_exp))
                if int(x4_current) == 2 and int(z4_current) == 10 or int(x4_current) == 2 and int(z4_current) == 9 or \
                    int(x4_current) == 1 and int(z4_current) == 8 or int(x4_current) == 0 and int(z4_current) == 8 or \
                    int(x4_current) == 8 and int(z4_current) == 1 or int(x4_current) == 8 and int(z4_current) == 0 or \
                    int(x4_current) == 10 and int(z4_current) == 2 or int(x4_current) == 9 and int(z4_current) == 2 or \
                    int(self.x4_exp) == int(x4_current) and int(self.z4_exp) == int(z4_current):
                    print("Coyote executed sucessfully.")
                    return True
                else:
                    print("Coyote has not executed the command by now.")
                    t += 1
                    time.sleep(0.1)
                    if t == 5:
                        executed_anyway = self.check_if_something_blocks_the_way(4)
                        if executed_anyway:
                            return True
                    print("Coyote's try Nr. ", t)
                    if t == 6:
                        print("Coyote needs to resend the command.")
                        return False

    def check_if_agent_looks_straight(self, agent_num):
        """
        checks for the given agent number, if the agent looks straight, or if he is looking down or up.
        then corrects it by sending the needed command
        RETURNS: 'True', when the agent looks straight again, otherwise 'False'
        """
        looks_straight = False
        world_state = 0
        counter = 0

        while not looks_straight:
            if counter > 2:
                time.sleep(2)

            world_state = self.get_safe_worldstate(agent_num)
            ob = self.get_worldstate_observations(world_state)

            if agent_num == 1:
                time.sleep(0.1)
                if counter > 2:
                    time.sleep(2)
                if "Pitch" in ob:
                    pitch = ob[u'Pitch']
                    if pitch != 0:
                        self.agent_host1.sendCommand('look -1')
                        print("Agent Tom is looking around. He needs to focus again. Pitch: ", pitch)
                        counter += 1
                    else:
                        looks_straight = True

            if agent_num == 2:
                time.sleep(0.1)
                if counter > 2:
                    time.sleep(2)
                if "Pitch" in ob:
                    pitch = ob[u'Pitch']
                    if pitch != 0:
                        self.agent_host2.sendCommand('look -1')
                        print("Agent Jerry is looking around. He needs to focus again. Pitch: ", pitch)
                        counter += 1
                    else:
                        looks_straight = True

            if agent_num == 3:
                time.sleep(0.1)
                if counter > 2:
                    time.sleep(2)
                if "Pitch" in ob:
                    pitch = ob[u'Pitch']
                    if pitch != 0:
                        self.agent_host4.sendCommand('look -1')
                        print("Agent Roadrunner is looking around. He needs to focus again. Pitch: ", pitch)
                        counter += 1
                    else:
                        looks_straight = True

            if agent_num == 4:
                time.sleep(0.1)
                if counter > 2:
                    time.sleep(2)
                if "Pitch" in ob:
                    pitch = ob[u'Pitch']
                    if pitch != 0:
                        self.agent_host4.sendCommand('look -1')
                        print("Agent Coyote is looking around. He needs to focus again. Pitch: ", pitch)
                        counter += 1
                    else:
                        looks_straight = True

        return looks_straight

    def check_if_something_blocks_the_way(self, agent_num):
        """ turns the agent so that he looks toward a possible limiting block
        if there really is a block, approve, that the expected position is the current one,
        because he can not move through the block
        gets world_state at normal sight // and at looking at his feet,
        // because a one-block-high-maze or the empty baseblocks could be there"""
        # WIP!!!
        block_yaw = 1
        right_direction = False

        while not right_direction:
            world_state = 0
            # world_state_at_feet = 0
            looks_straight = False

            while world_state == 0:
                if agent_num == 1:
                    world_state = self.get_safe_worldstate(1)

                    # self.agent_host1.sendCommand('look 1')
                    # time.sleep(0.2)
                    # while world_state_at_feet == 0 or world_state_at_feet.number_of_observations_since_last_state == 0 \
                    #         or world_state == world_state_at_feet:
                    #     world_state_at_feet = self.agent_host1.peekWorldState()
                    #     time.sleep(0.5)

                    block_yaw = self.block_yaw_tom
                    # self.agent_host1.sendCommand('look -1')

                    """ loop, until the agent looked up again to prevent accidents """
                    # while looks_straight == False or world_state.number_of_observations_since_last_state == \
                    #         world_state_at_feet.number_of_observations_since_last_state:
                    #     world_state = self.agent_host1.peekWorldState()
                    #     looks_straight = self.check_if_agent_looks_straight(1)

                    print("block_yaw_tom: ", block_yaw)

                if agent_num == 2:
                    world_state = self.get_safe_worldstate(2)

                    # self.agent_host2.sendCommand('look 1')

                    # while world_state_at_feet == 0 or world_state_at_feet.number_of_observations_since_last_state == 0 \
                    #         or world_state == world_state_at_feet:
                    #     world_state_at_feet = self.agent_host2.peekWorldState()
                    #     time.sleep(0.5)

                    block_yaw = self.block_yaw_jerry
                    # self.agent_host2.sendCommand('look -1')

                    """ loop, until the agent looked up again to prevent accidents """
                    # while looks_straight == False or world_state.number_of_observations_since_last_state == \
                    #         world_state_at_feet.number_of_observations_since_last_state:
                    #     world_state = self.agent_host2.peekWorldState()
                    #     looks_straight = self.check_if_agent_looks_straight(2)

                    print("block_yaw_jerry: ", block_yaw)

                if agent_num == 3:
                    world_state = self.get_safe_worldstate(3)

                    # self.agent_host3.sendCommand('look 1')

                    # while world_state_at_feet == 0 or world_state_at_feet.number_of_observations_since_last_state == 0 \
                    #         or world_state == world_state_at_feet:
                    #     world_state_at_feet = self.agent_host3.peekWorldState()
                    #     time.sleep(0.5)

                    block_yaw = self.block_yaw_roadrunner
                    # self.agent_host3.sendCommand('look -1')

                    """ loop, until the agent looked up again to prevent accidents """
                    # while looks_straight == False or world_state.number_of_observations_since_last_state == \
                    #         world_state_at_feet.number_of_observations_since_last_state:
                    #     world_state = self.agent_host3.peekWorldState()
                    #     looks_straight = self.check_if_agent_looks_straight(3)

                    print("block_yaw_roadrunner: ", block_yaw)
                if agent_num == 4:
                    world_state = self.get_safe_worldstate(4)

                    # self.agent_host4.sendCommand('look 1')

                    # while world_state_at_feet == 0 or world_state_at_feet.number_of_observations_since_last_state == 0 \
                    #         or world_state == world_state_at_feet:
                    #     world_state_at_feet = self.agent_host4.peekWorldState()
                    #     time.sleep(0.5)

                    block_yaw = self.block_yaw_coyote
                    # self.agent_host4.sendCommand('look -1')

                    """ loop, until the agent looked up again to prevent accidents """
                    # while looks_straight == False or world_state.number_of_observations_since_last_state == \
                    #         world_state_at_feet.number_of_observations_since_last_state:
                    #     world_state = self.agent_host4.peekWorldState()
                    #     looks_straight = self.check_if_agent_looks_straight(4)

                    print("block_yaw_coyote: ", block_yaw)

            if len(world_state.observations):  # and len(world_state_at_feet.observations) >= 1:
                ob = self.get_worldstate_observations(world_state)
                # ob_at_feet = self.get_worldstate_observations(world_state_at_feet)
                # print("ob: ", ob)
                # print("ob_at_feet: ", ob_at_feet)

                if "Yaw" in ob:  # and "Yaw" in ob_at_feet:
                    yaw = ob[u'Yaw']
                    # yaw_at_feet = ob_at_feet[u'Yaw']
                    print("yaw: ", yaw)
                    # print("yaw at feet: ", yaw_at_feet)
                    if int(yaw) == int(block_yaw):  # or int(yaw_at_feet) == int(block_yaw):
                        right_direction = True
                        if len(world_state.observations):  # or len(world_state_at_feet.observations) >= 1:
                            ob = self.get_worldstate_observations(world_state)
                            # ob_at_feet = self.get_worldstate_observations(world_state_at_feet)

                            if "LineOfSight" in ob:  # and "LineOfSight" in ob_at_feet:
                                lineofsight = ob[u'LineOfSight']
                                # lineofsight_at_feet = ob_at_feet[u'LineOfSight']
                                if "hitType" in lineofsight and "distance" in lineofsight:  # and \
                                    #    "hitType" in lineofsight_at_feet and "distance" in lineofsight_at_feet:
                                    hitType = lineofsight[u'hitType']
                                    # hitType_at_feet = lineofsight_at_feet[u'hitType']
                                    distance = lineofsight[u'distance']
                                    # distance_at_feet = lineofsight_at_feet[u'distance']
                                    if hitType == "block" and distance <= 1.5:  # or \
                                        #    hitType_at_feet == "block" and distance_at_feet <= 1.5:
                                        print("hitType: %s, distance: %f" % (hitType, distance))
                                        # print(
                                        #     "(at feet) hitType: %s, distance: %f" % (hitType_at_feet, distance_at_feet))
                                        return True
                                    else:
                                        return False
                    else:
                        if agent_num == 1:
                            self.agent_host1.sendCommand('turn 1')
                        if agent_num == 2:
                            self.agent_host2.sendCommand('turn 1')
                        if agent_num == 3:
                            self.agent_host3.sendCommand('turn 1')
                        if agent_num == 4:
                            self.agent_host4.sendCommand('turn 1')

    def winner_behaviour(self, agent_host, time_step, name):
        """
        hardcoded winner-behaviour to end the mission:
        look down, place flag, look up again and jump on flag
        """
        agent_host.sendCommand('chat I won the game!')
        self.append_save_file_with_finish(time_step, name)
        self.time_step_tom_won = time_step
        self.winner_agent = name

        agent_host.sendCommand('look 1')
        time.sleep(1)
        agent_host.sendCommand('use 1')
        time.sleep(1)
        agent_host.sendCommand('look -1')
        time.sleep(1)
        agent_host.sendCommand('jumpmove 1')
        time.sleep(1)

    def check_inventory(self, time_step, t, experiment_ID):
        """
        checks, if the agent got the flag in his inventory
        """
        world_state1 = self.get_safe_worldstate(1)
        world_state2 = self.get_safe_worldstate(2)
        world_state3 = self.get_safe_worldstate(3)
        world_state4 = self.get_safe_worldstate(4)

        x1 = y1 = z1 = x2 = y2 = z2 = x3 = y3 = z3 = x4 = y4 = z4 = 0
        breaker = 0

        obs1 = self.get_worldstate_observations(world_state1)
        obs2 = self.get_worldstate_observations(world_state2)
        obs3 = self.get_worldstate_observations(world_state3)
        obs4 = self.get_worldstate_observations(world_state4)

        """ checks, if position is calculated correctly, if not, trys again """
        while (y1 == 0) or (y2 == 0) or (y3 == 0) or (y4 == 0):
            x1, y1, z1 = self.get_position_in_arena(world_state1, time_step, 1)
            x2, y2, z2 = self.get_position_in_arena(world_state2, time_step, 2)
            x3, y3, z3 = self.get_position_in_arena(world_state3, time_step, 3)
            x4, y4, z4 = self.get_position_in_arena(world_state4, time_step, 4)
            print("..")

        """ fetch the current cells """
        self.get_current_cell_agents()

        if (self.flag_captured_tom and (7 <= x1 <= 10 and 0 <= z1 <= 4)) or self.flag_captured_roadrunner and \
                (7 <= x3 <= 10 and 0 <= z3 <= 4):
            """ 
            if agent reached the target area:
            look down, set block, jump on it to reach wanted position and win the game 
            """
            if self.flag_captured_tom:
                self.winner_behaviour(self.agent_host1, time_step, name="Tom")
            if self.flag_captured_roadrunner:
                self.winner_behaviour(self.agent_host3, time_step, name="Roadrunner")

            self.mission_end = True
        else:
            if self.flag_captured_tom or self.flag_captured_roadrunner:
                print("[INFO] Team Tom and Roadrunner holds the flag.")
            else:

                last_inventory_tom = obs1[u'inventory']
                last_inventory_roadrunner = obs3[u'inventory']
                inventory_string_tom = json.dumps(last_inventory_tom)
                inventory_string_roadrunner = json.dumps(last_inventory_roadrunner)
                if inventory_string_tom.find('quartz') != -1:
                    """ swaps quartz with log, to place back quartz """
                    if json.dumps(inventory_string_tom.find('log') != -1):
                        self.agent_host1.sendCommand('swapInventoryItems 0 1')
                        # self.agent_host1.sendCommand("chat Wrong flag, I'll put it back!")
                    time.sleep(0.1)

                    """ as long as there is a false flag in the inventory, place it back """
                    counter = 0
                    while inventory_string_tom.find('quartz') != -1:
                        print("%i : %s " % (counter, json.dumps(inventory_string_tom.find('quartz')) != -1))
                        time.sleep(1)
                        self.agent_host1.sendCommand('use 1')
                        self.agent_host1.sendCommand('move 1')
                        counter += 1
                        if counter == (5 or 6 or 10 or 12 or 14 or 20):
                            self.agent_host1.sendCommand('move -1')
                        time.sleep(1)
                        world_state1 = self.get_safe_worldstate(1)
                        obs1 = self.get_worldstate_observations(world_state1)
                        last_inventory_tom = obs1[u'inventory']
                        inventory_string_tom = json.dumps(last_inventory_tom)
                        breaker += 1
                        time.sleep(0.5)
                        if breaker == 120:
                            self.quit_after_crash(t, experiment_ID)

                    self.agent_host1.sendCommand('swapInventoryItems 0 1')
                    time.sleep(0.1)

                if inventory_string_roadrunner.find('quartz') != -1:
                    """ swaps quartz with log, to place back quartz """
                    if json.dumps(inventory_string_roadrunner.find('log') != -1):
                        self.agent_host3.sendCommand('swapInventoryItems 0 1')
                        # self.agent_host3.sendCommand("chat Wrong flag, I'll put it back!")
                    time.sleep(0.3)

                    """ as long as there is a false flag in the inventory, place it back """
                    counter = 0
                    while inventory_string_roadrunner.find('quartz') != -1:
                        print("%i : %s " % (counter, json.dumps(inventory_string_roadrunner.find('quartz')) != -1))
                        time.sleep(1)
                        self.agent_host3.sendCommand('use 1')
                        self.agent_host3.sendCommand('move 1')
                        counter += 1
                        if counter == (5 or 6 or 10 or 12 or 14 or 20):
                            self.agent_host3.sendCommand('move -1')
                        time.sleep(1)
                        world_state3 = self.get_safe_worldstate(3)
                        obs3 = self.get_worldstate_observations(world_state3)
                        last_inventory_roadrunner = obs3[u'inventory']
                        inventory_string_roadrunner = json.dumps(last_inventory_roadrunner)
                        breaker += 1
                        time.sleep(0.5)
                        if breaker == 120:
                            self.quit_after_crash(t, experiment_ID)

                    self.agent_host3.sendCommand('swapInventoryItems 0 1')
                    time.sleep(0.3)

                if inventory_string_tom.find('log') != -1:
                    self.flag_captured_tom = True
                    self.time_step_tom_captured_the_flag = time_step
                    self.append_save_file_with_flag(time_step, "Tom")
                    print(
                        "----------------------------------------------------------------Tom captured the flag after %i seconds!" % (
                            time_step))

                if inventory_string_roadrunner.find('log') != -1:
                    self.flag_captured_roadrunner = True
                    self.time_step_roadrunner_captured_the_flag = time_step
                    self.append_save_file_with_flag(time_step, "Roadrunner")
                    print(
                        "----------------------------------------------------------------Roadrunner captured the flag after %i seconds!" % (
                            time_step))

        if (self.flag_captured_jerry and (0 <= x2 <= 3 and 7 <= z2 <= 10)) or \
                (self.flag_captured_coyote and (0 <= x4 <= 3 and 7 <= z4 <= 10)):
            """ 
            if agent reached the target area:
            look down, set block, jump on it to reach wanted position and win the game 
            """
            if self.flag_captured_jerry:
                self.winner_behaviour(self.agent_host2, time_step, name="Jerry")
            if self.flag_captured_coyote:
                self.winner_behaviour(self.agent_host3, time_step, name="Coyote")

            self.mission_end = True
        else:
            if self.flag_captured_jerry or self.flag_captured_coyote:
                print("[INFO] Team Jerry and Coyote holds the flag.")
            else:
                last_inventory_jerry = obs2[u'inventory']
                last_inventory_coyote = obs4[u'inventory']
                inventory_string_jerry = json.dumps(last_inventory_jerry)
                inventory_string_coyote = json.dumps(last_inventory_coyote)
                if inventory_string_jerry.find('log') != -1:
                    """ swaps quartz with log, to place back quartz """
                    if json.dumps(inventory_string_jerry.find('quartz') != -1):
                        self.agent_host2.sendCommand('swapInventoryItems 0 1')
                        # self.agent_host2.sendCommand("chat Wrong flag, I'll put it back!")
                    time.sleep(0.3)

                    """ as long as there is a false flag in the inventory, place it back """
                    counter = 0
                    while inventory_string_jerry.find('log') != -1:
                        print("%i : %s " % (counter, json.dumps(inventory_string_jerry.find('log') != -1)))
                        time.sleep(1)
                        self.agent_host2.sendCommand('use 1')
                        self.agent_host2.sendCommand('move 1')
                        counter += 1
                        if counter == (5 or 6 or 10 or 12 or 14 or 20):
                            self.agent_host2.sendCommand('move -1')
                        time.sleep(1)
                        world_state2 = self.get_safe_worldstate(2)
                        obs2 = self.get_worldstate_observations(world_state2)
                        last_inventory_jerry = obs2[u'inventory']
                        inventory_string_jerry = json.dumps(last_inventory_jerry)
                        breaker += 1
                        time.sleep(0.5)
                        if breaker == 120:
                            self.quit_after_crash(t, experiment_ID)

                    self.agent_host2.sendCommand('swapInventoryItems 0 1')
                    time.sleep(0.3)

                if inventory_string_coyote.find('log') != -1:
                    """ swaps quartz with log, to place back quartz """
                    if json.dumps(inventory_string_coyote.find('quartz') != -1):
                        self.agent_host4.sendCommand('swapInventoryItems 0 1')
                        # self.agent_host4.sendCommand("chat Wrong flag, I'll put it back!")
                    time.sleep(0.3)

                    """ as long as there is a false flag in the inventory, place it back """
                    counter = 0
                    while inventory_string_coyote.find('log') != -1:
                        print("%i : %s " % (counter, json.dumps(inventory_string_coyote.find('log') != -1)))
                        time.sleep(1)
                        self.agent_host4.sendCommand('use 1')
                        self.agent_host4.sendCommand('move 1')
                        counter += 1
                        if counter == (5 or 6 or 10 or 12 or 14 or 20):
                            self.agent_host4.sendCommand('move -1')
                        time.sleep(1)
                        world_state4 = self.get_safe_worldstate(4)
                        obs4 = self.get_worldstate_observations(world_state4)
                        last_inventory_coyote = obs4[u'inventory']
                        inventory_string_coyote = json.dumps(last_inventory_coyote)
                        breaker += 1
                        time.sleep(0.1)
                        if breaker == 120:
                            self.quit_after_crash(t, experiment_ID)

                    self.agent_host4.sendCommand('swapInventoryItems 0 1')
                    time.sleep(0.3)

                if inventory_string_jerry.find('quartz') != -1:
                    self.flag_captured_jerry = True
                    self.time_step_jerry_captured_the_flag = time_step
                    self.append_save_file_with_flag(time_step, "Jerry")
                    print(
                        "----------------------------------------------------------------Jerry captured the flag after %i seconds!" % (
                            time_step))
                if inventory_string_coyote.find('quartz') != -1:
                    self.flag_captured_coyote = True
                    self.time_step_coyote_captured_the_flag = time_step
                    self.append_save_file_with_flag(time_step, "Coyote")
                    print(
                        "----------------------------------------------------------------Coyote captured the flag after %i seconds!" % (
                            time_step))

    def quit_after_crash(self, t, experiment_ID):

        """ send the MissionQuit Command to tell the Mod we finished """
        self.agent_host0.sendCommand('quit')
        self.agent_host1.sendCommand('quit')
        self.agent_host2.sendCommand('quit')
        self.agent_host3.sendCommand('quit')
        self.agent_host4.sendCommand('quit')

        """ Needed the Sleep-Timer, to give the Mod time, to quit the missions """
        time.sleep(3)

        """ Reset clients after crash """
        print("quit after crash: reset Clients")

        """ initialisation for the next episode, reset parameters, build new world """

        t += 1
        self.episode_counter += 1
        self.r1 = self.r2 = self.r3 = self.r4 = 0
        self.done1 = self.done2 = self.done3 = self.done4 = False
        self.done_team01 = self.done_team02 = self.mission_end = False
        self.overall_reward_agent_Jerry = self.overall_reward_agent_Tom = 0
        self.overall_reward_agent_roadrunner = self.overall_reward_agent_coyote = 0
        self.save_new_round(t)

        time.sleep(20)

        time_stamp_start = time.time()
        self.obs1, self.obs2, self.obs3, self.obs4 = self.reset_world(experiment_ID, time_stamp_start, t)

        self.too_close_counter = 0
        self.winner_agent = "-"
        self.time_step_tom_won = self.time_step_jerry_won = None
        self.time_step_roadrunner_won = self.time_step_coyote_won = None
        self.time_step_tom_captured_the_flag = self.time_step_jerry_captured_the_flag = None
        self.time_step_roadrunner_captured_the_flag = self.time_step_coyote_captured_the_flag = None
        self.time_step_agents_ran_into_each_other = None
        self.steps_tom = 0
        self.steps_jerry = 0
        self.steps_roadrunner = 0
        self.steps_coyote = 0

        """ recover """
        time.sleep(2)

        """if evaluator1 and evaluator2 is not None:
            evaluator1.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            evaluator2.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            if (successful_score is not None and
                    evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                break"""
        return t, self.obs1, self.obs2, self.r1, self.r2, self.obs3, self.obs4, self.r3, self.r4, self.done_team01, \
               self.done_team02, self.overall_reward_agent_Jerry, \
               self.overall_reward_agent_Tom, self.overall_reward_agent_roadrunner, self.overall_reward_agent_coyote

    def quit(self):
        """ send the MissionQuit Command to tell the Mod we finished """
        self.agent_host0.sendCommand('quit')
        self.agent_host1.sendCommand('quit')
        self.agent_host2.sendCommand('quit')
        self.agent_host3.sendCommand('quit')
        self.agent_host4.sendCommand('quit')

    def sending_mission_quit_commands(self, overall_reward_agent_Tom, overall_reward_agent_Jerry,
                                      overall_reward_agent_roadrunner, overall_reward_agent_coyote,
                                      time_step, obs1, r1, obs2, r2, obs3, r3, obs4, r4, outdir, outdir_loadable, t,
                                      tom, jerry, roadrunner, coyote, experiment_ID):

        """ send the MissionQuit Command to tell the Mod we finished """
        self.quit()

        """ Needed the Sleep-Timer, to give the Mod time, to quit the missions """
        time.sleep(3)

        """ Reset clients to speed up the following episodes. Otherwise the Minecraft client would get slow."""
        print("regular quit: reset Clients")

        """ save and show results of reward calculations """
        self.save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner, \
                          overall_reward_agent_coyote, time_step)
        print("Final Reward Tom:   ", overall_reward_agent_Tom)
        print("Final Reward Jerry: ", overall_reward_agent_Jerry)
        print("Final Reward Roadrunner:   ", overall_reward_agent_roadrunner)
        print("Final Reward Coyote: ", overall_reward_agent_coyote)

        """ end episode, save results """
        tom.stop_episode_and_train(obs1, r1, done=True)
        jerry.stop_episode_and_train(obs2, r2, done=True)
        roadrunner.stop_episode_and_train(obs3, r3, done=True)
        coyote.stop_episode_and_train(obs4, r4, done=True)
        print("outdir: %s step: %s " % (outdir, t))
        print("Tom's statistics:   ", tom.get_statistics())
        print("Jerry's statistics: ", jerry.get_statistics())
        print("Roadrunners's statistics:   ", roadrunner.get_statistics())
        print("Coyote's statistics: ", coyote.get_statistics())

        """ save the final model and results """
        save_agent(tom, t, outdir, outdir_loadable, logger, suffix='finished_agent_tom')
        save_agent(jerry, t, outdir, outdir_loadable, logger, suffix='finished_agent_jerry')
        save_agent(roadrunner, t, outdir, outdir_loadable, logger, suffix='finished_agent_roadrunner')
        save_agent(coyote, t, outdir, outdir_loadable, logger, suffix='finished_agent_coyote')

        print("save data for evaluation")
        """ save all the collected data for evaluation graphs """
        self.save_data_for_evaluation_plots(t, time_step, overall_reward_agent_Tom,
                                            overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
                                            overall_reward_agent_coyote)
        """ recover """
        time.sleep(2)

        """ initialisation for the next episode, reset parameters, build new world """
        print("initiate new episode")
        t += 1
        self.episode_counter += 1
        self.r1 = self.r2 = self.r3 = self.r4 = 0
        self.done1 = self.done2 = self.done3 = self.done4 = False
        self.done_team01 = self.done_team02 = self.mission_end = False
        self.overall_reward_agent_Jerry = self.overall_reward_agent_Tom = 0
        self.overall_reward_agent_roadrunner = self.overall_reward_agent_coyote = 0
        self.save_new_round(t)

        time_stamp_start = time.time()
        print("new start time: ", time_stamp_start)

        self.too_close_counter = 0
        self.winner_agent = "-"
        self.time_step_tom_won = self.time_step_jerry_won = None
        self.time_step_roadrunner_won = self.time_step_coyote_won = None
        self.time_step_tom_captured_the_flag = self.time_step_jerry_captured_the_flag = None
        self.time_step_roadrunner_captured_the_flag = self.time_step_coyote_captured_the_flag = None
        self.time_step_agents_ran_into_each_other = None
        self.steps_tom = 0
        self.steps_jerry = 0
        self.steps_roadrunner = 0
        self.steps_coyote = 0

        self.obs1, self.obs2, self.obs3, self.obs4 = self.reset_world(experiment_ID, time_stamp_start, t)

        """ recover """
        time.sleep(2)

        """if evaluator1 and evaluator2 is not None:
                evaluator1.evaluate_if_necessary(
                    t=t, episodes=episode_idx + 1)
                evaluator2.evaluate_if_necessary(
                    t=t, episodes=episode_idx + 1)
                if (successful_score is not None and
                        evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                    break"""
        return t, self.obs1, self.obs2, self.r1, self.r2, self.obs3, self.obs4, self.r3, self.r4, self.done_team01, \
               self.done_team02, self.overall_reward_agent_Jerry, \
               self.overall_reward_agent_Tom, self.overall_reward_agent_roadrunner, \
               self.overall_reward_agent_coyote

    def save_data(self):
        """
        save data to check if the plotted graph is correct
        just for user's information
        not for validation
        """
        datei = open(self.dirname + "/" + 'saved_data.txt', 'a')
        datei.write("\n\n#############################################################################################")
        datei.write("\nepisode_counter: %s" % self.evaluation_episode_counter)
        datei.write("\nepisode_time: %s" % self.evaluation_episode_time)
        datei.write("\ntoo_close_counter: %s" % self.evaluation_too_close_counter)
        datei.write("\nreward_tom: %s" % self.evaluation_reward_tom)
        datei.write("\nreward_jerry: %s" % self.evaluation_reward_jerry)
        datei.write("\nreward_roadrunner: %s" % self.evaluation_reward_roadrunner)
        datei.write("\nreward_coyote: %s" % self.evaluation_reward_coyote)
        datei.write("\nwinner: %s" % self.evaluation_winner_agent)
        datei.write("\ngame_won: %s" % self.evaluation_game_won_timestamp)
        datei.write("\nflag_captured_tom: %s" % self.evaluation_flag_captured_tom)
        datei.write("\nflag_captured_jerry: %s" % self.evaluation_flag_captured_jerry)
        datei.write("\nflag_captured_roadrunner: %s" % self.evaluation_flag_captured_roadrunner)
        datei.write("\nflag_captured_coyote: %s" % self.evaluation_flag_captured_coyote)
        datei.write("\nsteps_tom: %s" % self.evaluation_steps_tom)
        datei.write("\nsteps_jerry: %s" % self.evaluation_steps_jerry)
        datei.write("\nsteps_roadrunner: %s" % self.evaluation_steps_roadrunner)
        datei.write("\nsteps_coyote: %s" % self.evaluation_steps_coyote)
        datei.close()

    def save_data_for_evaluation_plots(self, t, time_step, overall_reward_agent_Tom,
                                       overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
                                       overall_reward_agent_coyote):
        """
        t: number of episode
        time_step: duration of the episode
        too_close_counter: how often agents came too close
        overall_reward_agent_<name>: reward of the agents
        winner_agent: agent's name who won the episode, if there is no "-"
        time_step_<name>_won: timestep, <agent_name> won the game, if not: 0
        time_step_<name>_captured_the_flag : timestep, <agent_name> captured the flag, if not: 0
        time_step_agents_ran_into_each_other: timestep; the agents ran into each other and the mission ends
        steps_<name>: steps per agent
        """

        if self.episode_counter >= 0:
            self.evaluation_agents_ran_into_each_other.append(self.time_step_agents_ran_into_each_other)
            if self.time_step_agents_ran_into_each_other is not None:
                print("Time, agents crashed: ", self.time_step_agents_ran_into_each_other)

            if self.time_step_agents_ran_into_each_other is None and self.winner_agent != "-":
                """ save data of valid episodes for the evaluation graph """
                print("save data of valid episodes for the evaluation graph")
                self.evaluation_episode_counter.append(self.episode_counter)
                self.evaluation_episode_time.append(time_step)
                self.evaluation_too_close_counter.append(self.too_close_counter)
                self.evaluation_reward_tom.append(overall_reward_agent_Tom)
                self.evaluation_reward_jerry.append(overall_reward_agent_Jerry)
                self.evaluation_reward_roadrunner.append(overall_reward_agent_roadrunner)
                self.evaluation_reward_coyote.append(overall_reward_agent_coyote)
                self.evaluation_winner_agent.append(self.winner_agent)

                if self.winner_agent == "Tom":
                    self.evaluation_game_won_timestamp.append(self.time_step_tom_won)
                if self.winner_agent == "Jerry":
                    self.evaluation_game_won_timestamp.append(self.time_step_jerry_won)
                if self.winner_agent == "Roadrunner":
                    self.evaluation_game_won_timestamp.append(self.time_step_roadrunner_won)
                if self.winner_agent == "Coyote":
                    self.evaluation_game_won_timestamp.append(self.time_step_coyote_won)

                self.evaluation_flag_captured_tom.append(self.time_step_tom_captured_the_flag)
                self.evaluation_flag_captured_jerry.append(self.time_step_jerry_captured_the_flag)
                self.evaluation_flag_captured_roadrunner.append(self.time_step_roadrunner_captured_the_flag)
                self.evaluation_flag_captured_coyote.append(self.time_step_coyote_captured_the_flag)
                self.evaluation_steps_tom.append(self.steps_tom)
                self.evaluation_steps_jerry.append(self.steps_jerry)
                self.evaluation_steps_roadrunner.append(self.steps_roadrunner)
                self.evaluation_steps_coyote.append(self.steps_coyote)

            print("save data")
            """ save above values to check the correctness of the graph, just for information """
            self.save_data()

            """ evaluate and print the plots """
            print("evaluate and plot")
            thesis_evaluation_swarm_intelligence.evaluate(t, self.evaluation_episode_counter,
                                                          self.evaluation_episode_time,
                                                          self.evaluation_too_close_counter, self.evaluation_reward_tom,
                                                          self.evaluation_reward_jerry,
                                                          self.evaluation_reward_roadrunner,
                                                          self.evaluation_reward_coyote,
                                                          self.evaluation_winner_agent,
                                                          self.evaluation_game_won_timestamp,
                                                          self.evaluation_flag_captured_tom,
                                                          self.evaluation_flag_captured_jerry,
                                                          self.evaluation_flag_captured_roadrunner,
                                                          self.evaluation_flag_captured_coyote,
                                                          self.evaluation_agents_ran_into_each_other,
                                                          self.evaluation_steps_tom, self.evaluation_steps_jerry,
                                                          self.evaluation_steps_roadrunner,
                                                          self.evaluation_steps_coyote, dirname=self.dirname)
            # print("left plot loop")
