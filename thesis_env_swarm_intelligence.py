import sys
import json
import logging
import time
import gym
import minecraft_py
from gym import spaces
import logging
from malmo import MalmoPython
import numpy as np
import malmoutils
import os
from chainer import optimizers
import chainer
from thesis.chainerrl import experiments
from thesis.chainerrl import explorers
from thesis.chainerrl import links
from thesis.chainerrl import misc
from thesis.chainerrl import q_functions
from thesis.chainerrl import replay_buffer
from thesis.chainerrl.experiments.evaluator import Evaluator
from thesis.chainerrl.experiments.evaluator import save_agent
from thesis.CDF_swarm_intelligence import thesis_evaluation_swarm_intelligence
from thesis import chainerrl

if sys.version_info[0] == 2:
    import Tkinter as tk
else:
    import tkinter as tk

"""
Logging for easier Debugging
options: DEBUG, ALL, INFO
"""
logging.basicConfig(level=logging.INFO)
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
    reset_world(experiment_ID)
    do_action(actions, agent_num)
    get_video_frame(world_state, agent_num)
    get_observation(world_state)
    save_new_round(t)
    append_save_file_with_flag(time_step, name)
    append_save_file_with_fail()
    append_save_file_with_agents_fail()
    append_save_file_with_finish(time_step, name)
    save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
        overall_reward_agent_coyote,time_step)
    get_cell_agents()
    get_current_cell_agents()
    get_world_state_ob(agent_num)
    renew_world_state(agent_num)
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
        overall_reward_agent_Jerry, overall_reward_agent_roadrunner, overall_reward_agent_coyote, dirname)
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

    """ collected data for evaluation """
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

        """ same clientpool as in main py script, needed in both places, otherwise doesn't work """
        self.client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001), ('127.0.0.1', 10002), ('127.0.0.1', 10003),
                            ('127.0.0.1', 10004)]
        self.mc_process = None
        self.mission_end = False

    def init(self, client_pool=None, start_minecraft=None,
             continuous_discrete=True, add_noop_command=None,
             max_retries=90, retry_sleep=10, step_sleep=0.001, skip_steps=0,
             videoResolution=None, videoWithDepth=None,
             observeRecentCommands=None, observeHotBar=None,
             observeFullInventory=None, observeGrid=None,
             observeDistance=None, observeChat=None,
             allowContinuousMovement=None, allowDiscreteMovement=None,
             allowAbsoluteMovement=None, recordDestination=None,
             recordObservations=None, recordRewards=None,
             recordCommands=None, recordMP4=None,
             gameMode=None, forceWorldReset=None):

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
            # if there are any parameters, remove current command handlers first
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
        unused_actions:     not wanted actions
        discrete_actions:   wanted actions
        """

        unused_actions = []
        discrete_actions = []
        chs = self.mission_spec.getListOfCommandHandlers(0)
        for ch in chs:
            cmds = self.mission_spec.getAllowedCommands(0, ch)
            for command in cmds:
                logger.debug(ch + ":" + command)
                if command in ["movenorth", "movesouth", "movewest", "moveeast", "attack", "turn"]: # wanted actions
                    discrete_actions.append(command + " 1")
                    if command == "turn":
                        discrete_actions.append(command + " -1")
                else:
                    unused_actions.append(
                        command)

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
        WIP! learning process, etc
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

    def step_generating(self, action, agent_num):
        """
        time step in arena
        next action is executed
        reward of current state is calculated and summed up with the overall reward
        PARAMETERS: action, agent_num
        RETURN: image, reward, done, info
        """
        reward1 = reward2 = reward3 = reward4 = 0
        world_state1 = world_state2 = world_state3 = world_state4 = 0
        image1 = image2 = image3 = image4 = 0
        done_team01 = done_team02 = False
        info1 = info2 = info3 = info4 = {}

        """ loop to minimize errors if there is a broken world_state"""
        while world_state1 == 0 or world_state2 == 0 or world_state3 == 0 or world_state4 == 0:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            world_state3 = self.agent_host3.peekWorldState()
            world_state4 = self.agent_host4.peekWorldState()

        if agent_num == 1:
            if world_state1.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state1 = self.agent_host1.getWorldState()
        if agent_num == 2:
            if world_state2.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state2 = self.agent_host2.getWorldState()
        if agent_num == 3:
            if world_state3.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state3 = self.agent_host3.getWorldState()
        if agent_num == 4:
            if world_state4.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state4 = self.agent_host4.getWorldState()

        """ calculate reward of current state """
        if agent_num == 1:
            for r in world_state1.rewards:
                reward1 += r.getValue()
        if agent_num == 2:
            for r in world_state2.rewards:
                reward2 += r.getValue()
        if agent_num == 3:
            for r in world_state3.rewards:
                reward3 += r.getValue()
        if agent_num == 4:
            for r in world_state4.rewards:
                reward4 += r.getValue()

        """ take the last frame from world state | 'done'-flag indicates, if mission is still running """
        if agent_num == 1:
            image1 = self.get_video_frame(world_state1, 1)
            done_team01 = not world_state1.is_mission_running
        if agent_num == 2:
            image2 = self.get_video_frame(world_state2, 1)
            done_team02 = not world_state2.is_mission_running
        if agent_num == 3:
            image3 = self.get_video_frame(world_state3, 1)
            done_team01 = not world_state3.is_mission_running
        if agent_num == 4:
            image4 = self.get_video_frame(world_state4, 1)
            done_team02 = not world_state4.is_mission_running

        """
        collected information during the run
        """
        if agent_num == 1:
            info1 = {}
            info1['has_mission_begun'] = world_state1.has_mission_begun
            info1['is_mission_running'] = world_state1.is_mission_running
            info1['number_of_video_frames_since_last_state'] = world_state1.number_of_video_frames_since_last_state
            info1['number_of_rewards_since_last_state'] = world_state1.number_of_rewards_since_last_state
            info1['number_of_observations_since_last_state'] = world_state1.number_of_observations_since_last_state
            info1['mission_control_messages'] = [msg.text for msg in world_state1.mission_control_messages]
            info1['observation'] = self.get_observation(world_state1)

        if agent_num == 2:
            info2 = {}
            info2['has_mission_begun'] = world_state2.has_mission_begun
            info2['is_mission_running'] = world_state2.is_mission_running
            info2['number_of_video_frames_since_last_state'] = world_state2.number_of_video_frames_since_last_state
            info2['number_of_rewards_since_last_state'] = world_state2.number_of_rewards_since_last_state
            info2['number_of_observations_since_last_state'] = world_state2.number_of_observations_since_last_state
            info2['mission_control_messages'] = [msg.text for msg in world_state2.mission_control_messages]
            info2['observation'] = self.get_observation(world_state2)

        if agent_num == 3:
            info3 = {}
            info3['has_mission_begun'] = world_state3.has_mission_begun
            info3['is_mission_running'] = world_state3.is_mission_running
            info3['number_of_video_frames_since_last_state'] = world_state3.number_of_video_frames_since_last_state
            info3['number_of_rewards_since_last_state'] = world_state3.number_of_rewards_since_last_state
            info3['number_of_observations_since_last_state'] = world_state3.number_of_observations_since_last_state
            info3['mission_control_messages'] = [msg.text for msg in world_state3.mission_control_messages]
            info3['observation'] = self.get_observation(world_state3)

        if agent_num == 4:
            info4 = {}
            info4['has_mission_begun'] = world_state4.has_mission_begun
            info4['is_mission_running'] = world_state4.is_mission_running
            info4['number_of_video_frames_since_last_state'] = world_state4.number_of_video_frames_since_last_state
            info4['number_of_rewards_since_last_state'] = world_state4.number_of_rewards_since_last_state
            info4['number_of_observations_since_last_state'] = world_state4.number_of_observations_since_last_state
            info4['mission_control_messages'] = [msg.text for msg in world_state4.mission_control_messages]
            info4['observation'] = self.get_observation(world_state4)

        if agent_num == 1:
            return image1, reward1, done_team01, info1
        if agent_num == 2:
            return image2, reward2, done_team02, info2
        if agent_num == 3:
            return image3, reward3, done_team01, info3
        if agent_num == 4:
            return image4, reward4, done_team02, info4

    def reset_world(self, experiment_ID):
        """
        reset the arena and start the missions per agent
        The sleep-timer of 6sec is required, because the client needs far too much time to set up the mission
        for the first time.
        All followed missions start faster (sometimes).
        PARAMETERS: experiment_ID
        """
        print("force world reset........")
        self.flag_captured_tom = False
        self.flag_captured_jerry = False
        self.flag_captured_roadrunner = False
        self.flag_captured_coyote = False

        time.sleep(0.1)

        print(self.client_pool)

        for retry in range(self.max_retries + 1):
            try:
                """ start missions for every client """
                print("starting mission for Tom")
                time.sleep(6)
                self.agent_host1.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              0, experiment_ID)

                print("starting mission for Jerry")
                time.sleep(6)
                self.agent_host2.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              1, experiment_ID)

                print("starting mission for Roadrunner")
                time.sleep(6)
                self.agent_host3.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              2, experiment_ID)

                print("starting mission for Coyote")
                time.sleep(6)
                self.agent_host4.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              3, experiment_ID)

                print("starting mission for Skye")
                time.sleep(6)
                self.agent_host0.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              4, experiment_ID)

                print("\nmissions successfully started.....\n")
                break
            except RuntimeError as e:
                if retry == self.max_retries:
                    logger.error("Error starting mission: " + str(e))
                    raise
                else:
                    logger.warning("Error starting mission: " + str(e))
                    logger.info("Sleeping for %d seconds...", self.retry_sleep)
                    time.sleep(self.retry_sleep)

        logger.info("Waiting for the mission to start.")
        world_state1 = self.agent_host1.getWorldState()
        world_state2 = self.agent_host2.getWorldState()
        world_state3 = self.agent_host3.getWorldState()
        world_state4 = self.agent_host4.getWorldState()
        while not world_state1.has_mission_begun and not world_state2.has_mission_begun and \
                not world_state3.has_mission_begun and not world_state4.has_mission_begun:
            time.sleep(0.1)
            world_state1 = self.agent_host1.getWorldState()
            world_state2 = self.agent_host2.getWorldState()
            world_state3 = self.agent_host3.getWorldState()
            world_state4 = self.agent_host4.getWorldState()
            for error in world_state1.errors and world_state2.errors and world_state3.errors and world_state4.errors:
                logger.warning(error.text)

        logger.info("Mission running")

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

        """ count the steps for the individual agent """
        if agent_num == 1:
            self.steps_tom += 1
        if agent_num == 2:
            self.steps_jerry += 1
        if agent_num == 3:
            self.steps_roadrunner += 1
        if agent_num == 4:
            self.steps_coyote += 1

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
            """ if mission ends befor we got a frame, just take the last frame to reduce exceptions """
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
        datei = open('results.txt', 'a')
        datei.write("-------------- ROUND %i --------------\n" % (t))
        datei.close()

    def append_save_file_with_flag(self, time_step, name):
        """
        saves the flagholder in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("%s captured the flag after %i seconds.\n" % (name, time_step))
        datei.close()

    def append_save_file_with_fail(self):
        """
        saves the failes in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("X the mission failed X.\n")
        datei.close()

    def append_save_file_with_agents_fail(self):
        """
        saves the explicit agent-failes in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("X the mission failed: the agents ran into each other or got stranded in the field X.\n")
        datei.close()

    def append_save_file_with_finish(self, time_step, name):
        """
        saves the winner in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("%s won the game after %i seconds.\n" % (name, time_step))
        datei.close()

    def save_results(self, overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
                     overall_reward_agent_coyote, time_step):
        """
        saves the results in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("The agents were %i times very close to each other.\n" % (self.too_close_counter))
        datei.write("Reward Tom: %i, Reward Jerry: %i , Reward Roadrunner: %i, Reward Coyote: %i , Time: %f \n\n" % (
            overall_reward_agent_Tom, overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
            overall_reward_agent_coyote, time_step))
        datei.close()

    def get_cell_agents(self):
        """
        gets the cell coordinates for the agents to compare with
        """
        world_state1 = world_state2 = world_state3 = world_state4 = 0
        while world_state1 == 0 or world_state2 == 0 or world_state3 == 0 or world_state4 == 0:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            world_state3 = self.agent_host3.peekWorldState()
            world_state4 = self.agent_host4.peekWorldState()
            if len(world_state1.observations) >= 1 and len(world_state2.observations) >= 1 and \
                    len(world_state3.observations) >= 1 and len(world_state4.observations) >= 1:
                msg1 = world_state1.observations[-1].text
                msg2 = world_state2.observations[-1].text
                msg3 = world_state3.observations[-1].text
                msg4 = world_state4.observations[-1].text
                ob1 = json.loads(msg1)
                ob2 = json.loads(msg2)
                ob3 = json.loads(msg3)
                ob4 = json.loads(msg4)

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
        world_state1 = world_state2 = world_state3 = world_state4 = 0
        while world_state1 == 0 or world_state2 == 0 or world_state3 == 0 or world_state4 == 0:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            world_state3 = self.agent_host3.peekWorldState()
            world_state4 = self.agent_host4.peekWorldState()
            if len(world_state1.observations) >= 1 and len(world_state2.observations) >= 1 and \
                    len(world_state3.observations) >= 1 and len(world_state4.observations) >= 1:
                msg1 = world_state1.observations[-1].text
                msg2 = world_state2.observations[-1].text
                msg3 = world_state3.observations[-1].text
                msg4 = world_state4.observations[-1].text
                ob1 = json.loads(msg1)
                ob2 = json.loads(msg2)
                ob3 = json.loads(msg3)
                ob4 = json.loads(msg4)
                if "cell" in ob1 and "cell" in ob2 and "cell" in ob3 and "cell" in ob4:
                    self.cell_now_tom = ob1.get(u'cell', 0)
                    self.cell_now_jerry = ob2.get(u'cell', 0)
                    self.cell_now_roadrunner = ob3.get(u'cell', 0)
                    self.cell_now_coyote = ob4.get(u'cell', 0)
                    print("current cell tom: ", self.cell_now_tom)
                    print("current cell jerry: ", self.cell_now_jerry)
                    print("current cell roadrunner: ", self.cell_now_roadrunner)
                    print("current cell coyote: ", self.cell_now_coyote)

    def get_world_state_ob(self, agent_num):
        """
        get current world_state of called agent
        """
        world_state = 0
        while world_state == 0:

            if agent_num == 1:
                world_state = self.agent_host1.peekWorldState()
            if agent_num == 2:
                world_state = self.agent_host2.peekWorldState()
            if agent_num == 3:
                world_state = self.agent_host3.peekWorldState()
            if agent_num == 4:
                world_state = self.agent_host4.peekWorldState()

            if len(world_state.observations) >= 1:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                return ob

    def renew_world_state(self, agent_num):
        """
        if the world_state failes and there are no observations,
        just get a new one here
        """
        world_state = 0

        if agent_num == 1:
            world_state = self.agent_host1.peekWorldState()
        if agent_num == 2:
            world_state = self.agent_host2.peekWorldState()
        if agent_num == 3:
            world_state = self.agent_host3.peekWorldState()
        if agent_num == 4:
            world_state = self.agent_host4.peekWorldState()

        print("world_state: ", world_state)

        return world_state

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
        while world_state:
            if len(world_state.observations) >= 1:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                time_now = time.time()

                if time_now - self.time_stamp_start_for_distance > 30:
                    """ fetch cell every 30 seconds """
                    self.get_cell_agents()
                    self.time_stamp_start_for_distance = time.time()

                if "XPos" in ob and "ZPos" in ob and "YPos" in ob:
                    x = ob[u'XPos']
                    y = ob[u'YPos']
                    z = ob[u'ZPos']

                seconds = time_now - self.time_stamp_start_for_distance
                print("seconds: ", int(seconds))
                """if int(seconds) == 15:
                    # WIP
                    if self.fetched_cell_tom == self.cell_now_tom and agent_num == 1:
                        teleport_x = 9.5
                        teleport_z = 0.5
                        tp_command = "tp " + str(teleport_x) + " 4 " + str(teleport_z)
                        self.agent_host1.sendCommand(tp_command)
                        print("teleported Tom")

                    if self.fetched_cell_jerry == self.cell_now_jerry and agent_num == 2:
                        teleport_x = 6.5
                        teleport_z = 15.5
                        tp_command = "tp " + str(teleport_x) + " 4 " + str(teleport_z)
                        self.agent_host2.sendCommand(tp_command)
                        print("teleported Jerry")

                    if self.fetched_cell_roadrunner == self.cell_now_roadrunner and agent_num == 3:
                        teleport_x = 15.5
                        teleport_z = 6.5

                        tp_command = "tp " + str(teleport_x) + " 4 " + str(teleport_z)
                        self.agent_host3.sendCommand(tp_command)
                        print("teleported Roadrunner")

                    if self.fetched_cell_coyote == self.cell_now_coyote and agent_num == 4:
                        teleport_x = 0.5
                        teleport_z = 9.5

                        tp_command = "tp " + str(teleport_x) + " 4 " + str(teleport_z)
                        self.agent_host4.sendCommand(tp_command)
                        print("teleported Coyote")"""

                if int(seconds) == 28:
                    self.get_current_cell_agents()
                    if self.fetched_cell_tom == self.cell_now_tom and self.fetched_cell_jerry == self.cell_now_jerry \
                            and self.fetched_cell_roadrunner == self.cell_now_roadrunner and \
                            self.fetched_cell_coyote == self.cell_now_coyote:
                        print("The agents crashed.")
                        # if abs(self.fetched_cell_tom - self.fetched_cell_jerry) <= 1:
                        #    print("The agents ran into each other again.")
                        self.time_step_agents_ran_into_each_other = time_step
                        self.append_save_file_with_agents_fail()
                        self.mission_end = True

                return x, y, z
            else:
                if t == 10:
                    self.append_save_file_with_fail()
                    self.time_step_agents_ran_into_each_other = time_step
                    self.mission_end = True
                    return x, y, z
                else:
                    time.sleep(1)
                    t += 1
                    world_state = self.renew_world_state(agent_num)
                    print(t)

    def movenorth1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step north """
        x_ziel = x
        if z == 0 or z == 2 and 14 <= x <= 15:
            z_ziel = z
        else:
            z_ziel = z - 1
        return x_ziel, z_ziel

    def movesouth1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step south """
        x_ziel = x
        if z == 15 or z == 13 and 0 <= x <= 1:
            z_ziel = z
        else:
            z_ziel = z + 1
        return x_ziel, z_ziel

    def moveeast1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step east """
        if x == 15 or x == 13 and 0 <= z <= 1:
            x_ziel = x
        else:
            x_ziel = x + 1
        z_ziel = z
        return x_ziel, z_ziel

    def movewest1_function(self, x, z):
        """ calculates the new x and z values after the agent moved one step west """
        if x == 0 or x == 2 and 14 <= z <= 15:
            x_ziel = x
        else:
            x_ziel = x - 1
        z_ziel = z
        return x_ziel, z_ziel

    def get_new_position(self, action, x, z):
        """ calculates the new position of the agent """

        for spc, cmds in zip(self.action_spaces, self.action_names):
            if isinstance(spc, spaces.Discrete):
                action_name = cmds[action]

        if action_name == "movenorth 1":
            x_new, z_new = self.movenorth1_function(x, z)
        elif action_name == "movesouth 1":
            x_new, z_new = self.movesouth1_function(x, z)
        elif action_name == "moveeast 1":
            x_new, z_new = self.moveeast1_function(x, z)
        elif action_name == "movewest 1":
            x_new, z_new = self.movewest1_function(x, z)
        else:
            x_new = x
            z_new = z

        return x_new, z_new

    def approve_distance(self, tom, jerry, roadrunner, coyote, obs1, obs2, obs3, obs4, r1, r2, r3, r4,
                         action1, action2, action3, action4, time_step):
        """
        check if agents are to near to eachother
        if so, the next actions are calculated new
        """
        steps_approved = needed_new_calculation = False
        world_state1 = world_state2 = world_state3 = world_state4 = 0

        """ checks, if world_state is read correctly, if not, trys again"""
        while (world_state1 == 0) or (world_state2 == 0) or (world_state3 == 0) or (world_state4 == 0):
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            world_state3 = self.agent_host3.peekWorldState()
            world_state4 = self.agent_host4.peekWorldState()
            print("..")

        x1, y1, z1 = self.get_position_in_arena(world_state1, time_step, 1)
        x2, y2, z2 = self.get_position_in_arena(world_state2, time_step, 2)
        x3, y3, z3 = self.get_position_in_arena(world_state3, time_step, 3)
        x4, y4, z4 = self.get_position_in_arena(world_state4, time_step, 4)

        time.sleep(0.1)
        while not steps_approved:
            """new position for agent tom"""
            x1_new, z1_new = self.get_new_position(action1, x1, z1)

            """new position for agent jerry"""
            x2_new, z2_new = self.get_new_position(action2, x2, z2)

            """new position for agent roadrunner"""
            x3_new, z3_new = self.get_new_position(action3, x3, z3)

            """new position for agent coyote"""
            x4_new, z4_new = self.get_new_position(action4, x4, z4)

            """checks, if the agents would run into each other if they took the step"""
            # WIP !!!
            time.sleep(0.5)
            if int(x1_new) == int(x2_new) and int(z1_new) == int(z2_new) or int(x1_new) == int(x3_new) \
                    and int(z1_new) == int(z3_new) or \
                    int(x1_new) == int(x4_new) and int(z1_new) == int(z4_new) or int(x2_new) == int(x3_new) \
                    and int(z2_new) == int(z3_new) or \
                    int(x2_new) == int(x4_new) and int(z2_new) == int(z4_new) or int(x3_new) == int(x4_new) \
                    and int(z3_new) == int(z4_new):

                needed_new_calculation = True
                action1 = tom.act_and_train(obs1, r1)
                action2 = jerry.act_and_train(obs2, r2)
                action3 = roadrunner.act_and_train(obs3, r3)
                action4 = coyote.act_and_train(obs4, r4)
                time.sleep(0.5)
            else:
                steps_approved = True
                if needed_new_calculation:
                    self.too_close_counter += 1

        print("calculated actions: %s, %s, %s, %s" % (action1, action2, action3, action4))
        return action1, action2, action3, action4

    def winner_behaviour(self, agent_host, time_step, name):
        """
        hardcoded winner-behaviour to end the mission:
        look down, place flag, look up again and jump on flag
        """
        agent_host.sendCommand("chat I won the game!")
        self.append_save_file_with_finish(time_step, name)
        self.time_step_tom_won = time_step
        self.winner_agent = name

        agent_host.sendCommand("look 1")
        time.sleep(0.2)
        agent_host.sendCommand("use 1")
        time.sleep(0.2)
        agent_host.sendCommand("look -1")
        time.sleep(0.2)
        agent_host.sendCommand("jumpmove 1")

    def check_inventory(self, time_step):
        """
        checks, if the agent got the flag in his inventory
        """
        world_state1 = world_state2 = world_state3 = world_state4 = 0
        x1 = y1 = z1 = x2 = y2 = z2 = x3 = y3 = z3 = x4 = y4 = z4 = 0

        """ checks, if world_state is read correctly, if not, trys again """
        while world_state1 == 0 or world_state2 == 0 or world_state3 == 0 or world_state4 == 0:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            world_state3 = self.agent_host3.peekWorldState()
            world_state4 = self.agent_host4.peekWorldState()
            print("..")

        if world_state1.observations and world_state2.observations and world_state3.observations \
                and world_state4.observations:

            msg1 = world_state1.observations[-1].text
            msg2 = world_state2.observations[-1].text
            msg3 = world_state3.observations[-1].text
            msg4 = world_state4.observations[-1].text
            obs1 = json.loads(msg1)
            obs2 = json.loads(msg2)
            obs3 = json.loads(msg3)
            obs4 = json.loads(msg4)

            """ checks, if position is calculated correctly, if not, trys again """
            while (y1 == 0) or (y2 == 0) or (y3 == 0) or (y4 == 0):
                x1, y1, z1 = self.get_position_in_arena(world_state1, time_step, 1)
                x2, y2, z2 = self.get_position_in_arena(world_state2, time_step, 2)
                x3, y3, z3 = self.get_position_in_arena(world_state3, time_step, 3)
                x4, y4, z4 = self.get_position_in_arena(world_state4, time_step, 4)
                print("..")

            """ fetch the current cells """
            self.get_current_cell_agents()

            if (self.flag_captured_tom and (12 <= x1 <= 15 and 0 <= z1 <= 4)) or self.flag_captured_roadrunner and \
                    (12 <= x3 <= 15 and 0 <= z3 <= 4):
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
                        if json.dumps(last_inventory_tom[1]).find('quartz') != -1:
                            self.agent_host1.sendCommand("swapInventoryItems 0 1")
                        #self.agent_host1.sendCommand("chat Wrong flag, I'll put it back!")
                        time.sleep(0.1)
                        self.agent_host1.sendCommand("use")
                        time.sleep(0.1)
                        self.agent_host1.sendCommand("swapInventoryItems 0 1")
                    if inventory_string_roadrunner.find('quartz') != -1:
                        """ swaps quartz with log, to place back quartz """
                        if json.dumps(last_inventory_roadrunner[1]).find('quartz') != -1:
                            self.agent_host3.sendCommand("swapInventoryItems 0 1")
                        #self.agent_host3.sendCommand("chat Wrong flag, I'll put it back!")
                        time.sleep(0.1)
                        self.agent_host3.sendCommand("use")
                        time.sleep(0.1)
                        self.agent_host3.sendCommand("swapInventoryItems 0 1")
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

            if (self.flag_captured_jerry and (0 <= x2 <= 4 and 10 <= z2 <= 15)) or \
                    (self.flag_captured_coyote and (0 <= x4 <= 4 and 10 <= z4 <= 15)):
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
                        """ swaps quartz with log, to place back log """
                        if json.dumps(last_inventory_jerry[1]).find('log') != -1:
                            self.agent_host2.sendCommand("swapInventoryItems 0 1")
                        #self.agent_host2.sendCommand("chat Wrong flag, I'll put it back!")
                        time.sleep(0.1)
                        self.agent_host2.sendCommand("use")
                        time.sleep(0.1)
                        self.agent_host2.sendCommand("swapInventoryItems 0 1")
                    if inventory_string_coyote.find('log') != -1:
                        """ swaps quartz with log, to place back log """
                        if json.dumps(last_inventory_coyote[1]).find('log') != -1:
                            self.agent_host4.sendCommand("swapInventoryItems 0 1")
                        #self.agent_host4.sendCommand("chat Wrong flag, I'll put it back!")
                        time.sleep(0.1)
                        self.agent_host4.sendCommand("use")
                        time.sleep(0.1)
                        self.agent_host4.sendCommand("swapInventoryItems 0 1")
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

    def sending_mission_quit_commands(self, overall_reward_agent_Tom, overall_reward_agent_Jerry,
                                      overall_reward_agent_roadrunner, overall_reward_agent_coyote,
                                      time_step, obs1, r1, obs2, r2, obs3, r3, obs4, r4, outdir, t,
                                      tom, jerry, roadrunner, coyote, experiment_ID):

        """ send the MissionQuit Command to tell the Mod we finished """
        self.agent_host0.sendCommand("quit")
        self.agent_host1.sendCommand("quit")
        self.agent_host2.sendCommand("quit")
        self.agent_host3.sendCommand("quit")
        self.agent_host4.sendCommand("quit")

        """ save-path for the evaluation-plot """
        dirname = os.path.join(outdir, 'plots')
        print("dirname: ", dirname)

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
        save_agent(tom, t, outdir, logger, suffix='_finish_01')
        save_agent(jerry, t, outdir, logger, suffix='_finish_02')
        save_agent(roadrunner, t, outdir, logger, suffix='_finish_03')
        save_agent(coyote, t, outdir, logger, suffix='_finish_04')

        """ save all the collected data for evaluation graphs """
        self.save_data_for_evaluation_plots(t, time_step, overall_reward_agent_Tom,
                                            overall_reward_agent_Jerry, overall_reward_agent_roadrunner,
                                            overall_reward_agent_coyote, dirname)
        """ recover """
        time.sleep(2)

        """ initialisation for the next episode, reset parameters, build new world """
        t += 1
        self.episode_counter += 1
        r1 = r2 = 0
        done_team01 = done_team02 = self.mission_end = False
        overall_reward_agent_Jerry = overall_reward_agent_Tom = 0
        overall_reward_agent_roadrunner = overall_reward_agent_coyote = 0
        self.save_new_round(t)

        obs1, obs2, obs3, obs4 = self.reset_world(experiment_ID)

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
        time.sleep(5)

        """if evaluator1 and evaluator2 is not None:
            evaluator1.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            evaluator2.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            if (successful_score is not None and
                    evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                break"""
        return t, obs1, obs2, r1, r2, obs3, obs4, r3, r4, done_team01, done_team02, overall_reward_agent_Jerry, \
               overall_reward_agent_Tom, overall_reward_agent_roadrunner, overall_reward_agent_coyote

    def save_data(self):
        """
        save data to check if the plotted graph is correct
        just for user's information
        not for validation
        """
        datei = open('saved_data.txt', 'a')
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
                                       overall_reward_agent_coyote, dirname):
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

        if self.episode_counter > 0:
            """ Episode 0 is skipped, because there just starts the initialisation of the world, they do nothing. """
            self.evaluation_agents_ran_into_each_other.append(self.time_step_agents_ran_into_each_other)
            print("Time, agents crashed: ", self.time_step_agents_ran_into_each_other)

            if self.time_step_agents_ran_into_each_other is None:
                """ save data of valid episodes for the evaluation graph """
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

            """ save above values to check the correctness of the graph, just for information """
            self.save_data()

            """ evaluate and print the plots """
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
                                                          self.evaluation_agents_ran_into_each_other, dirname,
                                                          self.evaluation_steps_tom, self.evaluation_steps_jerry,
                                                          self.evaluation_steps_roadrunner,
                                                          self.evaluation_steps_coyote)
