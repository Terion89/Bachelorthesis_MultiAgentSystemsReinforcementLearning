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
from thesis import thesis_evaluation_experiment

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
    reset_world()
    do_action(actions, agent_num)
    get_video_frame(world_state, agent_num)
    get_observation(world_state)
    save_new_round(t)
    append_save_file_with_flag(time_step, name)
    append_save_file_with_fail()
    append_save_file_with_agents_fail()
    append_save_file_with_finish(time_step, name)
    save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step)
    get_cell_agents()
    get_current_cell_agents()
    get_position_in_arena(world_state)
    distance()
    check_inventory(time_step)
"""


class ThesisEnvExperiment(gym.Env):
    """
    initialize agents and give commandline permissions
    """
    metadata = {'render.modes': ['human']}
    """ Agent 01: Tom """
    agent_host1 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host1)
    """ Agent 02: Jerry """
    agent_host2 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host2)
    """ Agent 03: Skye """
    agent_host3 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host3)

    """global variables to remember, if somebody already catched the flag"""
    flag_captured_tom = flag_captured_jerry = False
    fetched_cell_tom = fetched_cell_jerry = cell_now_tom = cell_now_jerry = 0
    time_stamp_start_for_distance = 0
    too_close_counter = 0
    time_step_tom_won = None
    time_step_jerry_won = None
    time_step_tom_captured_the_flag = None
    time_step_jerry_captured_the_flag = None
    winner_agent = "-"
    time_step_agents_ran_into_each_other = None
    steps_tom = 0
    steps_jerry = 0
    episode_counter = 0

    """ collected data for evaluation """
    evaluation_episode_counter = []
    evaluation_too_close_counter = []
    evaluation_episode_time = []
    evaluation_flag_captured_tom = []
    evaluation_flag_captured_jerry = []
    evaluation_agents_ran_into_each_other = []
    evaluation_game_won_timestamp = []
    evaluation_winner_agent = []
    evaluation_reward_tom = []
    evaluation_reward_jerry = []
    evaluation_steps_tom = []
    evaluation_steps_jerry = []

    def __init__(self):
        super(ThesisEnvExperiment, self).__init__()
        """
        load the mission file
        format: XML
        """
        mission_file = 'capture_the_flag_xml_mission_DQL.xml'
        self.load_mission_file(mission_file)
        print("Mission loaded: Capture the Flag")

        self.client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001), ('127.0.0.1', 10002)]
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
        dummy image just for the first observation
        """
        self.last_image1 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.last_image2 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.float32)
        self.create_action_space()

        """ 
        mission recording 
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
        game mode
        """
        if gameMode:
            if gameMode == "spectator":
                self.mission_spec.setModeToSpectator()
            elif gameMode == "creative":
                self.mission_spec.setModeToCreative()
            elif gameMode == "survival":
                logger.warn("Cannot force survival mode, assuming it is the default.")
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
        # collect different actions based on allowed commands
        unused_actions = []
        discrete_actions = []
        chs = self.mission_spec.getListOfCommandHandlers(0)
        for ch in chs:
            cmds = self.mission_spec.getAllowedCommands(0, ch)
            for command in cmds:
                logger.debug(ch + ":" + command)
                if command in ["movenorth", "movesouth", "moveeast", "movewest", "attack", "turn"]:
                    discrete_actions.append(command + " 1")
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
        learning process
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
            n_actions = action_space.n
            # print("n_actions: ", n_actions)
            q_func = q_functions.FCStateQFunctionWithDiscreteAction(
                obs_size, n_actions,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers)
            # print("q_func ", q_func)
            # Use epsilon-greedy for exploration
            explorer = explorers.LinearDecayEpsilonGreedy(
                args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
                action_space.sample)
            # print("explorer: ", explorer)

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
        reward of actual state is calculated and summed up with the overall reward
        RETURN: image, reward, done, info
        """
        reward1 = 0
        reward2 = 0

        world_state1 = self.agent_host1.peekWorldState()
        world_state2 = self.agent_host2.peekWorldState()
        if agent_num == 1:
            if world_state1.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state1 = self.agent_host1.getWorldState()
        else:
            if world_state2.is_mission_running:
                """ take action """
                self.do_action(action, agent_num)
                """ wait for the new state """
            world_state2 = self.agent_host2.getWorldState()

        """ calculate reward of current state """
        if agent_num == 1:
            for r in world_state1.rewards:
                reward1 += r.getValue()
        else:
            for r in world_state2.rewards:
                reward2 += r.getValue()

        """ take the last frame from world state | 'done'-flag indicated, if mission is still running """
        if agent_num == 1:
            image1 = self.get_video_frame(world_state1, 1)
            done1 = not world_state1.is_mission_running
        else:
            image2 = self.get_video_frame(world_state2, 2)
            done2 = not world_state1.is_mission_running

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

        else:
            info2 = {}
            info2['has_mission_begun'] = world_state2.has_mission_begun
            info2['is_mission_running'] = world_state2.is_mission_running
            info2['number_of_video_frames_since_last_state'] = world_state2.number_of_video_frames_since_last_state
            info2['number_of_rewards_since_last_state'] = world_state2.number_of_rewards_since_last_state
            info2['number_of_observations_since_last_state'] = world_state2.number_of_observations_since_last_state
            info2['mission_control_messages'] = [msg.text for msg in world_state2.mission_control_messages]
            info2['observation'] = self.get_observation(world_state2)

        if agent_num == 1:
            return image1, reward1, done1, info1
        else:
            return image2, reward2, done2, info2

    def reset_world(self, experiment_ID):
        """
        reset the arena and start the missions per agent
        The sleep-timer of 6sec is required, because the client needs far too much time to set up the mission
        for the first time.
        All followed missions start faster.
        """
        print("force world reset........")
        self.flag_captured_tom = False
        self.flag_captured_jerry = False

        time.sleep(0.1)

        print(self.client_pool)

        for retry in range(self.max_retries + 1):
            try:
                """ start missions for every client """

                print("\nstarting mission for agent #1")
                time.sleep(6)
                self.agent_host1.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              0, experiment_ID)

                print("starting mission for agent #2")
                time.sleep(6)
                self.agent_host2.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              1, experiment_ID)

                print("starting mission for agent #3")
                time.sleep(6)
                self.agent_host3.startMission(self.mission_spec, self.client_pool, self.mission_record_spec,
                                              2, experiment_ID)
                print("\nmissions successfully started.....\n")
                break
            except RuntimeError as e:
                if retry == self.max_retries:
                    logger.error("Error starting mission: " + str(e))
                    raise
                else:
                    logger.warn("Error starting mission: " + str(e))
                    logger.info("Sleeping for %d seconds...", self.retry_sleep)
                    time.sleep(self.retry_sleep)

        logger.info("Waiting for the mission to start.")
        world_state1 = self.agent_host1.getWorldState()
        world_state2 = self.agent_host2.getWorldState()
        while not world_state1.has_mission_begun and world_state2.has_mission_begun:
            time.sleep(0.1)
            world_state1 = self.agent_host1.getWorldState()
            world_state2 = self.agent_host2.getWorldState()
            for error in world_state1.errors and world_state2.errors:
                logger.warn(error.text)

        logger.info("Mission running")

        return self.get_video_frame(world_state1, 1), self.get_video_frame(world_state2, 2)

    def do_action(self, actions, agent_num):
        """
        get next action from action_space
        execute action in environment for the agent
        """
        if len(self.action_spaces) == 1:
            actions = [actions]
        #print(actions)

        if agent_num == 1:
            self.steps_tom += 1
        else:
            self.steps_jerry += 1

        for spc, cmds, acts in zip(self.action_spaces, self.action_names, actions):
            if isinstance(spc, spaces.Discrete):
                logger.debug(cmds[acts])
                if agent_num == 1:
                    print("Tom's next action: ", cmds[acts])
                    self.agent_host1.sendCommand(cmds[acts])
                else:
                    print("Jerry's next action: ", cmds[acts])
                    self.agent_host2.sendCommand(cmds[acts])
            elif isinstance(spc, spaces.Box):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    if agent_num == 1:
                        self.agent_host1.sendCommand(cmd + " " + str(val))
                    else:
                        self.agent_host2.sendCommand(cmd + " " + str(val))
            elif isinstance(spc, spaces.MultiDiscrete):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    if agent_num == 1:
                        self.agent_host1.sendCommand(cmd + " " + str(val))
                    else:
                        self.agent_host2.sendCommand(cmd + " " + str(val))
            else:
                logger.warn("Unknown action space for %s, ignoring." % cmds)

    def get_video_frame(self, world_state, agent_num):
        """
        process video frame for called agent
        RETURN: image for called agent
        """

        if world_state.number_of_video_frames_since_last_state > 0:
            assert len(world_state.video_frames) == 1
            frame = world_state.video_frames[0]
            reshaped = np.zeros((self.video_height * self.video_width * self.video_depth), dtype=np.float32)
            image = np.frombuffer(frame.pixels, dtype=np.int8)
            # print(reshaped.shape)
            for i in range(360000):
                reshaped[i] = image[i]

            image = np.frombuffer(frame.pixels, dtype=np.float32)  # 300x400 = 120000 Werte // np.float32
            image = reshaped.reshape((frame.height, frame.width, frame.channels))  # 300x400x3 = 360000

            if agent_num == 1:
                self.last_image1 = image
            else:
                self.last_image2 = image
        else:
            """ if mission ends befor we got a frame, just take the last frame to reduce exceptions """
            if agent_num == 1:
                image = self.last_image1
            else:
                image = self.last_image2

        return image

    def get_observation(self, world_state):
        """
        check observations during mission run
        RETURN: number of missed observations - if there are any
        """
        if world_state.number_of_observations_since_last_state > 0:
            missed = world_state.number_of_observations_since_last_state - len(
                world_state.observations) - self.skip_steps
            if missed > 0:
                logger.warn("Agent missed %d observation(s).", missed)
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
        saves the failes in results.txt
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

    def save_results(self, overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step):
        """
        saves the results in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("The agents were %i times very close to each other.\n" % (self.too_close_counter))
        datei.write("Reward Tom: %i, Reward Jerry: %i , Time: %f \n\n" % (
            overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step))
        datei.close()

    def get_cell_agents(self):
        """
        gets the cell coordinates for the agents to compare with every 20 seconds
        """
        world_state1 = self.agent_host1.peekWorldState()
        world_state2 = self.agent_host2.peekWorldState()
        msg1 = world_state1.observations[-1].text
        msg2 = world_state2.observations[-1].text
        ob1 = json.loads(msg1)
        ob2 = json.loads(msg2)
        if "cell" in ob1 and "cell" in ob2:
            self.fetched_cell_tom = ob1.get(u'cell', 0)
            self.fetched_cell_jerry = ob2.get(u'cell', 0)
            print("fetched cell tom: ", self.fetched_cell_tom)
            print("fetched cell jerry: ", self.fetched_cell_jerry)

    def get_current_cell_agents(self):
        """
        gets the cell coordinates for the agents at a state
        """
        world_state1 = self.agent_host1.peekWorldState()
        world_state2 = self.agent_host2.peekWorldState()
        msg1 = world_state1.observations[-1].text
        msg2 = world_state2.observations[-1].text
        ob1 = json.loads(msg1)
        ob2 = json.loads(msg2)
        if "cell" in ob1 and "cell" in ob2:
            self.cell_now_tom = ob1.get(u'cell', 0)
            self.cell_now_jerry = ob2.get(u'cell', 0)
            print("current cell tom: ", self.cell_now_tom)
            print("current cell jerry: ", self.cell_now_jerry)

    def get_position_in_arena(self, world_state, time_step):
        """
        get (x,y,z) Positioncoordinates of agent
        fetch the cell coordinates every 20 seconds
        check with current coordinates -> if they are the same more than 20 seconds, it is nearly safe, that the agents
        crashed into each other -> declare mission as failed and end it
        RETURN: x,y,z
        """

        x = y = z = t = 0
        while world_state:
            if len(world_state.observations) >= 1:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                time_now = time.time()

                if time_now - self.time_stamp_start_for_distance > 20:
                    """ fetch cell every 20 seconds """
                    self.get_cell_agents()
                    self.time_stamp_start_for_distance = time.time()

                seconds = time_now - self.time_stamp_start_for_distance
                # print("seconds: ", int(seconds))
                if int(seconds) == 18:
                    self.get_current_cell_agents()
                    if self.fetched_cell_tom == self.cell_now_tom and self.fetched_cell_jerry == self.cell_now_jerry:
                        print("They ran into each other again.")
                        self.append_save_file_with_agents_fail()
                        self.mission_end = True

                if "XPos" in ob and "ZPos" in ob and "YPos" in ob:
                    x = ob[u'XPos']
                    y = ob[u'YPos']
                    z = ob[u'ZPos']
                return x, y, z
            else:
                if t == 5:
                    self.append_save_file_with_fail()
                    self.time_step_agents_ran_into_each_other = time_step
                    self.mission_end = True
                    return x, y, z
                else:
                    time.sleep(1)
                    t += 1
                    print(t)

    def distance(self, time_step):
        """
        check if agents are to near to eachother
        move apart if so
        """

        x1 = y1 = z1 = x2 = y2 = z2 = 0

        """ checks, if world_state is read corrctly, if not, trys again"""
        while (x1 == y1 == z1 == 0) or (x2 == y2 == z2 == 0):
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()

            x1, y1, z1 = self.get_position_in_arena(world_state1, time_step)
            x2, y2, z2 = self.get_position_in_arena(world_state2, time_step)
            print("...")

        # print("  \tTom \tJerry \nX: \t %i\t %i \nY: \t %i\t %i \nZ: \t %i\t %i" % (x1, x2, y1, y2, z1, z2))

        """(x2 == x1+2 and z1 == z1+2) or (x2 == x1+1 and z2 == z1+2) or (x2 == x1 and z2 == z1+2) or \
        (x2 == x1-1 and z2 == z1+2) or (x2 == x1-2 and z2 == z1+2) or (x1 == x2+2 and z1 == z2-2) or \
        (x1 == x2+1 and z1 == z2-2) or (x1 == x2 and z1 == z2-2) or (x1 == x2-1 and z1 == z2-2) or \
        (x1 == x2-2 and z1 == z2-2) or """

        if (x1 == x2 and z1 == z2) or (x2 == x1 + 1 and z2 == z1 + 1) or (x2 == x1 and z2 == z1 + 1) or \
                (x2 == x1 - 1 and z2 == z1 + 1) or (x1 == x2 + 1 and z1 == z2 - 1) or (x1 == x2 and z1 == z2 - 1) or \
                (x1 == x2 - 1 and z1 == z2 - 1):
            print("---------------------------------------------------- stop!! agents too close!")
            self.too_close_counter += 1
            self.agent_host1.sendCommand("movenorth 1")
            self.agent_host2.sendCommand("movesouth 1")

        """(x2 == x1 + 2 and z2 == z1 + 1) or (x2 == x1 + 2 and z2 == z1) or (x2 == x1 + 2 and z2 == z1 - 1) or
        (x1 == x2-2 and z1 == z2+1) or (x1 == x2-2 and z1 == z2) or 
        (x1 == x2-2 and z1 == z2-1) or """

        if (x2 == x1 + 1 and z2 == z1) or (x1 == x2 - 1 and z1 == z2):
            print("---------------------------------------------------- stop!! agents too close!")
            self.too_close_counter += 1
            self.agent_host1.sendCommand("movewest 1")
            self.agent_host2.sendCommand("moveeast 1")

        """(x2 == x1 - 2 and z2 == z1 + 1) or (x2 == x1 - 2 and z2 == z1) or (x2 == x1 - 2 and z2 == z1 - 1) or
        (x1 == x2+2 and z1 == z2+1) or (x1 == x2+2 and z1 == z2) or \
        (x1 == x2+2 and z1 == z2-1) or """

        if (x2 == x1 - 1 and z2 == z1) or (x1 == x2 + 1 and z1 == z2):
            print("---------------------------------------------------- stop!! agents too close!")
            self.too_close_counter += 1
            self.agent_host1.sendCommand("moveeast 1")
            self.agent_host2.sendCommand("movewest 1")

        """(x2 == x1 + 2 and z1 == z1 - 2) or (x2 == x1 + 1 and z2 == z1 - 2) or (x2 == x1 and z2 == z1 - 2) or \
        (x2 == x1 - 1 and z2 == z1 - 2) or (x2 == x1 - 2 and z2 == z1 - 2) or (x1 == x2+2 and z1 == z2+2) or \
        (x1 == x2+1 and z1 == z2+2) or (x1 == x2 and z1 == z2+2) or (x1 == x2-1 and z1 == z2+2) or \
        (x1 == x2-2 and z1 == z2+2) or """

        if (x2 == x1 + 1 and z2 == z1 - 1) or (x2 == x1 and z2 == z1 - 1) or (x2 == x1 - 1 and z2 == z1 - 1) or \
                (x1 == x2 + 1 and z1 == z2 + 1) or (x1 == x2 and z1 == z2 + 1) or (x1 == x2 - 1 and z1 == z2 + 1):
            print("---------------------------------------------------- stop!! agents too close!")
            self.too_close_counter += 1
            self.agent_host1.sendCommand("movesouth 1")
            self.agent_host2.sendCommand("movennorth 1")

    def check_inventory(self, time_step):
        """
        checks, if the agent got the flag in his inventory
        """
        world_state1 = 0
        world_state2 = 0

        x1 = y1 = z1 = x2 = y2 = z2 = 0
        while world_state1 == 0 and world_state2 == 0:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            print("..")

        while not len(world_state1.observations) >= 1 and not len(world_state2.observations) >= 1:
            world_state1 = self.agent_host1.peekWorldState()
            world_state2 = self.agent_host2.peekWorldState()
            print("..")

        if json.dumps(world_state1.observations[-1].text) and json.dumps(world_state2.observations[-1].text):

            msg1 = world_state1.observations[-1].text
            msg2 = world_state2.observations[-1].text
            obs1 = json.loads(msg1)
            obs2 = json.loads(msg2)

            """ checks, if world_state is read corrctly, if not, trys again"""
            while (x1 == y1 == z1 == 0) or (x2 == y2 == z2 == 0):
                world_state1 = self.agent_host1.peekWorldState()
                world_state2 = self.agent_host2.peekWorldState()

                x1, y1, z1 = self.get_position_in_arena(world_state1, time_step)
                x2, y2, z2 = self.get_position_in_arena(world_state2, time_step)
                print("..")

            #if u'inventory' in obs1:

            self.get_current_cell_agents()

            if self.flag_captured_tom and (
                    12 <= x1 <= 15 and 0 <= z1 <= 4):  # "(11,0)" <= self.cell_now_tom < "(14,5)"
                """ 
                if agent reached the target area:
                look down, set block, jump on it to reach wanted position and win the game 
                """
                self.agent_host1.sendCommand("chat I won the game!")
                self.append_save_file_with_finish(time_step, "Tom")
                self.time_step_tom_won = time_step
                self.winner_agent = "Tom"

                self.agent_host1.sendCommand("look 1")
                time.sleep(0.2)
                self.agent_host1.sendCommand("use 1")
                time.sleep(0.2)
                self.agent_host1.sendCommand("jumpmove 1")
                time.sleep(0.2)
                self.agent_host1.sendCommand("look -1")
                self.mission_end = True
            else:
                if self.flag_captured_tom:
                    print("[INFO] Tom holds the flag.")
                else:
                    last_inventory_tom = obs1[u'inventory']
                    inventory_string_tom = json.dumps(last_inventory_tom)
                    # print("Toms last inventory: ", inventory_string_tom)
                    if (inventory_string_tom.find('quartz') != -1):
                        """ tauscht quartz mit log, sodass quartz zurück gelegt werden kann"""
                        if (json.dumps(last_inventory_tom[1]).find('quartz') != -1):
                            self.agent_host1.sendCommand("swapInventoryItems 0 1")
                        self.agent_host1.sendCommand("chat Wrong flag, I'll put it back!")
                        self.agent_host1.sendCommand("use")
                        self.agent_host1.sendCommand("swapInventoryItems 0 1")
                    if (inventory_string_tom.find('log') != -1):
                        self.flag_captured_tom = True
                        self.time_step_tom_captured_the_flag = time_step
                        self.append_save_file_with_flag(time_step, "Tom")
                        print(
                            "----------------------------------------------------------------Tom captured the flag after %i seconds!" % (
                                time_step))

        #if u'inventory' in obs2:

        if self.flag_captured_jerry and (0 <= x2 <= 4 and 11 <= z2 <= 15):
            """ 
                if agent reached the target area:
                look down, set block, jump on it to reach wanted position and win the game 
                """
            self.agent_host2.sendCommand("chat I won the game!")
            self.append_save_file_with_finish(time_step, "Jerry")
            self.time_step_jerry_won = time_step
            self.winner_agent = "Jerry"
            self.agent_host2.sendCommand("look 1")
            time.sleep(0.2)
            self.agent_host2.sendCommand("use 1")
            time.sleep(0.2)
            self.agent_host2.sendCommand("jumpmove 1")
            time.sleep(0.2)
            self.agent_host2.sendCommand("look -1")
            self.mission_end = True
        else:
            if self.flag_captured_jerry:
                print("[INFO] Jerry holds the flag.")
            else:
                last_inventory_jerry = obs2[u'inventory']
                inventory_string_jerry = json.dumps(last_inventory_jerry)
                # print("Jerrys last inventory: ", inventory_string_jerry)
                if (inventory_string_jerry.find('log') != -1):
                    """ tauscht quartz mit log, sodass log zurück gelegt werden kann"""
                    if (json.dumps(last_inventory_jerry[1]).find('log') != -1):
                        self.agent_host2.sendCommand("swapInventoryItems 0 1")
                    self.agent_host2.sendCommand("chat Wrong flag, I'll put it back!")
                    self.agent_host2.sendCommand("use")
                    self.agent_host1.sendCommand("swapInventoryItems 0 1")
                if (inventory_string_jerry.find('quartz') != -1):
                    self.flag_captured_jerry = True
                    self.time_step_jerry_captured_the_flag = time_step
                    self.append_save_file_with_flag(time_step, "Jerry")
                    print(
                        "----------------------------------------------------------------Jerry captured the flag after %i seconds!" % (
                            time_step))

    def sending_mission_quit_commands(self, overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step, obs1, r1,
                                      obs2, r2, outdir, t, tom, jerry, experiment_ID):

        self.agent_host1.sendCommand("quit")
        self.agent_host2.sendCommand("quit")
        self.agent_host3.sendCommand("quit")

        dirname = os.path.join(outdir, 'plots')
        print("dirname: ", dirname)

        """ save and show results of reward calculations """
        self.save_results(overall_reward_agent_Tom, overall_reward_agent_Jerry, time_step)
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

        """ save all the collected data for evaluation graphs """
        self.save_data_for_evaluation_plots(t, time_step, overall_reward_agent_Tom,
                                            overall_reward_agent_Jerry, dirname)
        time.sleep(2)
        """ initialisation for the next episode, reset parameters, build new world """
        t += 1
        self.episode_counter += 1
        r1 = r2 = 0
        done1 = done2 = self.mission_end = False
        overall_reward_agent_Jerry = overall_reward_agent_Tom = 0
        self.save_new_round(t)
        obs1, obs2 = self.reset_world(experiment_ID)
        self.too_close_counter = 0
        self.winner_agent = "-"
        self.time_step_tom_won = self.time_step_jerry_won = None
        self.time_step_tom_captured_the_flag = self.time_step_jerry_captured_the_flag = None
        self.time_step_agents_ran_into_each_other = None
        self.steps_tom = 0
        self.steps_jerry = 0

        """ recover """

        """if evaluator1 and evaluator2 is not None:
            evaluator1.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            evaluator2.evaluate_if_necessary(
                t=t, episodes=episode_idx + 1)
            if (successful_score is not None and
                    evaluator1.max_score >= successful_score and evaluator2.max_score >= successful_score):
                break"""
        return t, obs1, obs2, r1, r2, done1, done2, overall_reward_agent_Jerry, overall_reward_agent_Tom

    def save_data_for_evaluation_plots(self, t, time_step, overall_reward_agent_Tom,
                                       overall_reward_agent_Jerry, dirname):
        """
        t: number of episode
        time_step: duration of the episode
        too_close_counter: how often agents came too close
        overall_reward_agent_Tom, overall_reward_agent_Jerry: reward of the agents
        winner_agent: agent's name who won the episode, if there is no "-"
        time_step_tom_won: timestep, Tom won the game, if not: 0
        time_step_jerry_won: timestep, Jerry won the game, if not: 0
        time_step_tom_captured_the_flag : timestep, Tom captured the flag, if not: 0
        time_step_jerry_captured_the_flag : timestep, Jerry captured the flag, if not: 0
        time_step_agents_ran_into_each_other: timestep; the agents ran into each other and the mission ends

        """
        print("t : ", self.episode_counter)
        if self.episode_counter > 0:
            """ Episode 0 is skipped, because there just starts the initialisation of the world, they do nothing. """
            self.evaluation_agents_ran_into_each_other.append(self.time_step_agents_ran_into_each_other)
            print(self.time_step_agents_ran_into_each_other)
            if self.time_step_agents_ran_into_each_other is None:
                self.evaluation_episode_counter.append(self.episode_counter)
                self.evaluation_episode_time.append(time_step)
                self.evaluation_too_close_counter.append(self.too_close_counter)
                self.evaluation_reward_tom.append(overall_reward_agent_Tom)
                self.evaluation_reward_jerry.append(overall_reward_agent_Jerry)
                self.evaluation_winner_agent.append(self.winner_agent)

                if self.winner_agent == "Tom":
                    self.evaluation_game_won_timestamp.append(self.time_step_tom_won)
                if self.winner_agent == "Jerry":
                    self.evaluation_game_won_timestamp.append(self.time_step_jerry_won)

                self.evaluation_flag_captured_tom.append(self.time_step_tom_captured_the_flag)
                self.evaluation_flag_captured_jerry.append(self.time_step_jerry_captured_the_flag)
                self.evaluation_steps_tom.append(self.steps_tom)
                self.evaluation_steps_jerry.append(self.steps_jerry)

            """ evaluate and print the plots """
            thesis_evaluation_experiment.evaluate(t, self.evaluation_episode_counter, self.evaluation_episode_time,
                                                  self.evaluation_too_close_counter, self.evaluation_reward_tom,
                                                  self.evaluation_reward_jerry,
                                                  self.evaluation_winner_agent, self.evaluation_game_won_timestamp,
                                                  self.evaluation_flag_captured_tom,
                                                  self.evaluation_flag_captured_jerry,
                                                  self.evaluation_agents_ran_into_each_other, dirname,
                                                  self.evaluation_steps_tom, self.evaluation_steps_jerry)
