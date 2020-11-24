import json
import logging

import time

import gym
import minecraft_py
from gym import spaces
import logging
from malmo import MalmoPython
import numpy as np
import os
import malmoutils

import argparse
import os
import sys

from chainer import optimizers

import chainerrl
from chainerrl.agents.dqn import DQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer

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
    _create_action_space()
    load_mission_file(mission_file)
    load_mission_xml(mission_xml)
    clip_action_filter(a)
    dqn_q_values_and_neuronal_net(args, action_space, obs_size, obs_space)
    remember(buf, state, action, reward, new_state, done)
    step(action, agent_num)
    safeStartMission(agent_host, mission, client_pool, recording, role, experimentId)
    safeWaitForStart(agent_hosts)
    reset()
    render_first_agent(mode='human', close=False)
    render_second_agent(mode='human', close=False)
    _close()
    _seed(see=None)
    _take_action(actions, agent_num)
    _get_world_state()
    _get_video_frame(world_state, agent_num)
    _get_observation(world_state)
"""


class ThesisEnvExperiment(gym.Env):
    """
    initialize agents and give commandline permissions
    """
    metadata = {'render.modes': ['human']}
    agent_host1 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host1)
    agent_host2 = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host2)

    def __init__(self):
        super(ThesisEnvExperiment, self).__init__()
        """
        load the mission file
        format: XML
        """
        mission_file = 'capture_the_flag_xml_mission_DQL.xml'
        self.load_mission_file(mission_file)
        print("Mission loaded: Capture the Flag")

        self.client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
        self.mc_process = None
        self.screen = None


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

        if allowContinuousMovement or allowDiscreteMovement or allowAbsoluteMovement:
            # if there are any parameters, remove current command handlers first
            self.mission_spec.removeAllCommandHandlers()

            if allowContinuousMovement is True:
                self.mission_spec.allowAllContinuousMovementCommands()
            elif isinstance(allowContinuousMovement, list):
                for cmd in allowContinuousMovement:
                    self.mission_spec.allowContinuousMovementCommand(cmd)

            if allowDiscreteMovement is True:
                self.mission_spec.allowAllDiscreteMovementCommands()
            elif isinstance(allowDiscreteMovement, list):
                for cmd in allowDiscreteMovement:
                    self.mission_spec.allowDiscreteMovementCommand(cmd)

            if allowAbsoluteMovement is True:
                self.mission_spec.allowAllAbsoluteMovementCommands()
            elif isinstance(allowAbsoluteMovement, list):
                for cmd in allowAbsoluteMovement:
                    self.mission_spec.allowAbsoluteMovementCommand(cmd)

        if start_minecraft:
            # start Minecraft process assigning port dynamically
            self.mc_process, port = minecraft_py.start()
            logger.info("Started Minecraft on port %d, overriding client_pool.", port)
            client_pool = [('127.0.0.1', port)]

        if client_pool:
            # change format of the client_pool to struct
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
        self.last_image1 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.int8)
        self.last_image2 = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.int8)
        self._create_action_space()

        # mission recording
        self.mission_record_spec = MalmoPython.MissionRecordSpec()  # record nothing
        if recordDestination:
            self.mission_record_spec.setDestination(recordDestination)
        if recordRewards:
            self.mission_record_spec.recordRewards()
        if recordCommands:
            self.mission_record_spec.recordCommands()
        if recordMP4:
            self.mission_record_spec.recordMP4(*recordMP4)

        if gameMode:
            if gameMode == "spectator":
                self.mission_spec.setModeToSpectator()
            elif gameMode == "creative":
                self.mission_spec.setModeToCreative()
            elif gameMode == "survival":
                logger.warn("Cannot force survival mode, assuming it is the default.")
            else:
                assert False, "Unknown game mode: " + gameMode

    def _create_action_space(self):
        """
        create action_space from action_names to dynamically generate the needed movement
        format: Discrete
        """
        # collect different actions based on allowed commands
        unused_actions = []
        discrete_actions = []
        chs = self.mission_spec.getListOfCommandHandlers(0)
        for ch in chs:
            cmds = self.mission_spec.getAllowedCommands(0, ch)
            for command in cmds:
                logger.debug(ch + ":" + command)
                if command in ["movenorth", "movesouth", "moveeast", "movewest", "attack"]:
                    discrete_actions.append(command + " 1")
                    discrete_actions.append(command + " -1")
                else:
                    unused_actions.append(
                        command)  # "move", "jumpmove", "strafe", "jumpstrafe", "turn", "jumpnorth", "jumpsouth", "jumpwest", "jumpeast","look", "use", "jumpuse", "sleep", "movenorth", "movesouth", "moveeast", "movewest", "jump", "attack"
                    # raise ValueError("Unknown continuous action: " + command)

        # turn action lists into action spaces
        self.action_names = []
        self.action_spaces = []
        if len(discrete_actions) > 0:
            self.action_spaces.append(spaces.Discrete(len(discrete_actions)))
            self.action_names.append(discrete_actions)
            #print("action_names:  ", self.action_names)
            #print("action_spaces: ", self.action_spaces)

        # if there is only one action space, don't wrap it in Tuple
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
            #print("n_actions: ", n_actions)
            q_func = q_functions.FCStateQFunctionWithDiscreteAction(
                obs_size, n_actions,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers)
            #print("q_func ", q_func)
            # Use epsilon-greedy for exploration
            explorer = explorers.LinearDecayEpsilonGreedy(
                args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
                action_space.sample)
            #print("explorer: ", explorer)

        if args.noisy_net_sigma is not None:
            links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
            # Turn off explorer
            explorer = explorers.Greedy()
        #print("obs_space.low : ", obs_space.shape)
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

        #print("--q_func: ", q_func)
        #print("--opt: ", opt)
        #print("--rbuf: ", rbuf)
        #print("--explorer: ", explorer)
        return q_func, opt, rbuf, explorer


    def step_generating(self, action, agent_num):
        """
        time step in arena
        next action is executed
        reward of actual state is calculated and summed up with the overall reward
        RETURN: image, reward, done, info
        """
        world_state1 = self.agent_host1.peekWorldState()

        world_state2 = self.agent_host2.peekWorldState()
        if agent_num == 1:
            if world_state1.is_mission_running:
                # take action
                self._take_action(action, agent_num)
                # wait for the new state
            world_state = self.agent_host1.getWorldState()
        else:
            if world_state2.is_mission_running:
                # take action
                self._take_action(action, agent_num)
                # wait for the new state
            world_state = self.agent_host2.getWorldState()

        reward1 = 0
        reward2 = 0
        # calculate reward of current state
        if agent_num == 1:

            # self.agent_host1.sendCommand(action)
            time.sleep(0.1)
            for r in world_state1.rewards:
                reward1 += r.getValue()
        else:
            # self.agent_host2.sendCommand(action)
            time.sleep(0.1)
            for r in world_state2.rewards:
                reward2 += r.getValue()

        # take the last frame from world state
        if agent_num == 1:
            image1 = self._get_video_frame(world_state1, 1)
        else:
            image2 = self._get_video_frame(world_state2, 2)

        """
        'done'-flag indicated, if mission is still running 
        """
        if agent_num == 1:
            done1 = not world_state1.is_mission_running
        else:
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
            info1['observation'] = self._get_observation(world_state1)

        else:
            info2 = {}
            info2['has_mission_begun'] = world_state2.has_mission_begun
            info2['is_mission_running'] = world_state2.is_mission_running
            info2['number_of_video_frames_since_last_state'] = world_state2.number_of_video_frames_since_last_state
            info2['number_of_rewards_since_last_state'] = world_state2.number_of_rewards_since_last_state
            info2['number_of_observations_since_last_state'] = world_state2.number_of_observations_since_last_state
            info2['mission_control_messages'] = [msg.text for msg in world_state2.mission_control_messages]
            info2['observation'] = self._get_observation(world_state2)

        if agent_num == 1:
            return image1, reward1, done1, info1
        else:
            return image2, reward2, done2, info2

    def safeStartMission(self, agent_host, mission, client_pool, recording, role, experimentId):
        """
        safe start to provide a safe initialization
        wait for slow clients
        """
        used_attempts = 0
        max_attempts = 5
        print("Calling startMission for role", role)
        while True:
            try:
                agent_host.startMission(mission, client_pool, recording, role, experimentId)
                break
            except MalmoPython.MissionException as e:
                errorCode = e.details.errorCode
                if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                    print("Server not quite ready yet - waiting...")
                    time.sleep(2)
                elif errorCode == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                    print("Not enough available Minecraft instances running.")
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        print("Will wait in case they are starting up.", max_attempts - used_attempts, "attempts left.")
                        time.sleep(2)
                elif errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                    print("Server not found - has the mission with role 0 been started yet?")
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        print("Will wait and retry.", max_attempts - used_attempts, "attempts left.")
                        time.sleep(2)
                else:
                    print("Waiting will not help here - bailing immediately.")
                    exit(1)
            if used_attempts == max_attempts:
                print("All chances used up - bailing now.")
                exit(1)
        print("startMission called okay.")
        print()

    def safeWaitForStart(self, agent_hosts):
        """
        function must be placed after safeStartMission
        wait for slow client connectins
        """
        print("Waiting for the mission to start", end=' ')
        start_flags = [False for a in agent_hosts]
        start_time = time.time()
        time_out = 120  # Allow two minutes for mission to start.
        while not all(start_flags) and time.time() - start_time < time_out:
            states = [a.getWorldState() for a in agent_hosts]
            start_flags = [w.has_mission_begun for w in states]
            errors = [e for w in states for e in w.errors]
            if len(errors) > 0:
                print("Errors waiting for mission start:")
                for e in errors:
                    print(e.text)
                print("Bailing now.")
                exit(1)
            time.sleep(0.1)
            print(".", end=' ')
        print()
        if time.time() - start_time >= time_out:
            print("Timed out waiting for mission to begin. Bailing.")
            exit(1)
        print("Mission has started.")
        print()


    def reset_world(self):
        """
        reset the arena and start the missions per agent
        """
        print("force world reset........")
        if self.forceWorldReset:
            self.mission_spec.forceWorldReset()
        # this seemed to increase probability of success in first try
        time.sleep(0.1)
        # Attempt to start a mission
        print(self.client_pool)

        for retry in range(self.max_retries + 1):
            
            try:
                print("starting mission........")
                self.safeStartMission(self.agent_host1, self.mission_spec, self.client_pool, self.mission_record_spec,
                                      0, "experiment_id")

                time.sleep(4)
                self.safeStartMission(self.agent_host2, self.mission_spec, self.client_pool, self.mission_record_spec,
                                      1, "experiment_id")
                time.sleep(3)
                agent_hosts = [self.agent_host1, self.agent_host2]
                self.safeWaitForStart(agent_hosts)
                print("mission successfully started.....")
                break
            except RuntimeError as e:
                if retry == self.max_retries:
                    logger.error("Error starting mission: " + str(e))
                    raise
                else:
                    logger.warn("Error starting mission: " + str(e))
                    logger.info("Sleeping for %d seconds...", self.retry_sleep)
                    time.sleep(self.retry_sleep)

        # Loop until mission starts:
        logger.info("Waiting for the mission to start")
        world_state1 = self.agent_host1.getWorldState()
        world_state2 = self.agent_host2.getWorldState()
        while not world_state1.has_mission_begun and world_state2.has_mission_begun:
            time.sleep(0.1)
            world_state1 = self.agent_host1.getWorldState()
            world_state2 = self.agent_host2.getWorldState()
            for error in world_state1.errors and world_state2.errors:
                logger.warn(error.text)

        logger.info("Mission running")
        return self._get_video_frame(world_state1, 1), self._get_video_frame(world_state2, 2)

    def render_first_agent(self, mode='human', close=False):
        """
        render first agent
        update pygame window for every taken step
        """
        reshaped_pic = np.array(self.last_image1, dtype=float)
        reshaped_picture_01 = reshaped_pic[:, :, 0:3] # 3 dimensions
        if mode == 'rgb_array':
            return reshaped_picture_01
        elif mode == 'human':
            #print(self.last_image1.shape)
            try:
                import pygame
            except ImportError as e:
                raise logging.error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))

            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    pygame_width = self.video_width * 2
                    self.screen = pygame.display.set_mode((pygame_width, self.video_height))
                img1 = pygame.surfarray.make_surface(reshaped_picture_01.swapaxes(0, 1))
                #img1 = pygame.surfarray.make_surface(self.last_image1.swapaxes(0, 1))
                self.screen.blit(img1, (0, 0))
                pygame.display.update()
        else:
            raise logging.error.UnsupportedMode("Unsupported render mode: " + mode)

    def render_second_agent(self, mode='human', close=False):
        """
        render second agent
        update pygame window for every taken step
        """
        reshaped_picture = np.array(self.last_image2, dtype=float)
        reshaped_picture_02 = reshaped_picture[:, :, 0:3] # 3 dimensions
        if mode == 'rgb_array':
            return reshaped_picture_02
        elif mode == 'human':
            #print(self.last_image2.shape)
            try:
                import pygame
            except ImportError as e:
                raise logging.error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))

            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    pygame_width = self.video_width * 2
                    self.screen = pygame.display.set_mode((pygame_width, self.video_height))
                img2 = pygame.surfarray.make_surface(reshaped_picture_02.swapaxes(0, 1))
                #img2 = pygame.surfarray.make_surface(self.last_image2.swapaxes(0, 1))
                self.screen.blit(img2,
                                 (self.video_width, 0))  # second window is rendered beside first window, same thread
                pygame.display.update()
        else:
            raise logging.error.UnsupportedMode("Unsupported render mode: " + mode)

    def render_video_frame_picture(self, world_state, mode='human', close=False):
        """
        render first agent
        update pygame window for every taken step
        """
        frame = world_state.video_frames[0]
        reshaped_pic = np.array(frame, dtype=float)
        reshaped_picture_01 = reshaped_pic[:, :, 0:3] # 3 dimensions


        if mode == 'human':
            try:
                import pygame
            except ImportError as e:
                raise logging.error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))

            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    pygame_width = self.video_width * 2
                    self.screen = pygame.display.set_mode((pygame_width, self.video_height))
                img1 = pygame.surfarray.make_surface(reshaped_picture_01.swapaxes(0, 1))
                #img1 = pygame.surfarray.make_surface(self.last_image1.swapaxes(0, 1))
                self.screen.blit(img1, (0, 0))
                pygame.display.update()
        else:
            raise logging.error.UnsupportedMode("Unsupported render mode: " + mode)

    def _close(self):
        if hasattr(self, 'mc_process') and self.mc_process:
            minecraft_py.stop(self.mc_process)

    def _seed(self, seed=None):
        self.mission_spec.setWorldSeed(str(seed))
        return [seed]

    def _take_action(self, actions, agent_num):
        """
        calculate next action from action_space
        execute action in environment for the agent
        """
        # if there is only one action space, it wasn't wrapped in Tuple
        if len(self.action_spaces) == 1:
            actions = [actions]
        print(actions)
        # send appropriate command for different actions
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

    def _get_world_state(self):
        """
        see, if mission is running and calculate world_state
        RETURN: world_state for called agent_host
        """
        # wait till we have got at least one observation or mission has ended
        while True:
            time.sleep(self.step_sleep)  # wait for 1ms to not consume entire CPU
            world_state = self.agent_host.peekWorldState()
            if world_state.number_of_observations_since_last_state > self.skip_steps or not world_state.is_mission_running:
                break

        return self.agent_host.getWorldState()


    def _get_video_frame(self, world_state, agent_num):
        """
        process video frame for called agent
        RETURN: image for called agent
        """

        if world_state.number_of_video_frames_since_last_state > 0:
            assert len(world_state.video_frames) == 1
            frame = world_state.video_frames[0]
            #print("frame width: ", frame.width)
            #print("frame height: ", frame.height)
            #print("frame channels: ", frame.channels)
            #dt = np.dtype(float)
            #dt = dt.newbyteorder('>')
            #image = np.array([300, 400, 4], dtype=float)
            reshaped = np.zeros((self.video_height * self.video_width * self.video_depth), dtype=np.float32)
            image = np.frombuffer(frame.pixels, dtype=np.int8)
            #print(reshaped.shape)
            for i in range(360000):
                reshaped[i] = image[i]

            #reshaped_picture_02 = reshaped_picture[:, :, 0:3]  # 3 dimensions
            #print("reshaped: ", reshaped)
            image = np.frombuffer(frame.pixels, dtype=np.float32) # 300x400 = 120000 Werte // np.float32
            image = reshaped.reshape((frame.height, frame.width, frame.channels)) # 300x400x3 = 360000
            # logger.debug(image)
            if agent_num == 1:
                self.last_image1 = image
            else:
                self.last_image2 = image
        else:
            # can happen only when mission ends before we get frame
            # then just use the last frame, it doesn't matter much anyway
            if agent_num == 1:
                image = self.last_image1
            else:
                image = self.last_image2

        return image

    def _get_observation(self, world_state):
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

    def save_results(self, t, overall_reward_agent_Tom, overall_reward_agent_Jerry):
        """
        save the results in results.txt
        """
        datei = open('results.txt', 'a')
        datei.write("-------------- ROUND %i --------------\n" % (t))
        datei.write("Reward Tom: %i, Reward Jerry: %i \n\n" % (overall_reward_agent_Tom, overall_reward_agent_Jerry))
        datei.close()

