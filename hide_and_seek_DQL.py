from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
from malmo import MalmoPython
import json
import random
import time
from malmo import malmoutils
import time
import json
import gym
import gym.spaces
import numpy as np
from gym import spaces, error
import xml.etree.ElementTree as ET

import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from past.utils import old_div
from malmo import MalmoPython

import logging
from malmo import malmoutils

malmoutils.fix_print()

# initalize two agents
agent_host1 = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host1)
recordingsDirectory1 = malmoutils.get_recordings_directory(agent_host1)

agent_host2 = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host2)
recordingsDirectory2 = malmoutils.get_recordings_directory(agent_host2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# client pool
client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10001))

MalmoPython.setLogging("", MalmoPython.LoggingSeverityLevel.LOG_OFF)

malmoutils.parse_command_line(agent_host1)
malmoutils.parse_command_line(agent_host2)

# for video processing
current_yaw_delta_from_depth = 0
video_width = 700
video_height = 500

# unnecessary by now
if sys.version_info[0] == 2:
    import Tkinter as tk
else:
    import tkinter as tk


# 'processFrame' from depth_map_runner.py example
def processFrame(frame):
    '''Track through the middle line of the depth data and find the max discontinuities'''
    global current_yaw_delta_from_depth

    y = int(old_div(video_height, 2))
    rowstart = y * video_width

    v = 0
    v_max = 0
    v_max_pos = 0
    v_min = 0
    v_min_pos = 0

    dv = 0
    dv_max = 0
    dv_max_pos = 0
    dv_max_sign = 0

    d2v = 0
    d2v_max = 0
    d2v_max_pos = 0
    d2v_max_sign = 0

    for x in range(0, video_width):
        nv = frame[(rowstart + x) * 4 + 3]
        ndv = nv - v
        nd2v = ndv - dv

        if nv > v_max or x == 0:
            v_max = nv
            v_max_pos = x

        if nv < v_min or x == 0:
            v_min = nv
            v_min_pos = x

        if abs(ndv) > dv_max or x == 1:
            dv_max = abs(ndv)
            dv_max_pos = x
            dv_max_sign = ndv > 0

        if abs(nd2v) > d2v_max or x == 2:
            d2v_max = abs(nd2v)
            d2v_max_pos = x
            d2v_max_sign = nd2v > 0

        d2v = nd2v
        dv = ndv
        v = nv

    logger.info("d2v, dv, v: " + str(d2v) + ", " + str(dv) + ", " + str(v))

    # We want to steer towards the greatest d2v (ie the biggest discontinuity in the gradient of the depth map).
    # If it's a possitive value, then it represents a rapid change from close to far - eg the left-hand edge of a gap.
    # Aiming to put this point in the leftmost quarter of the screen will cause us to aim for the gap.
    # If it's a negative value, it represents a rapid change from far to close - eg the right-hand edge of a gap.
    # Aiming to put this point in the rightmost quarter of the screen will cause us to aim for the gap.
    if dv_max_sign:
        edge = old_div(video_width, 4)
    else:
        edge = 3 * video_width / 4

    # Now, if there is something noteworthy in d2v, steer according to the above comment:
    if d2v_max > 8:
        current_yaw_delta_from_depth = (old_div(float(d2v_max_pos - edge), video_width))
    else:
        # Nothing obvious to aim for, so aim for the farthest point:
        if v_max < 255:
            current_yaw_delta_from_depth = (old_div(float(v_max_pos), video_width)) - 0.5
        else:
            # No real data to be had in d2v or v, so just go by the direction we were already travelling in:
            if current_yaw_delta_from_depth < 0:
                current_yaw_delta_from_depth = -1
            else:
                current_yaw_delta_from_depth = 1


# replay buffer to remember latest actions
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype = np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# DQN
def build_dqn(lr, n_actions, input_dims, fcl_dims, fc2_dims):
    model = Sequential([
        Dense(fcl_dims, input_shape = (input_dims, )),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model


# Agent
class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname = 'dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete = True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis = 1)*done
        _ = self.q_eval.fit(state, q_target, verbose=0)
        self.epsilon = self.epsilon*self.epsilon_dec if self. epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

    # process the video frame input
    def process_video_input(self, world_state):
        INPUT_SHAPE = (84, 84)
        WINDOW_LENGTH = 4

        if world_state.number_of_video_frames_since_last_state > 0:
            assert len(world_state.video_frames) == 1
            frame = world_state.video_frames[0]
            image = np.frombuffer(frame.pixels, dtype=np.uint8)
            # reshape frame for faster processing
            image = image.reshape((frame.height, frame.width, frame.channels))
            # logger.debug(image)
            self.last_image = image
        else:
            image = self.last_image

        # grayscale for better processing
        # tf.image.rgb_to_grayscale(image, name=None)

        return image

    # from gym
    def _take_action(self, actions):
        # if there is only one action space, it wasn't wrapped in Tuple
        if len(self.action_spaces) == 1:
            actions = [actions]

        # send appropriate command for different actions
        for spc, cmds, acts in zip(self.action_spaces, self.action_names, actions):
            if isinstance(spc, spaces.Discrete):
                logger.debug(cmds[acts])
                self.agent_host.sendCommand(cmds[acts])
            elif isinstance(spc, spaces.Box):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    self.agent_host.sendCommand(cmd + " " + str(val))
            elif isinstance(spc, spaces.MultiDiscrete):
                for cmd, val in zip(cmds, acts):
                    logger.debug(cmd + " " + str(val))
                    self.agent_host.sendCommand(cmd + " " + str(val))
            else:
                logger.warn("Unknown action space for %s, ignoring." % cmds)

    # from gym
    def _get_world_state(self):
        # wait till we have got at least one observation or mission has ended
        while True:
            time.sleep(self.step_sleep)  # wait for 1ms to not consume entire CPU
            world_state = self.agent_host.peekWorldState()
            if world_state.number_of_observations_since_last_state > self.skip_steps or not world_state.is_mission_running:
                break

        return self.agent_host.getWorldState()

    # from gym
    def _get_video_frame(self, world_state):
        # process the video frame
        if world_state.number_of_video_frames_since_last_state > 0:
            assert len(world_state.video_frames) == 1
            frame = world_state.video_frames[0]
            image = np.frombuffer(frame.pixels, dtype=np.uint8)
            image = image.reshape((frame.height, frame.width, frame.channels))
            #logger.debug(image)
            self.last_image = image
        else:
            # can happen only when mission ends before we get frame
            # then just use the last frame, it doesn't matter much anyway
            image = self.last_image

        return image

    # from gym
    def _get_observation(self, world_state):
        if world_state.number_of_observations_since_last_state > 0:
            missed = world_state.number_of_observations_since_last_state - len(world_state.observations) - self.skip_steps
            if missed > 0:
                logger.warn("Agent missed %d observation(s).", missed)
            assert len(world_state.observations) == 1
            return json.loads(world_state.observations[0].text)
        else:
            return None

    # from gym
    def step(self, action):
        # take the action only if mission is still running
        world_state = self.agent_host.peekWorldState()
        if world_state.is_mission_running:
            # take action
            self._take_action(action)
        # wait for the new state
        world_state = self._get_world_state()

        # log errors and control messages
        for error in world_state.errors:
            logger.warn(error.text)
        for msg in world_state.mission_control_messages:
            logger.debug(msg.text)
            root = ET.fromstring(msg.text)
            if root.tag == '{http://ProjectMalmo.microsoft.com}MissionEnded':
                for el in root.findall('{http://ProjectMalmo.microsoft.com}HumanReadableStatus'):
                    logger.info("Mission ended: %s", el.text)

        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()

        # take the last frame from world state
        image = self._get_video_frame(world_state)

        # detect terminal state
        done = not world_state.is_mission_running

        # other auxiliary data
        info = {}
        info['has_mission_begun'] = world_state.has_mission_begun
        info['is_mission_running'] = world_state.is_mission_running
        info['number_of_video_frames_since_last_state'] = world_state.number_of_video_frames_since_last_state
        info['number_of_rewards_since_last_state'] = world_state.number_of_rewards_since_last_state
        info['number_of_observations_since_last_state'] = world_state.number_of_observations_since_last_state
        info['mission_control_messages'] = [msg.text for msg in world_state.mission_control_messages]
        info['observation'] = self._get_observation(world_state)

        return image, reward, done, info

    def run(self):

        total_reward = 0

        done = False
        score = 0

        while not done:
            action = agent1.choose_action(observation)
            observation_, reward, done, info = Agent.step(action)
            score += reward
            agent1.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent1.learn()

        # save the epsilon
        eps_history.append(agent1.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0,i-100):(i+1)])
        print('episode', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent1.save_model()

        return scores


# main
if __name__ == '__main__':

    max_retries = 3
    num_repeats = 10

    agent1 = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    agent2 = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.01)

    scores = []
    eps_history = []

    # load the mission file
    mission_file = 'hide_and_seek_xml_mission_DQL.xml'
    with open(mission_file, 'r') as f:
        print("Mission file used: %s" % mission_file)
        mission_xml = f.read()
        xml_mission = MalmoPython.MissionSpec(mission_xml, True)

    cumulative_rewards1 = []
    cumulative_rewards2 = []

    agent_01_recording_spec = MalmoPython.MissionRecordSpec()
    agent_02_recording_spec = MalmoPython.MissionRecordSpec()
    role1 = 0
    role2 = 1
    experimentID1 = ''
    experimentID2 = ''

    agent_host1.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host1.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

    agent_host2.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host2.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

    print(recordingsDirectory1)
    print(recordingsDirectory2)

    if recordingsDirectory1 and recordingsDirectory2:
            print("test 1")
            agent_01_recording_spec.recordRewards()
            agent_01_recording_spec.recordObservations()
            agent_01_recording_spec.recordCommands()
            agent_02_recording_spec.recordRewards()
            agent_02_recording_spec.recordObservations()
            agent_02_recording_spec.recordCommands()
            print("test 1 nachher")
            if agent_host1.receivedArgument("record_video") and agent_host2.receivedArgument("record_video"):
                print("test 2")
                agent_01_recording_spec.recordMP4(24, 2000000)
                agent_02_recording_spec.recordMP4(24, 2000000)
                print("test 2 nachher")

    for i in range(num_repeats):

        print()
        print('Repeat %d of %d' % (i+1, num_repeats))
        n_actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

        for retry in range(max_retries):
            if recordingsDirectory1 and recordingsDirectory2:
                print("test 3")
                agent_01_recording_spec.setDestination(recordingsDirectory1 + "//" + "Mission1_" + str(retry + 1) + ".tgz")
                agent_02_recording_spec.setDestination(recordingsDirectory2 + "//" + "Mission2_" + str(retry + 1) + ".tgz")
            max_retries = 3
            for retry in range(max_retries):
                try:

                    agent_host1.startMission(xml_mission, client_pool, agent_01_recording_spec, role1, experimentID1)
                    time.sleep(10)
                    print("test mission start 1")

                    agent_host2.startMission(xml_mission, client_pool, agent_02_recording_spec, role2, experimentID2)
                    time.sleep(10)
                    print("test mission start 2")
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        logger.error("Error starting mission: %s" % e)
                        exit(1)
                    else:
                        time.sleep(2)

        # world state
        print("Mission will start soon........", end=' ')
        world_state1 = agent_host1.getWorldState()
        world_state2 = agent_host2.getWorldState()
        print("world_state 1: ", world_state1)
        print("world_state 2: ", world_state2)

        while not world_state1.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state1 = agent_host1.getWorldState()

            for error in world_state1.errors:
                print("Error:", error.text)
        print()

        # Videostream processing
        while world_state1.is_mission_running and world_state2.is_mission_running:
            world_state1 = agent_host1.getWorldState()
            world_state2 = agent_host1.getWorldState()
            while world_state1.number_of_video_frames_since_last_state < 1 and \
                    world_state2.number_of_video_frames_since_last_state < 1 and \
                    world_state1.is_mission_running and world_state2.is_mission_running:
                logger.info("Waiting for frames...")
                time.sleep(0.05)
                world_state1 = agent_host1.getWorldState()
                world_state2 = agent_host1.getWorldState()

            logger.info("Got frame!")

            if world_state1.is_mission_running and world_state2.is_mission_running:
                processFrame(world_state1.video_frames[0].pixels)
                processFrame(world_state2.video_frames[0].pixels)

                agent_host1.sendCommand("turn " + str(current_yaw_delta_from_depth))
                agent_host2.sendCommand("turn " + str(current_yaw_delta_from_depth))

        # run the agents in the world, reward at first just for the seeker
        cumulative_reward1 = agent1.run(agent_host1)
        cumulative_reward2 = agent2.run(agent_host2)
        print('Cumulative reward: %d' % cumulative_reward1)
        # print('Cumulative reward: %d' % cumulative_reward2)
        cumulative_rewards1 += [cumulative_reward1]
        # cumulative_rewards2 += [cumulative_reward2]

        time.sleep(0.5)

    print("Successfully done.")

    print()
    print("Cumulative rewards for Jerry:" % num_repeats)
    print(cumulative_rewards1)
    # print("Cumulative rewards for all %d runs of agent_02:" % num_repeats)
    # print(cumulative_rewards2)
