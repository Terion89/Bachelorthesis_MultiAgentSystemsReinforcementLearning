import matplotlib
import time

import os

from matplotlib import pyplot as plt
import pandas as p
import numpy as np

from matplotlib.pyplot import xticks


def evaluate(t, evaluation_episode_counter, evaluation_episode_time,
             evaluation_too_close_counter, evaluation_reward_tom, evaluation_reward_jerry,
             evaluation_winner_agent, evaluation_game_won_timestamp, evaluation_flag_captured_tom,
             evaluation_flag_captured_jerry, evaluation_agents_ran_into_each_other, dirname, evaluation_steps_tom,
             evaluation_steps_jerry):
    """
    graph plots to visualize the result
    available data:

    evaluation_episode_counter = []             - counts the episodes for evaluation
    evaluation_episode_time = []                - episode duration per episode
    evaluation_too_close_counter = []           - how often they came close per episode
    evaluation_reward_tom = []                  - reward Tom per episode
    evaluation_reward_jerry = []                - reward Jerry per episode
    evaluation_winner_agent = []                - string, who won per episode
    evaluation_game_won_timestamp = []          - timestamp, somebody won per episode
    evaluation_flag_captured_tom = []           - timestamp tom captured the flag per episode
    evaluation_flag_captured_jerry = []         - timestamp jerry captured the flag per episode
    evaluation_agents_ran_into_each_other = []  - timestamp, the agents ran into each other per episode
    dirname: place to save the plots


    graph_01: episodes where somebody won the game
        x: episodes
        y: time in sec
           values: overall time, flag_captured_tom, flag_captured_jerry, game_won_timestep + who won

    graph_02:
        x: episodes
        y: time in sec
           values: timestamp, they ran into each other or ended the mission

    graph_03: episodes where somebody won the game
        x: episodes
        y: counts
           values: too_close_counter (number of how many times they were close to each other)

    graph_04: episodes where somebody won the game
        x: episodes
        y: rewards
    """
    """ font manipulation """
    """ font manipulation """
    matplotlib.rcParams.update({'font.size': 4})

    """ generate numpy-arrays """
    np_evaluation_episode_counter = np.asarray(evaluation_episode_counter)
    np_evaluation_episode_time = np.asarray(evaluation_episode_time)
    np_evaluation_too_close_counter = np.asarray(evaluation_too_close_counter)
    np_evaluation_reward_tom = np.asarray(evaluation_reward_tom)
    np_evaluation_reward_jerry = np.asarray(evaluation_reward_jerry)
    np_evaluation_game_won_timestamp = np.asarray(evaluation_game_won_timestamp)
    np_evaluation_flag_captured_tom = np.asarray(evaluation_flag_captured_tom)
    np_evaluation_flag_captured_jerry = np.asarray(evaluation_flag_captured_jerry)
    np_evaluation_agents_ran_into_each_other = np.asarray(evaluation_agents_ran_into_each_other)
    np_evaluation_steps_tom = np.asarray(evaluation_steps_tom)
    np_evaluation_steps_jerry = np.asarray(evaluation_steps_jerry)

    """ set name of plot to not override it """
    random_name = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())

    """ make dir, if it doesn't already exist """
    os.makedirs(dirname, exist_ok=True)

    plt.title("Episode results")

    """ create dataframes """
    df_evaluation_episode_time = p.DataFrame({'x': np_evaluation_episode_time})
    df_evaluation_too_close_counter = p.DataFrame({'x': np_evaluation_too_close_counter})
    df_evaluation_agents_ran_into_each_other = p.DataFrame({'x': np_evaluation_agents_ran_into_each_other})
    df_evaluation_reward_tom = p.DataFrame({'x': np_evaluation_reward_tom})
    df_evaluation_reward_jerry = p.DataFrame({'x': np_evaluation_reward_jerry})
    df_evaluation_game_won_timestamp = p.DataFrame({'x': np_evaluation_game_won_timestamp})
    df_evaluation_flag_captured_tom = p.DataFrame({'x': np_evaluation_flag_captured_tom})
    df_evaluation_flag_captured_jerry = p.DataFrame({'x': np_evaluation_flag_captured_jerry})
    df_evaluation_steps_tom = p.DataFrame({'x': np_evaluation_steps_tom})
    df_evaluation_steps_jerry = p.DataFrame({'x': np_evaluation_steps_jerry})

    """ graph 1 """
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('time in seconds')

    ax1.plot('x', 'y', data=df_evaluation_episode_time, marker='x', linestyle='-', color="red", alpha=0.8,
             label="relevant game time")
    ax1.plot('x', 'y', data=df_evaluation_game_won_timestamp, marker=',', linestyle='-', color="yellow", alpha=0.8,
             label="relevant game won after x seconds")
    ax1.plot('x', 'y', data=df_evaluation_flag_captured_tom, marker=',', linestyle='-', color="cornflowerblue",
             alpha=0.8,
             label="timestamp tom captured the flag")
    ax1.plot('x', 'y', data=df_evaluation_flag_captured_jerry, marker=',', linestyle='-', color="springgreen",
             alpha=0.8,
             label="timestamp jerry captured the flag")
    ax1.plot('x', 'y', data=df_evaluation_steps_tom, marker='x', linestyle='-', color="mediumslateblue", alpha=0.8,
             label="number of actions tom")
    ax1.plot('x', 'y', data=df_evaluation_steps_jerry, marker=',', linestyle='-', color="aquamarine", alpha=0.8,
             label="number of actions jerry")
    plt.legend(loc='best')
    locs, labels = xticks()
    xticks(np.arange(t), np_evaluation_episode_counter)
    ax11 = ax1.twiny()
    ax11.plot(range(t), np.ones(t))
    ax11.set_xlabel("winner of the episode")
    locs, labels = xticks()
    xticks(np.arange(t), evaluation_winner_agent)
    plt.grid(True)

    """ graph 2 """
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('time_in_seconds')
    ax2.plot('x', 'y', data=df_evaluation_agents_ran_into_each_other, marker='_', linestyle=':', color="red", alpha=0.8,
             label="time, agents ran into each other")
    plt.legend(loc='best')
    plt.grid(True)

    """ graph 3 """
    ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=1)
    ax3.set_xlabel('episodes')
    ax3.set_ylabel('#agents_were_close')
    locs, labels = xticks()
    xticks(np.arange(t), np_evaluation_episode_counter)
    ax3.plot('x', 'y', data=df_evaluation_too_close_counter, marker=',', linestyle='-', color="blue", alpha=0.8,
             label="number of close contacts per relevant episode")
    plt.legend(loc='best')
    plt.grid(True)

    """ graph 4 """
    ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    ax4.set_xlabel('episodes')
    ax4.set_ylabel('reward')
    locs, labels = xticks()
    xticks(np.arange(t), np_evaluation_episode_counter)
    ax4.plot('x', 'y', data=df_evaluation_reward_tom, marker=',', linestyle='-', color="cornflowerblue", alpha=0.8,
             label="reward tom")
    ax4.plot('x', 'y', data=df_evaluation_reward_jerry, marker=',', linestyle='-', color="springgreen", alpha=0.8,
             label="reward jerry")
    plt.grid(True)
    plt.legend(loc='best')

    """ save the graph """
    plt.savefig(dirname + "/" + random_name + ".png", dpi=300)

    """ show the graph """
    plt.show()

