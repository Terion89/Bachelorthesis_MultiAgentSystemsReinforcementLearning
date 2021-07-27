import matplotlib
import time

import os

from matplotlib import pyplot as plt
import pandas as p
import numpy as np

from matplotlib.pyplot import xticks

t = 1
evaluation_episode_counter = [2, 4, 5, 6, 20, 190]
evaluation_episode_time = [300, 450, 140, 589, 900, 345]
evaluation_too_close_counter = [4, 20, 3, 0, 4, 20]
evaluation_reward_tom = [50, -300, 20, 50, 20, 140]
evaluation_reward_jerry = [60, -34, 20, 23, -111, -56]
evaluation_reward_roadrunner = [67, 700, 90, 56, 333, 12]
evaluation_reward_coyote = [57, -65, 45, 3, 23, 677]
evaluation_winner_agent = ["tom", "roadrunner", "jerry", "tom", "roadrunner", "coyote"]
evaluation_game_won_timestamp = [300, 450, 140, 589, 900, 345]
evaluation_flag_captured_tom = [150, 30, 0, 34, 25, 66, 100]
evaluation_flag_captured_jerry = [10, 0, 40, 56, 200, 400, 23]
evaluation_flag_captured_roadrunner = [50, 250, 40, 40, 100, 345]
evaluation_flag_captured_coyote = [40, 56, 30, 0, 90, 200]
evaluation_agents_ran_into_each_other = [50, 78, 0, 0, 0, 0]
dirname = "samples"
evaluation_steps_tom = [3000, 3000, 488, 3888, 20000, 50003]
evaluation_steps_jerry = [3000, 3000, 488, 3888, 20000, 50003]
evaluation_steps_roadrunner = [3000, 3000, 488, 3888, 20000, 50003]
evaluation_steps_coyote = [3000, 3000, 488, 3888, 20000, 50003]
"""
graph plots to visualize the result
available data:

evaluation_episode_counter = []             - counts the episodes for evaluation
evaluation_episode_time = []                - episode duration per episode
evaluation_too_close_counter = []           - how often they came close per episode
evaluation_reward_<name> = []               - reward an agent gained per episode
evaluation_winner_agent = []                - string, who won per episode
evaluation_game_won_timestamp = []          - timestamp, somebody won per episode
evaluation_flag_captured_<name> = []        - timestamp an agent captured the flag per episode
evaluation_agents_ran_into_each_other = []  - timestamp, the agents ran into each other per episode
evaluation_steps_<name> = []                - steps an agent took during the episode
dirname: place to save the plots


graph_01: episodes where somebody won the game
    x: episodes
    y: time in sec
       values: overall time, flag_captured_<name>, game_won_timestep + who won

graph_02: episodes where somebody won the game
    x: episodes
    y: time in sec
       values: number of steps per agent

graph_03: episodes where somebody won the game
    x: episodes
    y: reward
       values: rewards of all agents

graph_04: episodes where somebody won the game
    x: episodes
    y: counts where agents came close but did not crash
       values: too_close_counter

graph_05: episodes where the agents crashed
    x: episodes
    y: time, when agents crashed
    values: evaluation_agents_ran_into_each_other
"""
""" font manipulation for better proportions """
matplotlib.rcParams.update({'font.size': 4})

""" generate numpy-arrays """
np_evaluation_episode_counter = np.asarray(evaluation_episode_counter)
np_evaluation_episode_time = np.asarray(evaluation_episode_time)
np_evaluation_too_close_counter = np.asarray(evaluation_too_close_counter)
np_evaluation_reward_tom = np.asarray(evaluation_reward_tom)
np_evaluation_reward_jerry = np.asarray(evaluation_reward_jerry)
np_evaluation_reward_roadrunner = np.asarray(evaluation_reward_roadrunner)
np_evaluation_reward_coyote = np.asarray(evaluation_reward_coyote)
np_evaluation_game_won_timestamp = np.asarray(evaluation_game_won_timestamp)
np_evaluation_flag_captured_tom = np.asarray(evaluation_flag_captured_tom)
np_evaluation_flag_captured_jerry = np.asarray(evaluation_flag_captured_jerry)
np_evaluation_flag_captured_roadrunner = np.asarray(evaluation_flag_captured_roadrunner)
np_evaluation_flag_captured_coyote = np.asarray(evaluation_flag_captured_coyote)
np_evaluation_agents_ran_into_each_other = np.asarray(evaluation_agents_ran_into_each_other)
np_evaluation_steps_tom = np.asarray(evaluation_steps_tom)
np_evaluation_steps_jerry = np.asarray(evaluation_steps_jerry)
np_evaluation_steps_roadrunner = np.asarray(evaluation_steps_roadrunner)
np_evaluation_steps_coyote = np.asarray(evaluation_steps_coyote)
np_evaluation_winner_agent = np.asarray(evaluation_winner_agent)

""" set name of plot to not overwrite it after every episode """
# individual_name = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())

""" make dir, if it doesn't already exist """
os.makedirs(dirname, exist_ok=True)

plt.title("Episode results")

""" create dataframes for the plots """
df_evaluation_episode_time = p.DataFrame({'x': np_evaluation_episode_time})
df_evaluation_too_close_counter = p.DataFrame({'x': np_evaluation_too_close_counter})
df_evaluation_agents_ran_into_each_other = p.DataFrame({'x': np_evaluation_agents_ran_into_each_other})
df_evaluation_reward_tom = p.DataFrame({'x': np_evaluation_reward_tom})
df_evaluation_reward_jerry = p.DataFrame({'x': np_evaluation_reward_jerry})
df_evaluation_reward_roadrunner = p.DataFrame({'x': np_evaluation_reward_roadrunner})
df_evaluation_reward_coyote = p.DataFrame({'x': np_evaluation_reward_coyote})
df_evaluation_game_won_timestamp = p.DataFrame({'x': np_evaluation_game_won_timestamp})
df_evaluation_flag_captured_tom = p.DataFrame({'x': np_evaluation_flag_captured_tom})
df_evaluation_flag_captured_jerry = p.DataFrame({'x': np_evaluation_flag_captured_jerry})
df_evaluation_flag_captured_roadrunner = p.DataFrame({'x': np_evaluation_flag_captured_roadrunner})
df_evaluation_flag_captured_coyote = p.DataFrame({'x': np_evaluation_flag_captured_coyote})
df_evaluation_steps_tom = p.DataFrame({'x': np_evaluation_steps_tom})
df_evaluation_steps_jerry = p.DataFrame({'x': np_evaluation_steps_jerry})
df_evaluation_steps_roadrunner = p.DataFrame({'x': np_evaluation_steps_roadrunner})
df_evaluation_steps_coyote = p.DataFrame({'x': np_evaluation_steps_coyote})
df_evaluation_winner_agent = p.DataFrame({'x': np_evaluation_winner_agent})

""" graph 1 """
ax1 = plt.subplot2grid((5, 1), (0, 0), colspan=1)
ax1.set_xlabel('episodes')
ax1.set_ylabel('time in seconds')
ax1.plot('x', 'y', data=df_evaluation_episode_time, marker='x', linestyle='-', color="red", alpha=0.8,
         label="relevant game time", markersize=3)
ax1.plot('x', 'y', data=df_evaluation_game_won_timestamp, marker='.', linestyle='-', color="yellow", alpha=0.8,
         label="relevant game won after x seconds", markersize=3)
ax1.plot('x', 'y', data=df_evaluation_flag_captured_tom, marker='.', linestyle='-', color="cornflowerblue",
         alpha=0.8,
         label="timestamp tom captured the flag", markersize=3)
ax1.plot('x', 'y', data=df_evaluation_flag_captured_jerry, marker='.', linestyle='-', color="springgreen",
         alpha=0.8,
         label="timestamp jerry captured the flag", markersize=3)
ax1.plot('x', 'y', data=df_evaluation_flag_captured_roadrunner, marker='.', linestyle='-', color="cyan",
         alpha=0.8,
         label="timestamp roadrunner captured the flag", markersize=3)
ax1.plot('x', 'y', data=df_evaluation_flag_captured_coyote, marker='.', linestyle='-', color="greenyellow",
         alpha=0.8,
         label="timestamp coyote captured the flag", markersize=3)
locs, labels = xticks()
plt.xticks(rotation=90)
xticks(np.arange(len(evaluation_episode_counter)), np_evaluation_episode_counter)
ax11 = ax1.twiny()
#ax11.plot(range(len(evaluation_episode_counter)), np.ones(len(evaluation_episode_counter)))
ax11.set_xlabel("winner of the episode")
plt.xticks(rotation=90)
xticks(np.arange(len(evaluation_episode_counter)), np_evaluation_winner_agent)
plt.grid(True)
ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")


""" graph 2"""
ax5 = plt.subplot2grid((4, 1), (1, 0), colspan=1)
ax5.set_xlabel('episodes')
ax5.set_ylabel('number of steps')

ax5.plot('x', 'y', data=df_evaluation_steps_tom, marker='x', linestyle='-', color="mediumslateblue", alpha=0.8,
         label="steps tom", markersize=3)
ax5.plot('x', 'y', data=df_evaluation_steps_jerry, marker='.', linestyle='-', color="aquamarine", alpha=0.8,
         label="steps jerry", markersize=3)
ax5.plot('x', 'y', data=df_evaluation_steps_roadrunner, marker='x', linestyle='-', color="mediumpurple", alpha=0.8,
         label="steps roadrunner", markersize=3)
ax5.plot('x', 'y', data=df_evaluation_steps_coyote, marker='.', linestyle='-', color="mediumspringgreen", alpha=0.8,
         label="steps coyote", markersize=3)
locs, labels = xticks()
plt.xticks(rotation=90)
xticks(np.arange(len(evaluation_episode_counter)), np_evaluation_episode_counter)
plt.grid(True)
ax5.legend(bbox_to_anchor=(1, 1), loc="upper left")

""" graph 3 """
ax2 = plt.subplot2grid((4, 1), (2, 0), colspan=1)
ax2.set_xlabel('episodes')
ax2.set_ylabel('reward')
locs, labels = xticks()
plt.xticks(rotation=90)
xticks(np.arange(len(evaluation_episode_counter)), np_evaluation_episode_counter)
ax2.plot('x', 'y', data=df_evaluation_reward_tom, marker='.', linestyle='-', color="cornflowerblue", alpha=0.8,
         label="reward tom", markersize=3)
ax2.plot('x', 'y', data=df_evaluation_reward_jerry, marker='.', linestyle='-', color="springgreen", alpha=0.8,
         label="reward jerry", markersize=3)
ax2.plot('x', 'y', data=df_evaluation_reward_roadrunner, marker='.', linestyle='-', color="cyan", alpha=0.8,
         label="reward roadrunner", markersize=3)
ax2.plot('x', 'y', data=df_evaluation_reward_coyote, marker='.', linestyle='-', color="greenyellow", alpha=0.8,
         label="reward coyote", markersize=3)
plt.grid(True)
ax2.legend(bbox_to_anchor=(1, 1), loc="upper left")

""" graph 3 """
ax3 = plt.subplot2grid((4, 1), (3, 0), colspan=1)
ax3.set_xlabel('episodes')
ax3.set_ylabel('#agents_were_close')
locs, labels = xticks()
plt.xticks(rotation=90)
xticks(np.arange(len(evaluation_episode_counter)), np_evaluation_episode_counter)
ax3.plot('x', 'y', data=df_evaluation_too_close_counter, marker='.', linestyle='-', color="blue", alpha=0.8,
         label="number of close contacts", markersize=3)
ax3.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.grid(True)

""" graph 4 """
# ax4 = plt.subplot2grid((5, 1), (4, 0), colspan=1)
# ax4.set_xlabel('episodes')
# ax4.set_ylabel('time_in_seconds')
# ax4.plot('x', 'y', data=df_evaluation_agents_ran_into_each_other, marker='_', linestyle=':', color="red", alpha=0.8,
#          label="mission length, before something crashed")
# ax4.legend(loc='best')
# plt.grid(True)

""" save the graph """
plt.subplots_adjust(hspace=0.6, right=0.75)
plt.savefig(dirname + "/" + "result_plot.png", dpi=1500)  # + individual_name

plt.show()
print("plot")
