import matplotlib
import time

import os

from matplotlib import pyplot as plt
import pandas as p
import numpy as np

from matplotlib.pyplot import xticks

t = 1
evaluation_episode_counter = [1, 4, 9, 10, 12, 15, 16, 18, 23, 27, 30, 31, 32, 33, 35, 36, 39, 41, 42, 43, 44, 45, 46, 48, 49, 51, 54, 55, 56, 57, 58, 61, 64, 66, 69, 71, 73, 80, 81, 85, 87, 92, 93, 94, 97, 102, 108, 113, 120, 121, 125, 127, 129, 130, 140, 143, 144, 146, 2, 5, 6, 9, 10, 16, 17, 19, 21, 23, 29, 30, 31, 32, 34, 35, 38, 43, 45]
evaluation_episode_time = [1116.8906354904175, 1707.703847169876, 1398.1529726982117, 248.5497908592224, 532.9221646785736, 1853.5424642562866, 584.3907163143158, 1957.3089611530304, 307.909068107605, 1066.8345594406128, 1013.4326107501984, 300.948504447937, 984.771815776825, 689.8141529560089, 992.1223590373993, 479.88032364845276, 941.1460852622986, 1751.0885787010193, 380.47644805908203, 668.8670742511749, 818.1349055767059, 428.2372405529022, 703.3073952198029, 1059.5534629821777, 215.37258505821228, 778.8175187110901, 207.9900209903717, 573.6225473880768, 106.68042778968811, 440.407488822937, 457.9913680553436, 1344.1545362472534, 675.5786492824554, 933.4780764579773, 208.83805012702942, 586.3815236091614, 1194.1532068252563, 1464.867344379425, 1727.1654601097107, 1608.164930343628, 1473.305671930313, 1904.945648908615, 808.4314970970154, 1058.8857114315033, 692.4427254199982, 836.993043422699, 796.0383796691895, 838.9753420352936, 571.2896356582642, 1275.3419227600098, 323.7079677581787, 839.8483057022095, 1126.629380941391, 1509.7244634628296, 918.4156897068024, 595.6244993209839, 1167.287090063095, 1248.9915471076965, 427.6671669483185, 1405.509408712387, 916.1034741401672, 771.8917076587677, 1108.7712376117706, 229.22479724884033, 232.4518449306488, 202.22605848312378, 607.7610061168671, 505.9081144332886, 447.83322381973267, 974.2037405967712, 884.9337668418884, 286.94513845443726, 1062.9161989688873, 632.071756362915, 552.7264499664307, 1752.0058183670044, 754.3280069828033]
evaluation_too_close_counter = [5, 32, 80, 12, 39, 56, 59, 75, 16, 18, 29, 12, 60, 22, 29, 21, 54, 50, 18, 23, 71, 13, 33, 52, 11, 41, 15, 31, 0, 20, 16, 37, 32, 29, 0, 6, 76, 0, 89, 76, 114, 42, 25, 37, 14, 37, 7, 21, 11, 42, 5, 0, 0, 23, 3, 21, 41, 33, 8, 4, 17, 14, 42, 3, 5, 2, 0, 26, 20, 27, 29, 3, 16, 23, 24, 75, 7]
evaluation_reward_tom = [-64452400.0, -128400785.0, 155898140.0, 6313395.0, -4333085.0, -23331960.0, -5860035.0, -3785750.0, -2475545.0, 5264945.0, -57674710.0, -12649930.0, -33182695.0, -73957005.0, -4681550.0, 12119625.0, -24163995.0, 1558130.0, 19855230.0, 45160920.0, -1371690.0, 23015180.0, -25126795.0, -72691390.0, -787305.0, -4311465.0, 6540105.0, -15991885.0, -5825895.0, -3367100.0, -29965045.0, -9660685.0, -5723230.0, 35073865.0, -7242665.0, -14497855.0, -25839750.0, -12473820.0, -110807105.0, -55603485.0, -46183055.0, -30323930.0, 96572565.0, -47446740.0, -37162945.0, 30761910.0, 6405835.0, -69414990.0, 51633420.0, 52804525.0, -1341475.0, -14298725.0, -4034860.0, 137286390.0, -1334210.0, -33365190.0, 25737800.0, 5071195.0, -20383005.0, -13640530.0, 60058045.0, -1227515.0, -129972370.0, -6152405.0, -385220.0, 12035825.0, -85575.0, -30141995.0, -35795605.0, 11581400.0, -26098930.0, 6641965.0, 6706640.0, 20825840.0, -32751950.0, -130213140.0, -9004490.0]
evaluation_reward_jerry = [-52650210.0, -355535205.0, -161898590.0, -10286240.0, -34241715.0, -6216095.0, 28491205.0, -25339825.0, -17773230.0, -16874615.0, -2879890.0, -5725685.0, 6906395.0, -30306915.0, -31309600.0, -26688830.0, -5899070.0, -55532145.0, -11327950.0, -23969270.0, -3407510.0, -1924820.0, -28772235.0, -121278110.0, -9607675.0, 43152695.0, 10169865.0, -18055215.0, -1773890.0, 16982050.0, 8711145.0, -8346250.0, 21291050.0, -25077155.0, -230275.0, -74150.0, -9854350.0, -41736535.0, -140891395.0, -146110845.0, -40417735.0, -24632885.0, -6961875.0, 101169770.0, -17233585.0, -53524150.0, -24176925.0, 24796215.0, -2353170.0, -110708305.0, -2981960.0, -4896035.0, -9609645.0, -89874215.0, -14418590.0, 17698355.0, -120005415.0, -37561265.0, -10363495.0, 9605985.0, -140444950.0, 22176015.0, 73560905.0, -7700130.0, 7623230.0, -9878990.0, -5170055.0, -31843205.0, -9261600.0, -31926495.0, 19521850.0, -15722250.0, -41017965.0, -57135760.0, -50412645.0, 53186725.0, -4188990.0]
evaluation_reward_roadrunner = [36327010.0, 44016835.0, 148182900.0, 2870000.0, -2386535.0, -58305365.0, 20738025.0, -134152035.0, -2318105.0, -1773405.0, -70519660.0, 4854660.0, -1990110.0, -26349770.0, -5559960.0, 30990000.0, -64196545.0, -1983995.0, -21064460.0, -28934785.0, -36520240.0, -9150730.0, -45244830.0, -37381395.0, 10658935.0, -36509795.0, -9475665.0, 6978895.0, 6686505.0, -21927380.0, -28096265.0, -13978590.0, -37009905.0, -2202550.0, -2537565.0, -58830145.0, -62652370.0, -23415790.0, -133243765.0, -136042845.0, -51724995.0, -145446210.0, -214572320.0, -82622210.0, -22693645.0, -26725415.0, 20272865.0, -30507635.0, 6016430.0, -36476600.0, 22450245.0, -35341710.0, 3009230.0, 18810495.0, -7465705.0, -54895635.0, 110509775.0, -6912925.0, -123025.0, -19356050.0, 89879575.0, -24326760.0, -70897990.0, -99585.0, 22646060.0, -5532350.0, -1775495.0, -28987550.0, -29317755.0, -33669305.0, -30383790.0, -263185.0, -115599780.0, -18394170.0, 45024315.0, -36235815.0, -17640435.0]
evaluation_reward_coyote = [111213275.0, -20799260.0, -9788495.0, -11083945.0, 3152540.0, 24326670.0, -4261585.0, 114139040.0, -1205435.0, 7295285.0, -57746220.0, -150050.0, -41106235.0, -22836235.0, 11402820.0, -28016725.0, 14518820.0, -20942305.0, -21146205.0, -169305.0, -20849580.0, -18830830.0, -2021395.0, 2838465.0, -588425.0, 22398435.0, 8542305.0, 36113295.0, -265705.0, -3103940.0, -5529925.0, -14995455.0, -4450950.0, -4763215.0, 826895.0, 20811090.0, 73711050.0, -12027940.0, 341370005.0, 9389375.0, -16553655.0, 108275060.0, -45717465.0, -14240470.0, -43835445.0, -79814075.0, -57782535.0, -56633825.0, -7919785.0, 57725895.0, 11481445.0, 11000575.0, 19541585.0, -131601095.0, 12354380.0, 2842470.0, -47724345.0, 4364305.0, -23362180.0, -414180.0, 106472980.0, -21376015.0, -5007085.0, 2917695.0, -119220.0, -4744720.0, 4237025.0, 11004315.0, 5766285.0, 45297975.0, -59397435.0, 2352725.0, -6478525.0, -22626885.0, -49189955.0, -38166025.0, 16269930.0]
evaluation_winner_agent = ['Roadrunner', 'Roadrunner', 'Tom', 'Tom', 'Coyote', 'Tom', 'Jerry', 'Jerry', 'Coyote', 'Coyote', 'Tom', 'Roadrunner', 'Coyote', 'Jerry', 'Coyote', 'Roadrunner', 'Coyote', 'Coyote', 'Tom', 'Tom', 'Tom', 'Tom', 'Roadrunner', 'Coyote', 'Roadrunner', 'Jerry', 'Tom', 'Tom', 'Roadrunner', 'Jerry', 'Jerry', 'Tom', 'Jerry', 'Tom', 'Coyote', 'Coyote', 'Coyote', 'Jerry', 'Coyote', 'Coyote', 'Jerry', 'Coyote', 'Tom', 'Jerry', 'Roadrunner', 'Tom', 'Roadrunner', 'Jerry', 'Tom', 'Coyote', 'Coyote', 'Coyote', 'Roadrunner', 'Roadrunner', 'Coyote', 'Jerry', 'Roadrunner', 'Tom', 'Jerry', 'Jerry', 'Roadrunner', 'Jerry', 'Jerry', 'Coyote', 'Jerry', 'Tom', 'Coyote', 'Coyote', 'Coyote', 'Tom', 'Tom', 'Tom', 'Coyote', 'Tom', 'Roadrunner', 'Coyote', 'Coyote']
evaluation_game_won_timestamp = [None, None, 1398.1529726982117, 248.5497908592224, None, 1853.5424642562866, None, None, None, None, 1013.4326107501984, None, None, None, None, None, None, None, 380.47644805908203, 668.8670742511749, 818.1349055767059, 428.2372405529022, None, None, None, None, 207.9900209903717, 573.6225473880768, None, None, None, 1344.1545362472534, None, 933.4780764579773, None, None, None, None, None, None, None, None, 808.4314970970154, None, None, 836.993043422699, None, None, 571.2896356582642, None, None, None, None, None, None, None, None, 1248.9915471076965, None, None, None, None, None, None, None, 202.22605848312378, None, None, None, 974.2037405967712, 884.9337668418884, 286.94513845443726, None, 632.071756362915, None, None, None]
evaluation_flag_captured_tom = [None, None, 121.38146686553955, 131.56653714179993, None, 1754.934564113617, None, None, None, None, 650.2932140827179, None, 851.5904393196106, None, None, None, None, None, 256.9701843261719, 235.5644793510437, 766.284943819046, 177.16642785072327, None, None, None, None, 152.9704988002777, 385.0996072292328, None, None, None, 1173.5608150959015, None, 527.8860244750977, None, None, None, None, 1654.8182718753815, None, 1461.565996646881, None, 107.13206386566162, None, None, 537.597775220871, None, None, 49.78904175758362, 209.52041244506836, None, None, None, None, None, None, None, 981.6592466831207, None, None, None, None, None, None, None, 74.22121524810791, None, None, None, 598.3876404762268, 483.4013955593109, 181.8611297607422, 930.5951642990112, 87.03929090499878, None, None, 722.6757197380066]
evaluation_flag_captured_jerry = [None, 1487.0618448257446, None, None, None, None, 41.246798515319824, 523.7099645137787, None, None, 580.1659755706787, None, None, 392.050758600235, None, 193.01531720161438, None, None, None, None, None, None, None, None, None, 192.01747608184814, None, None, None, 121.54304146766663, 270.4613411426544, None, 170.89116644859314, None, None, None, None, 927.9975605010986, None, None, 1070.268658399582, None, None, 329.5698628425598, None, None, 368.9648630619049, 375.65715861320496, None, None, None, None, None, None, None, 397.92722511291504, None, None, 294.77637481689453, 1157.5498580932617, None, 492.10435819625854, 393.94261026382446, None, 83.10236549377441, None, None, None, None, None, 267.48882150650024, None, None, None, None, None, None]
evaluation_flag_captured_roadrunner = [732.7431452274323, 318.9602999687195, None, None, None, None, 336.53666520118713, None, None, None, None, 134.0990822315216, None, None, None, 260.0069811344147, None, None, None, None, None, None, 222.49794507026672, None, 176.79314184188843, None, None, None, 44.49503421783447, None, None, None, None, None, 200.79790377616882, None, None, None, None, None, None, None, None, None, 551.5802626609802, None, 488.697384595871, 599.8739676475525, None, None, 172.59720635414124, None, 791.6642873287201, 123.70982456207275, None, None, 401.0728087425232, None, None, None, 255.00488424301147, None, None, None, 61.411964654922485, None, None, None, None, None, None, None, None, None, 69.90278840065002, 1189.634920835495, None]
evaluation_flag_captured_coyote = [678.1522750854492, None, None, None, 489.50806879997253, 1317.9504933357239, None, None, 283.6342124938965, 207.87621474266052, None, None, 933.6617658138275, None, 615.3235349655151, None, 402.9872646331787, 1515.1678845882416, None, 307.6935040950775, None, None, None, 855.5028636455536, None, None, None, 118.78612756729126, None, None, None, None, None, None, 207.50437211990356, 489.64727687835693, 43.59762763977051, None, 76.92796397209167, 1009.2750458717346, None, 1053.451570034027, None, None, None, None, None, None, None, 406.04429054260254, 90.44596481323242, 380.77680468559265, None, None, 647.5397779941559, None, None, 717.3041300773621, None, None, 170.44077229499817, None, None, 179.59107375144958, None, None, 185.9180133342743, 61.86953687667847, 326.33908438682556, 325.58244824409485, None, 249.08457946777344, 846.8088200092316, None, None, 1051.971718788147, 633.4850766658783]
evaluation_agents_ran_into_each_other = []
dirname = "samples"
evaluation_steps_tom = [481, 968, 722, 119, 311, 495, 334, 741, 175, 143, 530, 140, 468, 418, 252, 259, 392, 450, 191, 301, 333, 260, 391, 550, 106, 346, 126, 282, 62, 248, 295, 391, 324, 244, 83, 200, 613, 25, 1066, 931, 707, 726, 438, 661, 403, 484, 567, 554, 305, 824, 179, 27, 112, 804, 176, 345, 537, 360, 227, 201, 480, 225, 681, 127, 145, 91, 4, 254, 262, 512, 421, 151, 262, 229, 310, 1016, 128]
evaluation_steps_jerry = [443, 1031, 875, 119, 311, 495, 334, 684, 178, 144, 584, 216, 468, 342, 251, 259, 392, 450, 190, 339, 333, 260, 391, 454, 106, 346, 126, 282, 81, 248, 295, 391, 324, 242, 83, 219, 613, 25, 1104, 951, 707, 726, 438, 661, 403, 485, 567, 554, 305, 767, 160, 27, 131, 1013, 195, 345, 746, 360, 227, 258, 674, 225, 624, 127, 145, 91, 4, 274, 205, 550, 477, 151, 264, 250, 311, 940, 147]
evaluation_steps_roadrunner = [428, 973, 666, 120, 311, 496, 334, 1255, 175, 143, 530, 140, 468, 342, 251, 259, 396, 450, 190, 341, 333, 261, 392, 448, 106, 346, 126, 301, 62, 248, 295, 393, 324, 242, 84, 545, 613, 25, 1066, 722, 707, 727, 552, 661, 460, 485, 567, 554, 305, 675, 217, 27, 189, 861, 271, 345, 597, 360, 227, 201, 442, 225, 586, 127, 145, 92, 4, 254, 206, 360, 420, 151, 358, 265, 312, 940, 300]
evaluation_steps_coyote = [405, 949, 551, 119, 311, 495, 334, 913, 175, 143, 432, 140, 469, 342, 251, 259, 392, 450, 190, 380, 333, 260, 391, 486, 106, 346, 126, 283, 62, 248, 295, 391, 324, 242, 83, 181, 613, 25, 1066, 703, 707, 726, 438, 661, 403, 487, 567, 554, 381, 673, 160, 27, 321, 994, 176, 345, 575, 360, 227, 201, 444, 225, 586, 127, 145, 111, 4, 255, 206, 576, 479, 151, 263, 323, 314, 940, 128]
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
