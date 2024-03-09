#0 - Birth Rate                                                           10 - hospital capacity,  11 - gamma mor iso
#1 - Death Rate                                                           12 - gamma mor1
#2 - Vaccination Rate                                                     13 - gamma mor2
#3 - Vaccination Immunization Rate                                        14 - gamma imm
#4 - Maternally Immunized Rate                                            15 - simulation days
#5 - Beta Exposed                                                         16 - Incubation Period(Exposed State)
#6 - Quarantine Rate                                                      17 - Infection Period(Infected State)
#7 - eps_exp - transition rate exposed compared to infected               18 - Immunization Period(Vaccinated State)
#8 - eps_qua - transition rate quarantine compared to infected            19 - Initial Susceptible Population
#9 - eps_iso - transition rate isolated compared to infected              20 - Initial Exposed Population
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from maincode import Node3
params = [0.0, 0.0, 0.0, 0.0, 0.0,
          0.6, 50.0, 90.0, 0.0, 0.0,
          17380, 0.7 ,1, 3.5, 75.0, 300,
          5, 20, 14, 112374331, 2,1]

params1 = [0.0, 0.0, 0.0, 0.0, 0.0,
          0.02, 50.0, 70.0, 0.0, 0.0,
          12000, 2 ,3, 4.12, 83.0, 300,
          5, 20, 14, 20185000, 2,1]
lockdown =  [
    (datetime.date(2020, 3, 11), datetime.date(2020, 3, 23),0.7),
    (datetime.date(2020, 3, 24), datetime.date(2020, 4, 19),0.3),
    (datetime.date(2020, 4, 20), datetime.date(2020, 5, 3),0.5),
    (datetime.date(2020, 5, 4), datetime.date(2020, 5, 31),0.56),
    (datetime.date(2020, 6, 1), datetime.date(2020, 6, 30),0.58),
    (datetime.date(2020, 7, 1), datetime.date(2020, 7, 31),0.54),
    (datetime.date(2020, 8, 1), datetime.date(2020, 8, 4),0.65),
    (datetime.date(2020, 8, 5), datetime.date(2020, 8, 31),0.6),
    (datetime.date(2020, 9, 1), datetime.date(2020, 9, 6),0.65),
    (datetime.date(2020, 9, 7), datetime.date(2020, 9, 30),0.5),
    (datetime.date(2020, 10, 1), datetime.date(2020, 10, 14),0.6),
    (datetime.date(2020, 10, 15), datetime.date(2020, 10, 31),0.53),
    (datetime.date(2020, 11, 1), datetime.date(2020, 11, 30),0.5),
    (datetime.date(2020, 12, 1), datetime.date(2020, 12, 15),0.4),
]

lockdown_length = len(lockdown)
node = Node3(params1)
  # check correctenes of the initialization
if node.check_init():
    # check correctenes of the initialization
    node.set_sim_len((lockdown[-1][1] - lockdown[0][0]).days)
    node.check_init()

    # create states based on the
    # initialization parameters

    node.create_states()
    node.indexes()

    # create transitions based on
    # the created states
    node.create_transitions()
    # start simulation
    for i in range (lockdown_length):
      start = time.time()
      start_date = lockdown[i][0]
      end_date = lockdown[i][1]
      node.set_sim_len((end_date - start_date).days)  # Difference in days
      node.total_sim_len += node.param_num_sim
      print(node.total_sim_len)
      node.param_beta_exp = lockdown[i][2]
      #node.to_String()
      node.define_state_arr()
      for ind in range(node.param_num_sim):
        node.states_arr[ind,:] = node.states_x
        node.stoch_solver(ind+1)

        if ind % node.param_disp_interval == 0:
            print("Iteration: {}/{}".format(ind + 1, node.param_num_sim))
      end = time.time()
      #print("Simulation took {} sec".format(end - start))
      node.data_sus = np.append(node.data_sus, node.states_arr.dot(node.ind_sus))

      node.data_exp = np.append(node.data_exp, node.states_arr.dot(node.ind_exp))
      node.data_qua = np.append(node.data_qua, node.states_arr.dot(node.ind_qua))
      node.data_inf = np.append(node.data_inf, node.states_arr.dot(node.ind_inf))
      node.data_iso = np.append(node.data_iso, node.states_arr.dot(node.ind_iso))
      node.data_imm = np.append(node.data_imm, node.states_arr.dot(node.ind_imm))
      node.data_dea = np.append(node.data_dea, node.states_arr[:, -1])

      node.init_susceptible = node.data_sus[-1]
      node.init_exposed = node.data_exp[-1]

    node.df['sus'] = node.data_sus
    node.df['exp'] = node.data_exp
    node.df['qua'] = node.data_qua
    node.df['inf'] = node.data_inf
    node.df['iso'] = node.data_iso
    node.df['imm'] = node.data_imm
    node.df['dea'] = node.data_dea
    node.df['con'] = node.df['exp'] + node.df['qua'] + node.df['inf'] + node.df['iso'] + node.df['imm'] + node.df['dea']



