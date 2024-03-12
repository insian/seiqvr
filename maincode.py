import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import datetime

class Node3(object):
    def __init__(self, params):
        self.param_br = params[0]/100.0     # Daily birth rate
        self.param_dr = params[1]/100.0     # Daily mortality rate except infected people
        self.param_vr = params[2]/100.0           # Daily vaccination rate (Ratio of susceptible
                                       # population getting vaccinated)
        self.param_vir = params[3]/100.0           # Ratio of the immunized after vaccination
        self.param_mir = params[4]/100.0           # Maternal immunization rate

        self.param_beta_exp = params[5]             # Susceptible to exposed transition constant
        self.param_qr  = params[6]/100.0            # Daily quarantine rate (Ratio of Exposed getting Quarantined)
        self.param_beta_inf = 0.0                   # Susceptible to infected transition constant

        self.param_eps_exp = params[7]/100.0       # Disease transmission rate of exposed compared to the infected
        self.param_eps_qua = params[8]/100.0       # Disease transmission rate of quarantined compared to the infected
        self.param_eps_iso  = params[9]/100.0       # Disease transmission rate of Severe Infected compared to Infected

        self.param_hosp_capacity = params[10]      # Maximum amount patients that hospital can accommodate

        self.param_gamma_mor_iso = params[11]/100.0
        self.param_gamma_mor1 = params[12]/100.0     # Infected to Dead transition probability
        self.param_gamma_mor2 = params[13]/100.0     # # Infected to Dead transition probability (Hospital Cap. Exceeded)
        self.param_gamma_im = params[14]/100.0      # Infected to Recovery Immunized transition probability

        self.param_dt = params[21]/1.0             # Sampling time in days (1/24 corresponds to one hour)
        self.param_sim_len = params[15]       # Length of simulation in days

        self.param_num_states = 0    # Number of states
        self.param_num_sim = int(self.param_sim_len / self.param_dt) + 1       # Number of simulation
        self.total_sim_len = 0

        self.param_t_exp = params[16]           # Incubation period (The period from the start of
                                      # incubation to the end of the incubation state)
        self.param_t_inf = params[17]           # Infection period (The period from the start of
                                      # infection to the end of the infection state)
        self.param_t_vac = params[18] # Vaccination immunization period (The time to
                                      # vaccinatization immunization after being vaccinated)

        self.param_n_exp = int(self.param_t_exp / self.param_dt)
        self.param_n_inf = int(self.param_t_inf / self.param_dt)
        self.param_n_vac = int(self.param_t_vac / self.param_dt)

        self.param_save_res = 1
        self.param_disp_progress = 1
        self.param_disp_interval = 100
        self.param_vis_on = 1                 # Visualize results after simulation

        np.random.seed(1)
        self.param_rand_seed = np.random.randint(low = 1, high = 100, size = 625)

        # Define the initial values for the states
        self.init_susceptible = params[19]
        self.init_vaccinated = 0.0
        self.init_exposed = params[20]
        self.init_quarantined = 0.0
        self.init_infected = 0.0
        self.init_isolated = 0.0
        self.init_vaccination_imm = 0.0
        self.init_maternally_imm = 0.0
        self.init_recovery_imm = 0.0

        # Define states
        self.states_x = [0, self.init_susceptible]
        self.states_dx = []
        self.states_name = ['Birth', 'Susceptible']
        self.states_type = ['Birth', 'Susceptible']

        # Define transitions
        self.source = ['Birth', 'Birth']
        self.dest = ['Susceptible', 'Maternally_Immunized' ]
        self.source_ind = []
        self.dest_ind = []

        self.df = pd.DataFrame(columns = ['sus','exp','qua','inf','iso','imm','dea','con',])
        self.data_sus = np.array([], dtype=int)
        self.data_exp = np.array([], dtype=int)
        self.data_qua = np.array([], dtype=int)
        self.data_inf = np.array([], dtype=int)
        self.data_iso = np.array([], dtype=int)
        self.data_imm = np.array([], dtype=int)
        self.data_dea = np.array([], dtype=int)
        self.data_vac = np.array([], dtype=int)

    def get_param_br(self):
        return self.param_br

    def set_sim_len(self,val):
        self.param_sim_len = val
        self.set_param_num_sim()

    def set_init_suceptible(self,val):
        self.init_susceptible = val

    def set_init_exposed(self, val):
        self.init_exposed = val

    def set_param_num_sim(self):
        self.param_num_sim = int(self.param_sim_len / self.param_dt) + 1

    def to_String(self):
        print(self.param_beta_exp, self.init_susceptible, self.init_exposed, self.param_sim_len, self.param_num_sim,sep='\t')

    def check_init(self):
        if self.param_beta_exp == 0 and self.param_beta_inf == 0:
            print('[ERROR] Both beta_exp and beta_inf cannot be zero.')
            return 0
        elif self.param_beta_exp != 0 and self.param_beta_inf != 0:
            print('[ERROR] Both beta_exp and beta_inf cannot be non-zero.')
            return 0
        else:
            #print("[INFO] Initialization was done properly!")
            return 1

    def create_states(self):
        # create some temporal variables
        n_vac = self.param_n_vac
        n_vac_exp = n_vac + self.param_n_exp
        n_vac_2exp = n_vac_exp + self.param_n_exp
        n_vac_2exp_inf = n_vac_2exp + self.param_n_inf
        n_vac_2exp_2inf = n_vac_2exp_inf + self.param_n_inf + 1
        count = len(self.states_name) - 1

        # loop to create states
        while count < n_vac_2exp_2inf:
            # Vaccinated States
            if count <= n_vac:
                self.states_name.append('Vaccinated_{}'.format(count))
                self.states_type.append('Susceptible')
                if count == 1:
                    self.states_x.append(self.init_vaccinated)
                    count += 1
                    continue

            # Exposed States (Includes both exposed and quarantined)
            elif count > n_vac and count <= n_vac_exp:
                self.states_name.append('Exposed_{}'.format(count - n_vac))
                self.states_type.append('Exposed')
                if count == n_vac + 1:
                    self.states_x.append(self.init_exposed)
                    count += 1
                    continue

            elif count > n_vac_exp and count <= n_vac_2exp:
                self.states_name.append('Quarantined_{}'.format(count - n_vac_exp))
                self.states_type.append('Exposed')
                if count == n_vac_exp + 1:
                    self.states_x.append(self.init_quarantined)
                    count += 1
                    continue

            # Infected States (Includes: Infected, Isolated and Severe Infected)
            elif count > n_vac_2exp and count <= n_vac_2exp_inf:
                self.states_name.append('Isolated_{}'.format(count - n_vac_2exp))
                self.states_type.append('Isolated')
                if count == n_vac_2exp + 1:
                    self.states_x.append(self.init_isolated)
                    count += 1
                    continue

            else:
                self.states_name.append('Infected_{}'.format(count - n_vac_2exp_inf))
                self.states_type.append('Infected')
                if count == n_vac_2exp_inf + 1:
                    self.states_x.append(self.init_infected)
                    count += 1
                    continue

            self.states_x.append(0)

            count += 1

        # Add the Immunized and dead states
        self.states_name.append('Vaccination_Immunized')
        self.states_type.append('Recovered')
        self.states_x.append(self.init_vaccination_imm)

        self.states_name.append('Maternally_Immunized')
        self.states_type.append('Recovered')
        self.states_x.append(self.init_maternally_imm)

        self.states_name.append('Recovery_Immunized')
        self.states_type.append('Recovered')
        self.states_x.append(self.init_recovery_imm)

        self.states_name.append('Dead')
        self.states_type.append('Dead')
        self.states_x.append(0)

        # convert states into numpy arrays
        # for fast processing
        self.states_x = np.asarray(self.states_x, dtype=np.float32)
        self.states_dx = np.zeros(self.states_x.shape, dtype=np.float32)

        # initialize number of states
        self.param_num_states = len(self.states_x)
        #print(self.param_num_states)
        #print(self.states_x)

        #print("[INFO] States were created...")

    def create_transitions(self):
        # create some temporal variables
        count = len(self.source) - 1
        n_st = self.param_num_states
        n_vac = self.param_n_vac
        n_exp = self.param_n_exp
        n_inf = self.param_n_inf

        # Transition 1 - Birth to Susceptible
        # Already Present

        # Transition 2 - Birth to Maternally Immunized
        # Alread Present

        # Transition 3 - Any State except Birth to Dead (Natural Mortality)
        for ind in range(count, n_st - 1):
            self.source.append(self.states_name[ind])
            self.dest.append('Dead')

        # Transition 4 - Susceptible to Vaccinated[1]
        if self.param_vr != 0:
            self.source.append('Susceptible')
            self.dest.append('Vaccinated_1')

        # Transition 4a - Vaccinated[i] to Vaccinated[i+1] until i+1 == n_vac
        for ind in range(n_vac - 1):
            self.source.append(self.states_name[2 + ind])
            self.dest.append(self.states_name[3 + ind])

        # Transition 5 - Vaccinated[n_vac] to Vaccination_Immunized
        if self.param_vir != 0:
            self.source.append('Vaccinated_{}'.format(n_vac))
            self.dest.append('Vaccination_Immunized')

        # Transition 6 - Vaccinated[n_vac] to Susceptible
        if self.param_vir != 1:
            self.source.append('Vaccinated_{}'.format(n_vac))
            self.dest.append('Susceptible')

        # Transition 7 - Susceptible to Exposed[1]
        if self.param_beta_exp != 0:
            self.source.append('Susceptible')
            self.dest.append('Exposed_1')

        # Transition 8 - Susceptible to Infected[1]
        if self.param_beta_inf != 0:
            self.source.append('Susceptible')
            self.dest.append('Infected_1')

        # Transition 7a - Exposed[i] to Exposed[i+1] until i+1 == n_exp
        for ind in range(n_exp - 1):
            self.source.append('Exposed_{}'.format(ind + 1))
            self.dest.append('Exposed_{}'.format(ind + 2))

        # Transition 9 - Exposed[n_inc] to Infected[1]
        if self.param_n_exp != 0:
            self.source.append('Exposed_{}'.format(n_exp))
            self.dest.append('Infected_1')

        # Transition 10 - Exposed[i] to Quarantined[i+1] until i+1 == n_exp
        for ind in range(n_exp - 1):
            self.source.append('Exposed_{}'.format(ind + 1))
            self.dest.append('Quarantined_{}'.format(ind + 2))

        # Transition 10a - Quarantined[i] to Quarantined[i+1] until i+1 == n_exp
        for ind in range(n_exp - 1):
            self.source.append('Quarantined_{}'.format(ind + 1))
            self.dest.append('Quarantined_{}'.format(ind + 2))

        # Transition 11 - Quarantined[n_exp] to Isolated[1]
        if self.param_n_exp != 0:
            self.source.append('Quarantined_{}'.format(n_exp))
            self.dest.append('Isolated_1')

        # Transition 9a - Infected[i] to Infected[i+1] until i+1 == n_inf
        for ind in range(n_inf - 1):
            self.source.append('Infected_{}'.format(ind + 1))
            self.dest.append('Infected_{}'.format(ind + 2))

        # Transition 11a - Isolated[i] to Isolated[i+1] until i+1 == n_inf
        for ind in range(n_inf - 1):
            self.source.append('Isolated_{}'.format(ind + 1))
            self.dest.append('Isolated_{}'.format(ind + 2))

        # Transition 12 - Infected[n_inf] to Recovery_Immunized
        self.source.append('Infected_{}'.format(n_inf))
        self.dest.append('Recovery_Immunized')

        # Transition 13 - Isolated[n_inf] to Recovery Immunized
        self.source.append('Isolated_{}'.format(n_inf))
        self.dest.append('Recovery_Immunized')

        # Transition 14 - Infected[n_inf] to Susceptible
        self.source.append('Infected_{}'.format(n_inf))
        self.dest.append('Susceptible')

        # Transition 15 - Isolated[n_inf] to Susceptible
        self.source.append('Isolated_{}'.format(n_inf))
        self.dest.append('Susceptible')

        # Transition 16 - Infected[n_inf] to Dead
        self.source.append('Infected_{}'.format(n_inf))
        self.dest.append('Dead')

        # Transition 17 - Isolated[n_inf] to Dead
        self.source.append('Isolated_{}'.format(n_inf))
        self.dest.append('Dead')

        for ind in range(len(self.source)):
            self.source_ind.append(self.states_name.index(self.source[ind]))
            self.dest_ind.append(self.states_name.index(self.dest[ind]))


        #print("[INFO] State transitions were created...")

    def indexes(self):
        # define vectors of indices
        self.ind_vac = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_inf = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_exp = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_qua = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_imm = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_sus = np.zeros((len(self.states_x)), dtype=np.float32)
        self.ind_iso = np.zeros((len(self.states_x)), dtype=np.float32)

        # intialize vectors of indices
        for ind in range(len(self.states_name)):
            if 'Vaccinated_' in self.states_name[ind]:
                self.ind_vac[ind] = 1
            elif 'Infected_' in self.states_name[ind]:
                self.ind_inf[ind] = 1
            elif 'Exposed_' in self.states_name[ind]:
                self.ind_exp[ind] = 1
            elif 'Quarantined_' in self.states_name[ind]:
                self.ind_qua[ind] = 1
            elif 'Immunized' in self.states_name[ind]:
                self.ind_imm[ind] = 1
            elif 'Susceptible' in self.states_name[ind]:
                self.ind_sus[ind] = 1
            elif 'Isolated_' in self.states_name[ind]:
                self.ind_iso[ind] = 1


        # define other indices
        self.ind_vac1 = self.states_name.index('Vaccinated_1')
        self.ind_vacn = self.states_name.index('Vaccinated_{}'.format(self.param_n_vac))

        self.ind_exp1 = self.states_name.index('Exposed_1')
        self.ind_expn = self.states_name.index('Exposed_{}'.format(self.param_n_exp))

        self.ind_qua1 = self.states_name.index('Quarantined_1')
        self.ind_quan = self.states_name.index('Quarantined_{}'.format(self.param_n_exp))

        self.ind_iso1 = self.states_name.index('Isolated_1')
        self.ind_ison = self.states_name.index('Isolated_{}'.format(self.param_n_inf))

        self.ind_inf1 = self.states_name.index('Infected_1')
        self.ind_infn = self.states_name.index('Infected_{}'.format(self.param_n_inf))

    def dx_generator(self, size, val):
        dx = 0

        for ind in range(size):
            rand_num = random.uniform(0, 1)
            if rand_num < val:
                dx += 1

        return dx

    def stoch_solver(self,days):
        # define a list to store transitions
        expval = []
        state_1 = self.states_x[1]
        random.seed(1)
        # Total population is the sum of all states except birth and death
        total_pop = self.states_x[1:-1].sum()

        # Transition 1 - Birth to Susceptible
        expval.append(total_pop * self.param_br * (1 - self.param_mir) * self.param_dt)

        # Transition 2 - Birth to Maternally Immunized
        expval.append(total_pop * self.param_br * self.param_mir * self.param_dt)

        # Transition 3 - Any State except Birth to Dead (Natural Mortality)
        expval += (self.states_x[1:self.param_num_states - 1] * self.param_dr * self.param_dt).tolist()

        # Transition 4 - Susceptible to Vaccinated[1]
        if self.param_vr != 0:
            expval.append(state_1 * self.param_vr * self.param_dt)

        # Transition 4a - Vaccinated[i] to Vaccinated[i+1] until i+1 == n_vac
        if self.param_n_vac != 0:
            expval += (self.states_x[2:self.param_n_vac + 1] * \
                   ((1 - self.param_dr) * self.param_dt)).tolist()

        state_vac = self.states_x.dot(self.ind_vac).sum()
        # Transition 5 - Vaccinated[n_vac] to Vaccination_Immunized
        if self.param_vir != 0:
            expval.append(state_vac * self.param_vir)

        # Transition 6 - Vaccinated[n_vac] to Susceptible
        if self.param_vir != 1:
            expval.append(state_vac * (1 - self.param_dr * self.param_dt - self.param_vir))

        # Transition 7 - Susceptible to Exposed[1]
        temp_1 = self.states_x.dot(self.ind_exp).sum()
        if self.param_n_exp != 0:
            expval.append(state_1 * temp_1 * self.param_beta_exp * self.param_dt / total_pop)
        # Transition 8 - Susceptible to Infected[1]
        if self.param_beta_inf != 0:
            expval.append(state_1 * temp_1 * self.param_beta_inf * self.param_dt / total_pop)

        # Transition 7a - Exposed[i] to Exposed[i+1] until i+1 == n_exp
        expval += (self.states_x[self.ind_exp1:self.ind_exp1 + self.param_n_exp - 1] * \
                   ((1 - self.param_dr - self.param_qr) * self.param_dt)).tolist()

        # Transition 9 - Exposed[n_exp] to Infected[1]
        if self.param_n_exp != 0:
            expval.append(self.states_x[self.ind_expn] * (1 - self.param_dr) * self.param_dt)

        # Transition 10 - Exposed[i] to Quarantined[i+1] until i+1 == n_exp
        expval += (self.states_x[self.ind_exp1:self.ind_exp1 + self.param_n_exp - 1] * \
                  (self.param_qr * self.param_dt)).tolist()

        # Transition 10a - Quarantined[i] to Quarantined[i+1] until i+1 == n_exp
        expval += (self.states_x[self.ind_qua1:self.ind_qua1 + self.param_n_exp - 1] * \
                   ((1 - self.param_dr) * self.param_dt)).tolist()

        # Transition 11 - Quarantined[n_exp] to Isolated[1]
        if self.param_n_exp != 0:
            expval.append(self.states_x[self.ind_quan] * (1 - self.param_dr) * self.param_dt)

        # Transition 9a - Infected[i] to Infected[i+1] until i+1 == n_inf
        expval += (self.states_x[self.ind_inf1:self.ind_inf1 + self.param_n_inf - 1] * \
                   ((1 - self.param_dr) * self.param_dt)).tolist()

        # Transition 11a - Isolated[i] to Isolated[i+1] until i+1 == n_inf
        expval += (self.states_x[self.ind_iso1:self.ind_iso1 + self.param_n_inf - 1] * \
                   ((1 - self.param_dr) * self.param_dt)).tolist()

        total_inf = self.states_x.dot(self.ind_inf).sum()
        # Transition 12 - Infected[n_inf] to Recovery_Immunized
        if total_inf <= self.param_hosp_capacity:
            expval.append(self.states_x[self.ind_infn] * \
                          (1 - self.param_gamma_mor1) * self.param_gamma_im)
        else:
            expval.append(self.states_x[self.ind_infn] * \
                          (1 - self.param_gamma_mor2) * self.param_gamma_im)

        # Transition 13 - Isolated[n_inf] to Recovery_Immunized
        expval.append(self.states_x[self.ind_ison] * \
                          (1 - self.param_gamma_mor_iso) * self.param_gamma_im)

        # Transition 14 - Infected[n_inf] to Susceptible
        if total_inf <= self.param_hosp_capacity:
            expval.append(self.states_x[self.ind_infn] * \
                          (1 - self.param_gamma_mor1) * (1 - self.param_gamma_im))
        else:
            expval.append(self.states_x[self.ind_infn] * \
                          (1 - self.param_gamma_mor2) * (1 - self.param_gamma_im))
        # Transition 15 - Isolated[n_inf] to Susceptible
        expval.append(self.states_x[self.ind_ison] * \
                          (1 - self.param_gamma_mor_iso) * (1 - self.param_gamma_im))

        # Transition 16 - Infected[n_inf] to Dead
        if total_inf < self.param_hosp_capacity:
            expval.append(self.states_x[self.ind_infn] *self.param_gamma_mor1)
        else:
            expval.append(self.states_x[self.ind_infn] *self.param_gamma_mor2)

        # Transition 17 - Isolated[n_iso] to Dead
        expval.append(self.states_x[self.ind_ison] *self.param_gamma_mor_iso)

        # Randomly generate the transition value based on the expected value
        l = []
        #print(len(expval))
        #print(len(self.source_ind))
        #print(len(self.dest_ind))
        for i, (eval, sind, dind) in enumerate(zip(expval, self.source_ind, self.dest_ind)):
            #print('index: ',i)
            #print('eval:',eval,'  sind:',sind,'  dind:',dind)
            if eval < 0:
                dx = 0
            else:
                dx = round(eval)
            '''if eval < 10 and eval > 0:
                temp1 = int(np.ceil(eval * 10 + np.finfo(np.float32).eps))
                temp2 = eval/temp1
                dx = self.dx_generator(temp1, temp2)
                #print("dx0_10: ",dx)
            elif eval < 0:
                dx = 0
                #print('dx0: ',dx)
            else:
                dx = round(eval)
            '''
            #print('dxelse: ',dx)
            #print('final_dx: ',dx,end='\n\n')
            #time.sleep(0.25)
            # Apply the changes for the transitions to the
            # corresponding source and destination states
            temp = self.states_x[sind] - dx
            if temp <= 0:
                self.states_x[dind] += self.states_x[sind]
                self.states_x[sind] = 0
            else:
                self.states_x[sind] = temp
                self.states_x[dind] += dx

    def define_state_arr(self):
        # create a containers to store states
        self.states_arr = np.zeros((self.param_num_sim, len(self.states_name)), dtype=np.float32)
        self.iter = -1

    def visualize_states(self):

        if self.param_vis_on:
            time_arr = np.linspace(0, self.param_num_sim, self.param_num_sim) * self.param_dt
            state_sus = self.states_arr.dot(self.ind_sus)
            state_vac = self.states_arr.dot(self.ind_vac)
            state_exp = self.states_arr.dot(self.ind_exp)
            state_inf = self.states_arr.dot(self.ind_inf)
            state_iso = self.states_arr.dot(self.ind_iso)
            state_qua = self.states_arr.dot(self.ind_qua)
            state_conf = state_exp + state_inf + state_iso + state_qua
            state_imm = self.states_arr.dot(self.ind_imm)
            state_dea = self.states_arr[:, -1]

            fig = plt.figure(figsize = (10,10))
            plt.plot(time_arr, state_sus, label = 'Susceptible',color='blue')
            plt.plot(time_arr, state_vac, label = 'Vaccinated',color='#cde5ff')
            plt.plot(time_arr, state_conf, label = 'Confirmed',color='#fd9432')
            plt.plot(time_arr, state_imm, label = 'Immunized',color='green')
            plt.plot(time_arr, state_dea, label = 'Dead',color='black')
            plt.xlabel("Day")
            plt.ylabel("Population")
            plt.legend(loc="upper right")
            plt.show()