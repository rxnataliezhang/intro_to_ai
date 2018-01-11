import pickle
import numpy as np
import matplotlib.pyplot as plt

reward_log_1 = pickle.load(open( 'reward_log1.p', 'rb'))
reward_log_2 = pickle.load(open( 'reward_log2.p', 'rb'))
reward_log_3 = pickle.load(open( 'reward_log3.p', 'rb'))
reward_log_4 = pickle.load(open( 'reward_log4.p', 'rb'))
plt.plot(np.arange(len(reward_log_1)), reward_log_1, label='rt')
plt.plot(np.arange(len(reward_log_2)), reward_log_2, label='r~t')
plt.plot(np.arange(len(reward_log_3)), reward_log_3, label='~rt')
plt.plot(np.arange(len(reward_log_4)), reward_log_4, label='~r~t')
plt.ylabel('reward_per_ep')
plt.xlabel('episode')
plt.legend()
plt.show()

# timestep_log1 = pickle.load(open( 'timestep_log1.p', 'rb'))
# timestep_log2 = pickle.load(open( 'timestep_log2.p', 'rb'))
# timestep_log3 = pickle.load(open( 'timestep_log3.p', 'rb'))
# timestep_log4 = pickle.load(open( 'timestep_log4.p', 'rb'))
# plt.plot(np.arange(len(timestep_log1)), timestep_log1, label='rt')
# plt.plot(np.arange(len(timestep_log2)), timestep_log2, label='r~t')
# plt.plot(np.arange(len(timestep_log3)), timestep_log3, label='~rt')
# plt.plot(np.arange(len(timestep_log4)), timestep_log4, label='~r~t')
# plt.ylabel('timestep_per_ep')
# plt.xlabel('episode')
# plt.legend()
# plt.show()