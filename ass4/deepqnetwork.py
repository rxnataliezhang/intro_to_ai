import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import pickle

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(self, num_action, num_obser, learning_rate=0.1,
            gamma_value=0.99, epsilon_value=0.05, buffer_size=1000,
            batch_size=50, replay=True, target=True):
        self.num_action = num_action
        self.num_obser = num_obser
        self.learning_rate = learning_rate
        self.gamma_value = gamma_value
        self.epsilon_value = epsilon_value
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay = replay
        self.target = target

        self.replay_buffer = np.zeros((self.buffer_size, self.num_obser * 2 + 2))
        self.build_q_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def build_q_network(self):
        '''build Q network and target network'''
        
        ## define state and q_matrix (placeholder)
        self.curr_state = tf.placeholder(tf.float32, [None, self.num_obser], name='curr_state')  
        self.next_state = tf.placeholder(tf.float32, [None, self.num_obser], name='next_state')
        self.target_value = tf.placeholder(tf.float32, [None, self.num_action], name='target_value')

        #### build Q network ####
        ## define Q network with two hidden layers of 10 rectified linear units
        ## initialize parameters collection of Q network collection_name, l1_size = 10, l2_size = 10, weight, bias
        with tf.variable_scope('Q_network'):
            num_l1 = 10
            num_l2 = 10
            coll = ['Q_netwrok_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            # w_init = tf.random_uniform_initializer(-0.1,0.1)
            w_init = tf.random_normal_initializer(0,0.5)
            b_init = tf.constant_initializer(0.1)
            
            ## define l1 
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.num_obser, num_l1], initializer=w_init, collections=coll)
                b1 = tf.get_variable('b1', [1, num_l1], initializer=b_init, collections=coll)
                l1 = tf.nn.relu(tf.matmul(self.curr_state, w1) + b1)

            ## define l2 
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [num_l1, num_l2], initializer=w_init, collections=coll)
                b2 = tf.get_variable('b2', [1, num_l2], initializer=b_init, collections=coll)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            ## define l3 
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [num_l2, self.num_action], initializer=w_init, collections=coll)
                b3 = tf.get_variable('b3', [1, self.num_action], initializer=b_init, collections=coll)
                self.q_value = tf.identity(tf.matmul(l2, w3) + b3)

        ## define error
        with tf.variable_scope('error'):
            self.error = tf.reduce_sum(tf.square(self.target_value-self.q_value))

        ## define optimizer
        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)

        #### build target network ####
        ## define target network with two hidden layers of 10 rectified linear units
        ## initialize parameters collection of target nerwork
        with tf.variable_scope('target_network'):
            coll = ['tar_netwrok_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            
            ## define l1 
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.num_obser, num_l1], initializer=w_init, collections=coll)
                b1 = tf.get_variable('b1', [1, num_l1], initializer=b_init, collections=coll)
                l1 = tf.nn.relu(tf.matmul(self.next_state, w1) + b1)

            ## define l2 
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [num_l1, num_l2], initializer=w_init, collections=coll)
                b2 = tf.get_variable('b2', [1, num_l2], initializer=b_init, collections=coll)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            ## define l3 
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [num_l2, self.num_action], initializer=w_init, collections=coll)
                b3 = tf.get_variable('b3', [1, self.num_action], initializer=b_init, collections=coll)
                self.next_q_value = tf.identity(tf.matmul(l2, w3) + b3)

    def update_replay_buffer(self, curr_state, action, reward, next_state):
        '''write transition into replay buffer
           Use a replay buffer of size 1000 and replay a mini-batch of size 50 after each new experience.'''
        
        if self.replay == True:
            
            if not hasattr(self, 'buffer_counter'):
                self.buffer_counter = 0
            idx = self.buffer_counter % self.buffer_size
            self.replay_buffer[idx, :] = np.hstack((curr_state, action, reward, next_state))
            self.buffer_counter += 1

    def choose_action(self, observation):
        '''choose one action to execute'''
        observation = observation[np.newaxis, :] 

        if np.random.uniform() > self.epsilon_value:
            action = np.argmax(self.sess.run(self.q_value, feed_dict={self.curr_state: observation}))
        else:
            action = action = np.random.randint(0, self.num_action)
        return action

    def update_target_network(self):
        '''update the parameters of target network with thoses of q network'''

        if self.target == True:
            self.sess.run([tf.assign(a, b) for a, b in zip(tf.get_collection('tar_netwrok_parameters'), \
                                                        tf.get_collection('Q_netwrok_parameters'))])


    def learn(self, curr_state, action, reward, next_state):
        ''''''
        ## sample to get the mini-batch from replay buffer
        if self.replay == True:
            if self.buffer_counter > self.buffer_size:
                idx = np.random.choice(self.buffer_size, size=self.batch_size)
            else:
                idx = np.random.choice(self.buffer_counter, size=self.batch_size)
            mini_batch = self.replay_buffer[idx, :]

        else: 
            mini_batch = np.zeros((1, self.num_obser * 2 + 2))
            mini_batch[0,:] = np.hstack((curr_state, action, reward, next_state))

        ## exectute to get current q value and next q value of every actions
        if self.target == True:
            q_value, next_q_value = self.sess.run([self.q_value, self.next_q_value],
                                    feed_dict={self.curr_state: mini_batch[:, :self.num_obser],
                                            self.next_state: mini_batch[:, -self.num_obser:]})
        else:
            q_value, next_q_value = self.sess.run([self.q_value, self.q_value],
                                    feed_dict={self.curr_state: mini_batch[:, :self.num_obser],
                                            self.next_state: mini_batch[:, -self.num_obser:]})
            


        ## calculate target q value for every actions
        target_value = q_value.copy()
        action_idx = mini_batch[:, self.num_obser].astype(int)
        reward_value = mini_batch[:, self.num_obser+1]
        if self.replay == True:
            sample_idx = np.arange(self.batch_size)
            target_value[sample_idx, action_idx] = reward_value + self.gamma_value*np.max(next_q_value, axis=1)
        else: target_value[:, action_idx] = reward_value + self.gamma_value*np.max(next_q_value, axis=1)
                                                    
        self.sess.run(self.train_step, feed_dict={self.curr_state: mini_batch[:, :self.num_obser], 
                                                self.target_value: target_value})

        self.sess.run(self.error, feed_dict={self.curr_state: mini_batch[:, :self.num_obser], 
                                                self.target_value: target_value})  
                                            

env = gym.make('CartPole-v0')
env = env.unwrapped
RL = DeepQNetwork(num_action=env.action_space.n, num_obser=env.observation_space.shape[0], 
                  learning_rate=0.01, gamma_value=0.99, epsilon_value=0.05, buffer_size=1000, 
                  batch_size=50, replay=False, target=False)

reward_log = []
timestep_log = []
for i_episode in range(500):
    if i_episode % 2 == 0:
        RL.update_target_network()

    observation = env.reset()
    reward_per_ep = 0
    for t in range(500):
            
        env.render()
        action = RL.choose_action(observation)
        next_observation, reward, done, info = env.step(action)

        ## define a new reward
        x, x_dot, theta, theta_dot = next_observation
        r1 = (env.x_threshold - abs(x))/env.x_threshold
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
        reward = 2*r1*r2 / (r1+r2)

        RL.update_replay_buffer(observation, action, reward, next_observation)

        reward_per_ep = reward + reward_per_ep * 0.99

        RL.learn(observation, action, reward, next_observation)

        if done:
            timestep_log.append(t+1)
            print('Episode:{:3} Timesteps:{:4} Reward:{}'.format(i_episode, t+1, round(reward_per_ep,2)))
            break
        if t == 499:
            timestep_log.append(t+1)
            print('Episode:{:3} Timesteps:{:4} Reward:{}'.format(i_episode, t+1, round(reward_per_ep,2)))

        observation = next_observation

    reward_log.append(reward_per_ep)

# pickle.dump(reward_log, open('reward_log1.p','wb'))
pickle.dump(timestep_log, open('timestep_log4.p','wb'))

    