#!/usr/bin/env python
import pdb
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse
import os.path
from collections import deque

class QNetwork():
    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 
    def __init__(self, env=None, lr=1e-4):
        # Define your network architecture here. It is also a good idea to define any training operations 
        if env==None:
            env = gym.make("CartPole-v0")

        self.model = Sequential()
        numStates = env.observation_space.shape[0]
        numActions = env.action_space.n
        # Linear
        self.model.add(Dense(numActions, input_shape=(numStates,), kernel_initializer='glorot_normal', use_bias=True))
        
        # MLP
#        self.model.add(Dense(256, input_shape=(numStates,), use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(128, use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(64, use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(numActions))

        # MLP small
#        self.model.add(Dense(24, input_shape=(numStates,), use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(24, use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(24, use_bias=True))
#        self.model.add(Activation('relu'))
#        self.model.add(Dense(numActions))

        # Add optimizers here, initialize your variables, or alternately compile your model here.
        self.lr=lr
        # optimizer = keras.optimizers.RMSprop(lr=self.lr, decay=1e-5)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='MSE')
        self.model.summary()
        return

    def save_model(self, model_file):
        # Helper function to save your model
        model_json = self.model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to ", model_file)
        return

    def save_model_weights(self, model_weights_name):
        # serialize weights to HDF5
        self.model.save_weights(model_weights_name)
        print("Saved model weights to ", model_weights_name)
        return

    def load_model(self, model_file):
        # Helper function to load an existing model.
        if os.path.isfile(model_file):
            print("Loading existing model definition\n")
            json_file = open(model_file, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            adam = keras.optimizers.Adam(lr=self.lr, decay=1e-5)
            self.model.compile(optimizer=adam, loss='MSE')
            self.model.summary()
        return

    def load_model_weights(self, model_weights_file):
        # Helper funciton to load model weights.
        if os.path.isfile(model_weights_file):
            # load weights into new model
            model.load_weights(model_weights_file)
            print("Loaded model weights from file: ", model_weights_file)
        pass

class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        # memSize x 4 (s, a, r, s)
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        pass

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
        return

class DQN_Agent():
    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    def __init__(self, environment_name="CartPole-v0", render=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        print (environment_name)
        self.env = gym.make(environment_name)
        self.env.reset()
        self.numStates = self.env.observation_space.shape[0]
        self.numActions = self.env.action_space.n
        self.max_episodes = 5000

        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        sess = tf.Session(config=config)
        keras.backend.tensorflow_backend.set_session(sess)
        self.max_iter = 1000000
        self.eps = 0.5

        # Init training params
        model_file = "model_linear.json"
        model_weights_file = "weights_MLP.h5"
        self.QNet = QNetwork(self.env)
        self.model = self.QNet.model
        #self.QNet.load_model(model_file)
        #self.QNet.load_model_weights(model_weights_file)

        # experience replay
        self.replay = Replay_Memory()
        self.burn_in_memory()

        return

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand() <= self.eps: # take random action
            return self.env.action_space.sample()
        # Take greedy action
        next_action = np.argmax(q_values)
        return next_action

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        next_action = np.argmax(q_values)
        return next_action

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        target_model = keras.models.clone_model(self.model)
        target_model.set_weights(self.model.get_weights())

        self.gamma = 1
        self.eps = 0.5
        anneal_till = 1000
        eps_space = np.linspace(self.eps, 0.05, anneal_till)
        anneal_iter = 10  # anneal eps linearly iters
        self.sync_iter = 10
        self.batch_size = 32
        self.test_episodes = 200
        do_anneal = True
        do_save = False

        # target_model = keras.models.clone_model(model)
        model_file = 'model_linear_MC.json'
        model_weights_file = "weights_linear_MC.h5"
        reward_count = 0
        done_count = 0

        state = self.env.reset()
        # list of trajectories
        state_batch = np.zeros((self.batch_size, self.numStates))
        nstate_batch = np.zeros((self.batch_size, self.numStates))
        action_batch = np.zeros((self.batch_size), dtype=int)
        reward_batch = np.zeros((self.batch_size))
        not_done_batch = np.zeros((self.batch_size), dtype=bool)

        numEpisodes = 0

        for q_iter in range(self.max_iter):
            # get value function for current state, sim and add to replay mem
            target_values = self.model.predict(state[None])
            action = self.epsilon_greedy_policy(target_values)
            next_state, reward, is_done, _  = self.env.step(action)

            if next_state[0]>0.5:
                target_values[0][action]= reward
            else:
                target_values[0][action]= reward + self.gamma*np.amax(target_model.predict(next_state[None]))

            self.model.fit(state[None], target_values, verbose=0)
            self.replay.append([state, action, reward, next_state, is_done])
            reward_count += reward

            state = next_state

#                print("annealed eps to ", self.eps)

            if (is_done == True):
                numEpisodes += 1
                if numEpisodes % self.test_episodes == 1:
                    self.test()

                if numEpisodes % 500 == 1:
                    print("saving model and weights")
                    self.QNet.save_model("model_linear_MC.json")
                    self.QNet.save_model_weights(model_weights_file)

                target_model.set_weights(self.model.get_weights())
                state = self.env.reset()

                # anneal eps
                if (self.eps > 0.05):
                    self.eps = eps_space[numEpisodes]

                print("Episode ", numEpisodes, "/", self.max_episodes)
                print("Average reward of episode: ", reward_count, " Epsilon", self.eps)
                reward_count = 0
                done_count = 0
                if numEpisodes> self.max_episodes:
                    break

#            if q_iter % self.sync_iter == 0:
#                target_model.set_weights(self.model.get_weights())
        print("Finished running env for ", self.max_iter, "iterations")

    def test(self, model_weights=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        print ("\n\nLet's play....\n\n")
        reward_count = 0
        done_count = 0

        # load weights if prompted
        if model_weights!=None:
            self.model.load_model_weights(model_weights)

        state = self.env.reset()
        while True:
            # self.env.render()

            q_values= self.model.predict(state[None])
            action = self.greedy_policy(q_values)
            [state, reward, done, _] = self.env.step(action)

            reward_count+=reward

            print("action ", action)

            if (done ==True):
                state = self.env.reset()
                done_count+=1
                if(done_count==20):
                    print("Average reward of last 20 episodes: ", reward_count/20)
                    print("Test DONE!!\n\n");
                    reward_count=0
                    done_count=0
                    break
        return

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        state = self.env.reset()
        print("Burn in with epsilon=", self.eps)
        for i in range(self.replay.burn_in):
            q_values = self.model.predict(x=state[None])
            action = self.epsilon_greedy_policy(q_values[0])
            next_state, reward, done, _ = self.env.step(action)
            self.replay.append( [state, action, reward, next_state, done] )
            if done:
                state = self.env.reset()
            else:
                state = next_state
        return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train', dest='train',type=int,default=1)
    parser.add_argument('--model', dest='model_file',type=str)
    parser.add_argument('--max_episodes', dest='max_episodes',type=str)
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    env_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    agent = DQN_Agent("MountainCar-v0")
    agent.train()
    #train_model_with_target(env, replay, model, iter_max=100000,do_save = False)
    # train_model(env,model,iter_max=100000,do_save=True)

    #play(env, model, 2000)

if __name__ == '__main__':
    main(sys.argv)
