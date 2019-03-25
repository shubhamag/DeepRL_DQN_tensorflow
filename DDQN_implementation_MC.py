#!/usr/bin/env python
import pdb
import keras 
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Activation, Input,Lambda
from keras.models import model_from_json
import keras.backend as K

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
            env = gym.make("MountainCar-v0")

        self.model = Sequential()
        self.dueling_type = 'max'
        numStates = env.observation_space.shape[0]
        numActions = env.action_space.n
        # Linear
        #self.model.add(Dense(numActions, input_shape=(numStates,), kernel_initializer='normal', use_bias=True))
        
        # MLP

        # MLP small
        input_layer = Input(shape=(numStates,))
        hl =  Dense(24, use_bias = True, activation='relu')(input_layer)
        # self.model.add(Activation('relu'))
        hl = Dense(24, use_bias=True, activation='relu')(hl)
        hl = Dense(24, use_bias=True, activation='relu')(hl)


        # V = Dense(1)(hl)
        y = Dense(numActions + 1, activation='linear')(hl)
        # A = Dense(numActions)(hl)

        # Q_layer = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(numActions,) , arguments = [V,A])(hl)

        if self.dueling_type == 'avg':
            Q_layer = Lambda(lambda x: K.expand_dims(x[:, 0], -1) + x[:, 1:] - K.mean(x[:, 1:], keepdims=True),
                             output_shape=(numActions,))(y)
        elif self.dueling_type == 'max':
            Q_layer = Lambda(lambda x: K.expand_dims(x[:, 0], -1) + x[:, 1:] - K.max(x[:, 1:], keepdims=True),
                             output_shape=(numActions,))(y)

        # self.model.add(Dense(numActions))
        self.model = Model(input=[input_layer], output=[Q_layer])

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
            self.model.load_weights(model_weights_file)
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
        print ("Env name:", environment_name)
        self.env_name = environment_name
        self.env = gym.make(environment_name)
        self.env.reset()
        self.numStates = self.env.observation_space.shape[0]
        self.numActions = self.env.action_space.n
        self.max_episodes = 10000

        # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        sess = tf.Session(config=config)
        keras.backend.tensorflow_backend.set_session(sess)
        self.max_iter = 1000000


        # Init training params
        self.model_file = "model_DDQN.json"
        self.model_weights_file = "model_DDQN_MC_fri_"
        self.QNet = QNetwork(self.env)
        self.model = self.QNet.model
        #self.QNet.load_model(model_file)
        #self.QNet.load_model_weights(model_weights_file)
        self.test_results =[]
        self.train_results =[]
        self.batch_losses = []
        self.max_test_reward = -1*170
        
        # experience replay
        if(self.env_name == 'MountainCar-v0'):
            self.gamma = 1
            self.eps = 0.7
        else:
            self.gamma = 0.99
            self.eps = 0.5

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


        if(self.env_name == 'MountainCar-v0'):
            self.gamma = 1
            self.eps = 0.7
            anneal_till_episode = 1500
        else:
            self.gamma = 0.99
            self.eps = 0.5
            anneal_till_episode = 1000
        print ("Training with gamma=",self.gamma)


        eps_space = np.linspace(self.eps, 0.05, anneal_till_episode)
        print("eps_space size: ",eps_space.shape)
        # exit()
        anneal_iter = 10  # anneal eps linearly iters
        self.sync_iter = 20
        self.batch_size = 32
        self.test_episodes = 200
        do_anneal = True
        do_save = False

        reward_count = 0
        done_count = 0

        state = self.env.reset()
        # list of trajectories
        state_batch = np.zeros((self.batch_size, self.numStates))
        nstate_batch = np.zeros((self.batch_size, self.numStates))
        action_batch = np.zeros((self.batch_size), dtype=int)
        reward_batch = np.zeros((self.batch_size))
        not_done_batch = np.zeros((self.batch_size), dtype=bool)
        
        numEpisodes = 1

        for q_iter in range(self.max_iter):
            # fit on minibatch            
            minibatch = self.replay.sample_batch(self.batch_size)

            for i in range(self.batch_size): 
                s, a, r, s1, done = minibatch[i]
                state_batch[i, :] = s
                nstate_batch[i, :] = s1
                action_batch[i] = a
                not_done_batch[i] = not done
                reward_batch[i] = r
        
            target_values = self.model.predict(state_batch)
            target_values[ np.arange(self.batch_size) , action_batch] = reward_batch
            next_values = np.max(target_model.predict(nstate_batch), axis=1)
            #predicted_value[:, action_batch] += self.gamma*next_values
            target_values[not_done_batch, action_batch[not_done_batch]] += self.gamma*next_values[not_done_batch]
            self.model.fit(state_batch, target_values, verbose=0)
                        

 
            # get value function for current state, sim and add to replay mem
            value_predictions = self.model.predict(x=state[None])
            action = self.epsilon_greedy_policy(value_predictions)
            next_state, reward, is_done, _  = self.env.step(action)
            self.replay.append([state, action, reward, next_state, is_done])
            reward_count += reward
            
            state = next_state
            
#                print("annealed eps to ", self.eps)

            if (is_done == True):
                numEpisodes += 1
                if numEpisodes % self.test_episodes == 1:
                    self.test()

                if numEpisodes % 500 == 1:
                    weight_file = self.model_weights_file + str(numEpisodes) + ".h5"

                    print("saving model and weights to ",weight_file)
                    self.QNet.save_model(self.model_file)
                    self.QNet.save_model_weights(weight_file)
                    
                target_model.set_weights(self.model.get_weights())
                state = self.env.reset()
                self.train_results.append(reward_count)
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
        if model_weights != None:
            self.model.load_model_weights(model_weights)

        state = self.env.reset()
        while True:
            # self.env.render()

            q_values = self.model.predict(state[None])
            action = self.greedy_policy(q_values)
            [state, reward, done, _] = self.env.step(action)

            reward_count += reward

            # print("action ", action)

            if (done == True):
                state = self.env.reset()
                done_count += 1
                if (done_count == 20):
                    avg_reward = float(reward_count / 20)
                    print("Average reward of last 20 episodes: ", avg_reward)
                    print("Test DONE!!\n\n");
                    print ("original lr:", self.QNet.lr)

                    self.test_results.append(avg_reward)

                    self.write_test_results()
                    if (abs(avg_reward) < abs(self.max_test_reward)):
                        self.max_test_reward = avg_reward
                        print ("updates self.max_test_reward to: ", self.max_test_reward)
                        self.QNet.save_model_weights("best_weights_DDQN_MC.h5")
                    reward_count = 0
                    done_count = 0

                    break

        return
    def test_and_log(self, model_weights=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        if model_weights != None:
            self.model.load_model_weights(model_weights)
            print ("loaded model wieights from ",model_weights)

        print ("\n\nLet's play....\n\n")
        reward_count = 0
        done_count = 0
        ep_reward = 0

        state = self.env.reset()
        while True:
            # self.env.render()

            q_values = self.model.predict(state[None])
            action = self.greedy_policy(q_values)
            [state, reward, done, _] = self.env.step(action)
            ep_reward +=reward

            reward_count += reward


            if (done == True):
                state = self.env.reset()
                self.test_results.append(ep_reward)
                ep_reward=0
                done_count += 1
                if (done_count == 100):
                    avg_reward = float(reward_count / 100)
                    print("Average reward of last 100 episodes: ", avg_reward)
                    print("Test DONE!!\n\n");
                    print ("original lr:", self.QNet.lr)

                    self.test_results.append(avg_reward)

                    self.write_test_results("test_100")


                    break

        return


    def write_test_results(self, filename="DDQN_replay_MC_test_results"):
        file = open(filename, 'w')
        for item in self.test_results:
            file.write("%s\n" % item)
        file.write("\n\nTraining:")
        for item in self.train_results:
            file.write("%s\n" % item)
        file.close()

        return
    def test_record_video(self,weights_file):


        print ("\n\nLet's play....\n\n")
        reward_count = 0
        done_count = 0

        if weights_file != None:
            self.QNet.load_model_weights(weights_file)
            print ("loaded weights from: ",weights_file)
        else:
            print ("Error, weight file not set")
            return

        # self.env = wrappers.Monitor(self.env, "./CP_vid_ddqn", force=True)
        state = self.env.reset()

        # self.env.render()
        # input("paused to set recorder, press enter to cont...")
        while True:
            # self.env.render()
            self.env.render()

            q_values = self.model.predict(state[None])
            action = self.greedy_policy(q_values)
            [state, reward, done, _] = self.env.step(action)

            reward_count += reward

            # print("action ", action)

            if (done == True):
                # state = self.env.reset()

                # self.env = wrappers.Monitor(self.env, "./CP_vid_ddqn", force=True)
                state = self.env.reset()
                done_count += 1
                if (done_count == 3):
                    avg_reward = float(reward_count / 20)
                    print("Average reward of last 20 episodes: ", avg_reward)


                    break

        return
    def wrap_and_record(self,paths):
        self.env = gym.wrappers.Monitor(self.env, './ddqn_mc_vids',
                                   video_callable=lambda episode_id: True, force=True)
        for path in paths:
            self.test_record_video(path)

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
    parser = argparse.ArgumentParser(description='Deep Q Network    Argument Parser')
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
    # agent.test_and_log("best_weights_DDQN_MC.h5")
    # agent.wrap_and_record(["model_DDQN_MC_fri_501.h5","model_DDQN_MC_fri_1501.h5", "model_DDQN_MC_fri_3001.h5","model_DDQN_MC_fri_4001.h5", "model_DDQN_MC_fri_5001.h5"])
    #train_model_with_target(env, replay, model, iter_max=100000,do_save = False)
    # train_model(env,model,iter_max=100000,do_save=True)


if __name__ == '__main__':
    main(sys.argv)
