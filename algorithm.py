'''
 -------------------------
 COMP219-Ass2-Task2
 DQN for OpanAI video game
 Name: Maizhen Ning
 ID: 201376369
 Date: 6-Jan-2019
 -------------------------
'''

import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # observations before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 0.6 # starting value of epsilon
REPLAY_MEMORY = 30000 # number of previous transitions to remember
BATCH_SIZE = 64 # size of minibatch
UPDATE_TIME = 300 # after a number of steps to update traget Q Network




class DQN:

	def __init__(self,actions, pattern):
		# init the pattern of program
		self.pattern = pattern
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		# init Q network
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
		self.createTrainingMethod()


		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print ("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print ("Could not find old network weights")

	# design the CNN layers in DQN
	def createQNetwork(self):
		# design hyper parameters for CNN network layers' weights
		W_conv1 = tf.Variable(tf.truncated_normal([8,8,4,32], stddev = 0.01))
		b_conv1 = tf.Variable(tf.constant(0.01, shape = [32]))

		W_conv2 = tf.Variable(tf.truncated_normal([4,4,32,64], stddev = 0.01))
		b_conv2 = tf.Variable(tf.constant(0.01, shape = [64]))

		W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev = 0.01))
		b_conv3 = tf.Variable(tf.constant(0.01, shape = [64]))

		W_fc1 = tf.Variable(tf.truncated_normal([7040,512], stddev = 0.01))
		b_fc1 = tf.Variable(tf.constant(0.01, shape = [512]))

		W_fc2 = tf.Variable(tf.truncated_normal([512,self.actions], stddev = 0.01))
		b_fc2 = tf.Variable(tf.constant(0.01, shape = [self.actions]))

		# input layer
		stateInput = tf.placeholder("float",[None,170,150,4])

		# hidden layers
		# the 1st conv layer with relu activation function and pool layer
		conv1 = tf.nn.conv2d(stateInput, W_conv1, strides = [1, 4, 4, 1], padding = "SAME")
		relu1 = tf.nn.relu(conv1 + b_conv1)
		pool1 = tf.nn.max_pool(relu1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

		# the 2nd conv layer with relu activation function
		conv2 = tf.nn.conv2d(pool1, W_conv2, strides = [1, 2, 2, 1], padding = "SAME")
		relu2 = tf.nn.relu(conv2 + b_conv2)

		# the 3rd conv layer with relu activation function
		conv3 = tf.nn.conv2d(relu2, W_conv3, strides = [1, 1, 1, 1], padding = "SAME")
		relu3 = tf.nn.relu(conv3 + b_conv3)

		# reshape conv layers' output result for dense layers
		flat = tf.reshape(relu3,[-1,7040])

		# full conection layer
		h_fc1 = tf.nn.relu(tf.matmul(flat,W_fc1) + b_fc1)

		# calculate Q Value layer
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	# update target Q Networks 
	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	# design the training method of DQN
	def createTrainingMethod(self):
		# design the input of action and value of y in DRL
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


	def trainQNetwork(self):	
		# get random minibatch from replay memory to train the network
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
			

		# calculate y for DQN training method
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			# judge whether action lead to die in game and give out y
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.trainStep.run(feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch})

		# save network every 20000 running steps
		if self.timeStep % 20000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		# update traget q network after a number of running steps
		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		
	def setPerception(self,nextObservation,action,reward,terminal):
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		state = ""
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()

		# training pettern 
		if self.pattern == 0:
			if self.timeStep > OBSERVE:
				# Train the network
				self.trainQNetwork()
			# print information			
			if self.timeStep <= OBSERVE:
				state = "observe"
			elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
				state = "explore"
			else:
				state = "train"
			print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon, "/REWARD", reward)

		# testing pattern
		elif self.pattern == 1:
			state = "test"
			print ("TIMESTEP", self.timeStep, "/ STATE", state, "/REWARD", reward)

		self.currentState = newState
		self.timeStep += 1

	# get the action predict result from Network
	def getAction(self):
		# get the result from CNN layers
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = np.zeros(self.actions)
		# the index will determine the actual aciton command in environment
		action_index = 0
		# training pattern of program
		if self.pattern == 0: 
			# to judge whether use result from network or use random action result to help training 
			# the value of epsilon will start decrease to start training the network
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				# use network to get the most accuracy action
				action_index = np.argmax(QValue)
				action[action_index] = 1

			# decrease the episilon
			if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
				self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		# testing pattern of program
		elif self.pattern == 1:
			action_index = np.argmax(QValue)
			action[action_index] = 1

		return action

	# initialize the fisrt input data to CNN layers
	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

