'''
 -------------------------
 COMP219-Ass2-Task2
 DQN for OpanAI video game
 Date: 6-Jan-2019
 -------------------------
'''

import cv2
import gym
import universe
from universe import wrappers
from algorithm import DQN
import numpy as np


# some useful parameters
LIFE_CHECKPOINT = 0
SHOOT_CHECKPOINT = 0
GAME_START_FLAG = True
# manually design the aciton in the game to make agorithm file can be used into other situations
action_list = [[('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'Z', False)],
				[('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'Z', False)],
				[('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'Z', False)],
				[('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'Z', True)]
				]

# preprocess raw image to 170*150 gray image
def preprocess(observation):
	# get original image from universe
	observation = np.array(observation[0]['vision'], dtype = np.uint8)
	# get useful area from image and do some processes
	observation = observation[25:195,5:155,:]
	observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(170,150,1))
	
# translate action index command to game aciton command
def action_process(action_cmd):
	num=0
	for cmd in range(4):
		if action_cmd[cmd] == 1:
			num = cmd
			break
	print(action_cmd)
	return [action_list[num]]

# check whether environment still provides image information to avoid NoneType error
def check_image(observation):
	if observation[0] is not None:
		return True
	else:
		return False

# manually adjustment of reward value from environment to make algorithm run better
def get_reward(info, reward):
	global GAME_START_FLAG
	global SHOOT_CHECKPOINT
	global LIFE_CHECKPOINT
	if reward > 0:
		reward = 20
	
	if reward == 0:
		reward = -0.02


	if GAME_START_FLAG:
		if 'ale.lives' in info['n'][0]:
			GAME_START_FLAG = False
		reward = 0
		return reward
	else:
		if 'ale.lives' in info['n'][0] is not None:
			if info['n'][0]['ale.lives'] != LIFE_CHECKPOINT:
				if LIFE_CHECKPOINT == 0:
					LIFE_CHECKPOINT = 3
					reward = 0
				else:
					LIFE_CHECKPOINT = info['n'][0]['ale.lives']
					reward = -80
			return reward
		else:
			if LIFE_CHECKPOINT != 0:
				LIFE_CHECKPOINT = 0
				reward = -80
			else:
				reward = 0
			return reward
			

def start():
	# initialize algorithm
	actions = 4	# there are 4 actions in this game
	DQNetwork = DQN(actions, 1)# 1 means testing pattern


	action0 = [action_list[0]]  # do nothing
	# wait for game start running actually
	while True:
		observation0, reward0, terminal , info = env.step(action0)	
		if observation0[0] is not None:
			break
		env.render()

	# initialize the first state of program with some operation of the first image
	observation0 = np.array(observation0[0]['vision'], dtype = np.uint8)
	observation0 = observation0[25:195,5:155,:]
	observation0 = cv2.cvtColor(observation0, cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	DQNetwork.setInitState(observation0)

	# start run the game and algorithm
	while True:
		# get aciton from network
		action = DQNetwork.getAction()
		# give the action to environment and get back information
		nextObservation,reward,terminal, info = env.step(action_process(action))
		# check whether get image
		if check_image(nextObservation):
			# do the iamge preprocess
			nextObservation = preprocess(nextObservation)
			# get reward the be adjusted
			reward = get_reward(info, reward[0])
			# use information to train the network
			DQNetwork.setPerception(nextObservation,action,reward,terminal)
		env.render()


if __name__ == '__main__':
	env = gym.make('gym-core.SpaceInvaders-v0') # start game in environment
	env = wrappers.experimental.SafeActionSpace(env)
	env.configure(remotes=1)
	observation0 = env.reset()
	env.render()
	start() # start the main part of program
