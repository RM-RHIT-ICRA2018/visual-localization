from vizdoom import *
from random import choice
from time import sleep
import skimage.color, skimage.transform
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
import math

game = DoomGame()
game.load_config("./final+sc.cfg")
game.init()

resolution = (200, 160)

actions = [[True, False], [False, True]]
episodes=100000
sleep_time=0.5
learning_rate=0.00025
conv_outdim=11040
Likelihood_map_dim=50*80
Likelihood_angel_dim=72

def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=9, stride=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.loca_fc1 = nn.Linear(conv_outdim, Likelihood_map_dim)
        self.ang_fc1 = nn.Linear(conv_outdim, Likelihood_angel_dim)
        self.softmax=nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, conv_outdim)
        pos = self.softmax(self.loca_fc1(x))
        ang = self.softmax(self.ang_fc1(x))
        return pos,ang
        

model=Net().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()
        image= preprocess(state.screen_buffer)
        image = Variable(torch.from_numpy(image.reshape([1, 1, resolution[0], resolution[1]]))).cuda()
        pos,ang=model(image)
        vars= state.game_variables
        
        true_map=torch.FloatTensor([[0 for j in range(80)] for i in range(50)])
        true_ang=torch.FloatTensor([0 for i in range(72)])
        true_map[math.floor(vars[0]/100)][math.floor(vars[1]/100)]=1
        true_map=Variable(true_map.view(-1,Likelihood_map_dim)).cuda()

        true_ang[math.floor(vars[2]/5)]=1
        true_ang=Variable(true_ang).cuda()

        loss=criterion(pos,true_map)+criterion(ang,true_ang)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Makes a random action and get remember reward.
        game.make_action(choice(actions),10)

        print(loss)
