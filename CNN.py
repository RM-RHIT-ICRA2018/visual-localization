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
#from logger import Logger
import matplotlib.pyplot as plt

#logger = Logger('./logs')

game = DoomGame()
game.load_config("./final+sc.cfg")
game.init()

resolution = (60, 108)

actions = [[False,False, False]]
episodes=1000000
sleep_time=0.5
learning_rate=1e-3
conv_outdim=19968
Likelihood_map_dim=5*8
Likelihood_angel_dim=30

def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.drop = nn.Dropout2d(p=0.3)
        #self.pool = nn.MaxPool2d(4,stride=3)
        self.loca_fc1 = nn.Linear(conv_outdim, Likelihood_map_dim)
        self.ang_fc1 = nn.Linear(conv_outdim, Likelihood_angel_dim)
        self.softmax=nn.Softmax()
        nn.init.xavier_normal(self.conv1.weight)
        nn.init.xavier_normal(self.conv2.weight)
        nn.init.xavier_normal(self.loca_fc1.weight)
        nn.init.xavier_normal(self.ang_fc1.weight)

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

step=0

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()
    state = game.get_state()
    while (state.number<17):
        game.make_action(choice(actions),10)
        state = game.get_state()
    game.make_action(choice(actions),20)
    
    while not game.is_episode_finished():

        # Gets the state
        for j in range(4):
            state = game.get_state()

            image= preprocess(state.screen_buffer)
            if j==0:
                vars= state.game_variables
                total_image=image
            else:
                total_image=np.append(total_image,image,axis=1)

            #print(j)
            #print(state.game_variables)
            game.make_action(choice(actions),10)
        game.make_action(choice(actions),20)
        
        #plt.imshow(total_image)
        #plt.show()
        
        input_image = Variable(torch.from_numpy(total_image.reshape([1, 3, resolution[0], resolution[1]*4]))).cuda()
        pos,ang=model(input_image)
        
        #print(vars)
        #print(pos)
        #print(ang)

        true_map=torch.FloatTensor([[0 for j in range(8)] for i in range(5)])
        true_ang=torch.FloatTensor([0 for i in range(30)])
        print(vars[1]*8/547,',',vars[0]*5/342)
        true_map[math.floor(vars[1]*8/547)][math.floor(vars[0]*5/342)]=1
        true_map=Variable(true_map.view(-1,Likelihood_map_dim)).cuda()

        true_ang[math.floor(vars[2]/12)]=1
        true_ang=Variable(true_ang).cuda()

        loss_pos=criterion(pos,true_map)
        loss_ang=criterion(ang,true_ang)
        loss=loss_pos+loss_ang
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #step=step+1
        #info = {
        #    'loss': loss.data[0]
        #}
        #for tag, value in info.items():
        #    logger.scalar_summary(tag, value, step)
        print(loss_pos,loss_ang)
        
