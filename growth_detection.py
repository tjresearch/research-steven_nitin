import os
import sys
import pickle
import argparse
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class LR(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.model(x)

count = 0
a_list = []
b_list = []
time_data = torch.FloatTensor( [[2], [0], [3], [4], [1], [6], [5]])
bottle_height = 0.155956 # meter
debug= False

model = LR(1, 1)
w, b = model.parameters()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


def get_minmax(cnt):
    """get minmax of each contour (x and y)
    """
    minx = 1000000
    maxx = 0
    miny = 100000
    maxy = 0
    for j in range(len(cnt)):
        x = cnt[j][0][0]
        y = cnt[j][0][1]
        if x < minx: minx = x
        if x > maxx: maxx = x
        if y < miny: miny = y
        if y > maxy: maxy = y
    if debug:
        print ("contour xrange: [{}-{}], yrange: [{},{}], width: {}".format(minx, maxx, miny, maxy, maxx-minx))
    return [minx, maxx, miny, maxy]

def read_pickle(pfile):
    """read pickle file and figure out size of bottle and plant
    """
    gap_y = 300
    date = os.path.split(pfile)[1][:-2]
    plant_type = date[:1]+date[-1:]
    datestr = date[1:-1]
    t = time.mktime(datetime.strptime(datestr, '%m%d%y').timetuple())
    print("{},{},".format(int(t), plant_type), end ="")
    cnts = pickle.load(open(pfile, 'rb'))
    print ("number of contours: {}".format(len(cnts)))
    last_y = -1
    bottom_bottle = -1 
    top_bottle = -1 
    # go through each contour find all contours belong to bottle
    # there must be a gap between bottle and plant
    bottle_min = 1000000
    bottle_max = 0
    found = False
    bottle_size = 0
    i = 0
    while (bottle_size == 0):
        for i in range(len(cnts)):
            cnt = cnts[i]
            minx, maxx, miny, maxy = get_minmax(cnt)
            # if bottom is not found then take this contour
            if bottom_bottle == -1:
                if last_y == -1:
                    bottom_bottle = miny
            else:
                if top_bottle == -1:
                    if last_y - miny > gap_y:
                        top_bottle = last_y
                        bottle_size = bottle_max-bottle_min
                        break
            if miny < bottle_min: bottle_min = miny
            if maxy > bottle_max: bottle_max = maxy
            last_y = miny     
        if debug:
            print ("bottle bottom: {}, top: {}, size: {}".format(bottle_max, bottle_min, bottle_size))
            print("i: {}".format(i))
        if bottle_size == 0:
            gap_y -= 10
            continue
        # find the plant

        bottom_plant = -1
        top_plant = -1
        last_y = -1
        plant_min = 10000000
        plant_max = 0
        plant_size = 0
        minx, maxx, miny, maxy = 0,0,0,0
        for j in range(i, len(cnts)):
            cnt = cnts[j]
            minx, maxx, miny, maxy = get_minmax(cnt)
            # if bottom is not found then take this contour
            if bottom_plant == -1:
                if last_y == -1:
                    bottom_plant = miny
            if miny < plant_min: plant_min = miny
            if maxy > plant_max: plant_max = maxy
        top_plant = miny
        plant_size = plant_max-top_plant
        if debug:
            print ("plant bottom: {}, top: {}, size: {}".format(plant_max, top_plant, plant_size))
        print("plant_size {:.4f}".format((plant_size*1.0/bottle_size)*bottle_height))
        if(plant_type == 'Ac'):
            a_list.append([(plant_size*1.0/bottle_size)*bottle_height])
        else:
            b_list.append([(plant_size*1.0/bottle_size)*bottle_height])

def display():
    plt.plot(time_data.numpy(), a_list.numpy(), 'o')
    plt.show()

def get_param():
    return(w[0][0].item(), b[0].item())

def plot_best_fit(title, range, filename):
    plt.title = title
    w1, b1 = get_param()
    x1 = np.array(range)
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    print(w1)
    print(b1)
    display()
    plt.savefig(filename)

def regression():
    epochs = 10000
    losses = []
    for i in range(epochs):
        y_pred = model.forward(time_data)
        loss_value = criterion(y_pred, a_list)
        if i%100 == 0:
            print('epoch {}, loss: {}'.format(i, loss_value))
        losses.append(loss_value)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--contour", dest="contour",
                help="contour file")
    parser.add_argument("-f", "--folder", dest="folder",
                help="folder of contour files")

    args = parser.parse_args()

    contour_files = []
    if args.folder is not None:
        #debug=True
        for f in  os.listdir(args.folder):
            contour_files.append(os.path.join(args.folder, f))
    elif args.contour is not None:
        #debug=True
        contour_files.append(args.contour)
    else:
        print("Must specified contour file or folder")
        parser.print_help(sys.stderr)
        exit(1)

    for f in contour_files:
        print(f)
        read_pickle(f)
        count+=1
    a_list = torch.FloatTensor(a_list)
    b_list = torch.FloatTensor(b_list)
    print(time_data)
    print(a_list)
    print(b_list)      
    plot_best_fit("initial", [0, 6], "output.png")
    regression()
    plt.clf()
    plot_best_fit("trained", [0, 6], "output1.png")
