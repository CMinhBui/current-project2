import matplotlib.pyplot as plt

import random
import numpy as np
from random import randint
try:
    from .sxsy import SXSY
    from .map import ENV_MAP
    from .controller import Player
except ModuleNotFoundError:
    from sxsy import SXSY
    from map import ENV_MAP
    from controller import Player

class Terrain:
    def __init__(self, map_index, use_laser = False):
        self.MAP = ENV_MAP[map_index]['map']
        self.map_array = np.array(self.MAP, dtype = int)
        self.reward_locs = [list(z) for z in  zip(np.where(self.map_array == 3)[1].tolist(), np.where(self.map_array == 3)[0].tolist())]
        
        self.state_space = [list(z) for z in  zip(np.where(self.map_array != 0)[1].tolist(), np.where(self.map_array != 0)[0].tolist())]

        self.state_to_index = np.zeros_like(self.MAP) - 1
        
        for i, s in enumerate(self.state_space):
            self.state_to_index[s[1]][s[0]] = i
        
        assert np.sum(self.state_to_index != -1) == len(self.state_space)

        self.action_size = 8
        self.reward_range = 1.0
        self.reward_goal = 1.0
        
        self.num_task = len(self.reward_locs)

        if not use_laser:
            self.cv_state_onehot = np.identity(len(self.state_space), dtype=int)
        else:
            self.cv_state_onehot = [0] * len(self.state_space)

            for s in self.state_space:
                state_index = self.state_to_index[s[1]][s[0]]
                self.cv_state_onehot[state_index] = np.array(self.laser(s[0], s[1]))

            self.cv_state_onehot = np.asarray(self.cv_state_onehot)
            # self.cv_state_onehot = self.cv_state_onehot

        self.cv_action_onehot = np.identity(self.action_size, dtype=int)
        self.cv_task_onehot = np.identity(len(self.reward_locs), dtype=int)
        
        self.min_dist = []
        for i in range(self.num_task):
            self.min_dist.append(self.cal_min_dist(i))

    def cal_min_dist(self, task_idx):
        distance = np.zeros_like(self.map_array)
        target = [list(z) for z in  zip(np.where(self.map_array == 3)[0].tolist(), np.where(self.map_array == 3)[1].tolist())][task_idx]
        move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

        visisted = {}
        for i in range(distance.shape[0]):
            for j in range(distance.shape[1]):
                visisted[i, j] = False

        queue = []
        queue.append(target)

        while len(queue) > 0:
            pos = queue[0]
            queue = queue[1:]

            visisted[pos[0], pos[1]] = True

            for m in move:

                neighbor = [pos[0] + m[0], pos[1] + m[1]]

                if self.map_array[neighbor[0], neighbor[1]] == 0:
                    continue 

                if not visisted[neighbor[0], neighbor[1]]:
                    distance[neighbor[0], neighbor[1]] = distance[pos[0], pos[1]] + 1           
                    queue.append(neighbor)
                    visisted[neighbor[0], neighbor[1]] = True

        return distance

    def laser(self, x, y):
        assert self.map_array[y][x] != 0
        move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        res = []
        for mv in move:
            nx, ny = x + mv[0], y + mv[1]
            temp = 0
            while self.map_array[ny][nx] > 0:
                nx, ny = nx + mv[0], ny + mv[1]
                temp += 1

            res.append(temp)

        return res

    def getreward(self):
        done = False
        reward = -0.01

        x_pos, y_pos = self.reward_locs[self.task]

        if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
            reward = self.reward_goal
            done = True

        return reward, done

    def checkepisodeend(self):
        for x_pos, y_pos in self.reward_locs:
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                return 1
        return 0

    def plotgame(self):

        plt.clf()
        plt.xlim([-1, self.map_array.shape[1]])
        plt.ylim([-1, self.map_array.shape[0]])

        for y in range(self.map_array.shape[0]):
            for x in range(self.map_array.shape[1]):
                if self.MAP[y][x] == 0:
                    plt.scatter(x, y, marker='x', color="red")

        # for (x, y) in self.state_space:
        #     plt.scatter(x, y, marker='o', color="green", s = 5)

        # rands = self.state_space[np.random.choice(range(len(self.state_space)))]
        # plt.annotate(str(self.laser(*rands)), (rands[0], rands[1]))

        # for x_pos, y_pos in self.reward_locs:
        #     plt.scatter(x_pos, y_pos, marker='o', color="blue")

        # count = np.load('count.npy')
        # for i in range(count.shape[0]):
        #     for j in range(count.shape[1]):
        #         if count[i][j] > 0:                    
        #             plt.annotate(int(count[i][j]), (i, j))

        vmin = 0
        vmax = 400
        pts = SXSY[4]
        s = [l for ep in pts for l in ep]
        # s = [l for ep in pts for l in sorted(ep, key=lambda element: (element[1], element[0]))]

        s = sorted(s, key=lambda element: (element[1], element[0]))
        
        init_count = np.zeros_like(self.state_to_index, dtype = np.float32)
        for i, (x, y) in enumerate(s):
            init_count[y][x] += 10 * 0.99 ** np.sqrt(i)

        # print(init_count)
        max_cnt, min_cnt = np.max(init_count), np.min(init_count)
        sx, sy, cxy, sxy = [], [], [], []
        for i in range(init_count.shape[0]):
            for j in range(init_count.shape[1]):
                if init_count[i][j] > 0:
                    sx.append(j)
                    sy.append(i)
                    cxy.append((init_count[i][j] - min_cnt) * (vmax - vmin) / (max_cnt - min_cnt) + vmin)
                    sxy.append(init_count[i][j]**5 * 200 / np.max(init_count)**5)
        
        cm = plt.cm.get_cmap('jet')
        sc = plt.scatter(sx, sy, c = cxy, vmin = vmin, vmax = vmax, s = sxy, cmap = cm)
        plt.colorbar(sc)

        # # plt.scatter([self.player.x,], [self.player.y,], marker='x', color="red")
        # # plt.pause(0.001)
        plt.show()
        # plt.savefig('s2')

    def resetgame(self, task, sx, sy):
        #self.player = Player(7, 1, self)
       
        self.player = Player(sx, sy, self)

        self.task = task
            
if __name__ == '__main__':
    ter = Terrain(4)
    print(ter.min_dist[0])
    print(ter.min_dist[1])