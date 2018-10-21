class Player:
    def __init__(self, x, y, terrain):
        self.x = x
        self.y = y
        self.terrain = terrain
        self.cv_action=[[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]

    def getposition(self):
        return [self.x, self.y]

    def action(self, xy_m):

        if self.terrain.MAP[self.y + self.cv_action[xy_m][1]][self.x + self.cv_action[xy_m][0]] != 0:
            self.x += self.cv_action[xy_m][0]
            self.y += self.cv_action[xy_m][1]
        
        return self.terrain.getreward()
