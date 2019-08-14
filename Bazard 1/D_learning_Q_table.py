import numpy as np

#Class Environement, the origine of the repere is the top right corner
class Environement:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.map = np.zeros((x, y))

    def add_a_reward(self, x, y, reward):
        self.map[x][y] = reward

#Class Agent
class Agent:    
    def __init__(self,x, y, map ):
        self.map = map
        self.x = x
        self.y = y
        self.score = 0
        self.q_table = np.zeros((map.shape[0], map.shape[1], 4))
    def moove(self, number):  # top bot left right
        #We change our position
        if number == 0 :
                self.y -= 1
        elif number == 1 :
                self.y += 1
        elif number == 2 :
                self.x -= 1
        elif number == 3 :
                self.x += 1
        #We check if we're still inside the map
        if self.x < 0 :
            self.x = 0 
        if self.x > self.map.shape[0]-1 :
            self.x = self.map.shape[0]-1 
        if self.y < 0 :
            self.y = 0 
        if self.y > self.map.shape[1]-1 :
            self.y = self.map.shape[1]-1
        self.score += self.map[(self.x, self.y)]
        
    def explore(self, lr, gamma):
        s0 = (self.x, self.y)
        action = np.random.randint(4)
        self.moove(action)
        s1 =(self.x, self.y)
        reward = self.map[self.x][self.y]
        self.q_table[s0][action] = (1-lr)*self.q_table[s0][action] + lr*(self.map[s1] + gamma*np.max(self.q_table[s1]))
    
    def play(self):
        state = self.x, self.y
        self.moove(np.argmax(self.q_table[state]))
    def reset(self, x, y, total = False):
        self.score = 0
        self.x = x
        self.y = y
        if total :
            self.q_table = np.zeros((self.map.shape[0], self.map.shape[1], 4))





#Show an agent inside an environement 
def show(Agent, Environement): 
    array_to_print = Environement.map   
    for y in range(array_to_print.shape[1]):
        line =""
        for x in range(array_to_print.shape[0]):
            if(x == Agent.x and Agent.y == y):
                line += 'A' 
            else:
                line += str(array_to_print[x][y])
            line += "\t"
        print(line)


#Small 3*3 Environement and an agent
env = Environement(5, 5)
agent = Agent(4, 4, env.map)
#Let's design an interisting map
env.add_a_reward(1, 0, -2)
env.add_a_reward(1, 1, -2)
env.add_a_reward(1, 2, -2)
env.add_a_reward(1, 3, -2)

env.add_a_reward(3, 4, -2)
env.add_a_reward(3, 1, -2)
env.add_a_reward(3, 2, -2)
env.add_a_reward(3, 3, -2)

env.add_a_reward(4, 4, 1)


#We make somme test to know how gamma an lr affect the agent 
ratio = 0
lr = 0.1
gamma = 0.2
repetitions = 100
for b in range(1, 2):
    for n in range(repetitions):
        #To do that we test 10 training with the same setting and cunt how offens we won with those setting
        #training the agent few times
        for i in range(b*50):
            agent.explore(lr, gamma)

        agent.reset(0, 0)    
        #Now we play 16 times wihs the number of mouvement to ccomplete the lvl
        for i in range(16):
            #print("Step :" + str(i))
            agent.play()
            
        if agent.x == 4 and agent.y == 4 :
            ratio += 1/repetitions
        #print(agent.score)
        agent.reset(0, 0, True)
    print("Ratio : " + str(ratio) + "\tLearning rate : " + str(b*50) + "\tGamma : " + str(gamma))
    ratio = 0
    show(agent, env)
 
#-------------------------------------Bilan-----------------------------
#In this case the only setting who maters in the number of exploration the agent makes
#More he explores, better his ratio is , the learning rate and gamma dont affect his ratio








