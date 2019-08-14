import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import Sequential 
import os


#Create the class env
#The environement is the simple  2D game where the agent has to finf his way to the rewards 
#In that environement the state is defined by what is around him in an area of 3*3
class Environement :
    def __init__(self, width, height):
        #Here we stock our environement
        self.map = np.zeros((width, height, 3))
        #And here we got our rewards 
        self.x_agent = 0
        self.y_agent = 0   
        self.victory_reward = 2   

    def reset_agent(self):
        self.x_agent = 0
        self.y_agent = 0 
        
    def step(self, action):
        #We take the intial position of our agent, mahe the moovement and then the return reward 
        state = self.get_state()
        #We change our position checking if we're in a wall or out of our environement
        if action == 0 :   
            if state[1][0][0] != 1:
                self.y_agent -= 1                
        elif action == 1 :
            if state[1][2][0] != 1:
                self.y_agent += 1
        elif action == 2 :
            if state[0][1][0] != 1:
                self.x_agent -= 1
        elif action == 3 :
            if state[2][1][0] != 1:
                self.x_agent += 1

        #Now we check if we won, if we won we return our victory_reward and we reset our position to the origin 
        if self.map[self.x_agent][self.y_agent][1] == 1:
            self.x_agent = 0
            self.y_agent = 0
            return self.victory_reward
        #And we didnt won we jsut return the digital value wich will be me most of the time a zero 
        else:
            return self.map[self.x_agent][self.y_agent][2]
        

    def get_state(self):
        state = np.zeros((3, 3, 3))
        for width in range (3):
            for height in range(3):
                if self.x_agent-1+width >= 0 and self.x_agent-1+width < self.map.shape[0] and self.y_agent-1+height >= 0 and self.y_agent-1+height < self.map.shape[1] :
                    state[width][height] = self.map[self.x_agent-1+width][self.y_agent-1+height]
                else :
                    state[width][height] = (1, 0, 0) 
        return state 

    def add_a_wall(self,  x, y, object=1):
        #Put a char , W for wall, V for victory, numbers for a reward 
        self.map[x][y][1] = object
    def add_a_win(self, x, y, object=1):
        #Put a char , W for wall, V for victory, numbers for a reward 
        self.map[x][y][0] = object
    def add_a_reward(self,  x, y, object=1):
        #Put a char , W for wall, V for victory, numbers for a reward 
        self.map[x][y][2] = object
     
    

    def show(self, agent=False, state = False, map=False):
        if state:
            print(self.get_state())
        if map:
            print(self.map)  
        if agent:
            array_to_print = self.map 
            for y in range(array_to_print.shape[1]):
                line =""
                for x in range(array_to_print.shape[0]):
                    if(x == self.x_agent and y == self.y_agent):
                        line += 'Agent' 
                    else:
                        if array_to_print[x][y][1] == 1:
                            line += 'W'
                        elif array_to_print[x][y][0] == 1:
                            line += 'WIN'
                        elif array_to_print[x][y][2] == 0:
                            line += '_'
                        else:
                            line += str(array_to_print[x][y][2])
                    line += "\t"
                print(line)

def loss_function(y_true, y_pred, action):
    loss = np.math.sqrt(np.subtract(y_true, y_pred)*action) 
    return loss
    
#Here we define our agent who will learn to progress in our environement
class Agent:
    def __init__(self):
        #Here we'll create the brain of our agent, a deep neural network with Keras
        self.model = Sequential()
        self.q_targets =[]
        self.model.add(kl.Dense(10, input_dim = 27))
        self.model.add(kl.Activation('relu'))
        self.model.add(kl.Dense(4, input_dim = 10))
        self.model.add(kl.Activation('tanh'))
        self.model.compile(optimizer= "adam", 
                      loss = 'MSE', 
                      metrics=['accuracy'])
        
        

    def pick_action(self, state, eps):  #Epsilon is the probability that wa take a random action
        #we chose if we take a random action
        if np.random.rand() < eps :
            return np.random.randint(4)
        #Else we take the best action acording to our Neural network(NN) 
        else :
            return int(np.argmax(self.model.predict(np.reshape(state, (-1, 27) ))))
    
    def train_model(self, states0, states1, actions, rewards):
        length = np.min((len(actions), 1000))
        self.q_targets =[]
        prediction = self.model.predict(np.reshape(states0, (-1, 27)))
        

        for n in range (length):    

            action_mask = np.array([1, 1, 1, 1])
            action_mask = np.logical_xor(action_mask, actions[n])
            pred = self.model.predict(np.reshape(states1[n], (-1, 27)))
            #print(pred.shape)
            q_target = prediction[n] * action_mask + actions[n] *(rewards[n]) # + 0.99*pred[0][np.argmax(pred)] ) 
            self.q_targets.append(q_target)
            print(states0[n], actions[n], rewards[n], q_target)
        
        self.model.fit(np.reshape(states0, (length, 27)),np.array(self.q_targets), batch_size=1, )
    

        



#Methode wich will insert randomly a valu in our list
def random_insert(sts0, st0, sts1, st1, acts, act, rewards, reward):
    
    
    sts0.append(st0)
    sts1.append(st1)
    acts.append(act)
    rewards.append(reward)
    
    return sts0, sts1, actions, rewards

#We create an environement and an agent, then we put walls in our environement and finaly we put a reward
agent = Agent()
env = Environement(3, 3)
env.add_a_wall(1, 1)  #Minus 1 is the wall accordingt to our environement
env.add_a_win(2, 2)             #Victory at the right bottom corner
env.add_a_reward(1, 2, object=-1)             #Negative reward in the bottom 


#Now let's play our game 
#let's create our data set to train our network
statet0 = []
statet1 = []
actions = []
rewards = []
#Exploration wich will be decreased 
epsilon = 1
st0 = env.get_state()
for epi in range (50):

    for step in range(4):
        #We save our value 
        act = agent.pick_action(st0, epsilon)
        action = np.zeros(4)
        action[act] = 1
        
        #print(env.get_state())
        #print(env.x_agent, env.y_agent)
        reward = env.step(act)
        st2 = env.get_state()
        #env.show(agent=True)
        #print("--------------------------------------------------")
        #We save those values in our liste and insert them randomly
        statet0, statet1, actions, rewards = random_insert(statet0, st0, statet1, st2, actions, action, rewards, reward)

        #If we have data set > 10 000 , we delete the first data
        if(len(statet0)>1000):
            statet0.pop(0)
            statet1.pop(0)
            actions.pop(0)
            rewards.pop(0)
            
        #We update the actual state as the last "next step"
        st0 = st2
        epsilon = np.max((epsilon*0.99, 0.05)) 
    if epi % 5 == 0 :
        
        agent.train_model(statet0,statet1, actions, rewards)
        q_values = []
        for x in range (3):
            for y in range (3):
                env.x_agent = x
                env.y_agent = y 
                q_values.append(agent.model.predict(np.reshape(env.get_state(), (-1, 27) )))
        print(q_values)

    env.x_agent = 0
    env.y_agent = 0




#let s play to see how bad he is
env.reset_agent()
env.show(agent=True)


"""
print("--------------------------------------------------")
for i in range(5):
    env.step(agent.pick_action(env.get_state(), 0))
    env.show(agent=True)
    print("--------------------------------------------------")

"""









