from tkinter import *
import numpy as np
import random

class Environnement(object):

    def __init__(self):
         super(Environnement, self).__init__()

         #lecture du fichier
         text = open("décor.txt")
         tab = text.readlines ()
         text.close ()

         #Ã©criture de la grille
         grille = []
         for i in range (len(tab)) :
            #print(len(tab[i]))
            ligne = []
            for j in range (len(tab[i])) :
               case = -1
               #print(tab [i][j])
               if tab [i][j] == 'X' :
                  Fond.create_rectangle(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'red')
               elif tab [i][j] == 'T' :
                  Fond.create_rectangle(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'blue')
               elif tab [i][j] == 'H' :
                  Fond.create_rectangle(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'yellow')
               elif tab [i][j] == 'P' :
                  Fond.create_oval(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'red')
               elif tab [i][j] == 'C' :
                  Fond.create_oval(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'black')
               elif tab [i][j] == 'S' :
                  Fond.create_oval(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'green')
               else :
                  case = 0
                  if random.uniform(0,1) < 0.05 :
                    case = 1
                    Fond.create_oval(j*12.5, i*12.5, j*12.5+12.5, i*12.5+12.5, fill = 'yellow')
               ligne.append(case)
            grille.append(ligne)


         self.grid = np.array(grille)

         # Starting position
         self.y = 2
         self.x = 2

         self.actions = [
            [-1, 0], # Up
            [1, 0], #Down
            [0, -1], # Left
            [0, 1] # Right
         ]

    def reset(self):

        self.y = 2
        self.x = 2
        return [self.x, self.y]

    def step(self, action):

         new_Y = self.y + self.actions[action][0]
         new_X = self.x + self.actions[action][1]

         reward = self.grid[new_Y][new_X]

         if self.grid[new_Y][new_X] != -1 :
            self.x = new_X
            self.y = new_Y

         return [self.x, self.y] , reward

    def get_grille(self):
         return self.grid

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

    def get_state(self) :
        return [self.x, self.y]

    def get_coord(self) :
        return [self.x, self.y]


def clavier(evt) :
    global env, Fond
    direction = evt.keysym
    act = -1
    if (direction == "Right"):
        act = 3
    elif (direction == "Down"):
        act = 1
    elif (direction == "Left"):
        act = 2
    elif (direction == "Up"):
        act = 0

    if act != -1 :
      pos, rew = env.step(act)
      X = pos[0]
      Y = pos[1]
      print(rew)
      Fond.coords(perso, X*12.5, Y*12.5, X*12.5+12.5, Y*12.5+12.5,)
      Fond.update()






fen = Tk ()
fen.title ("Jeu")
fen.geometry ("500x500")
Fond=Canvas(fen, width=500, height=500, bg="white")
Fond.grid()

env = Environnement()

pos = env.get_coord()
X = pos[0]
Y = pos[1]

perso = Fond.create_oval(X*12.5, Y*12.5, X*12.5+12.5, Y*12.5+12.5, fill = 'blue')



fen.bind_all('<Key>', clavier)
fen.mainloop ()