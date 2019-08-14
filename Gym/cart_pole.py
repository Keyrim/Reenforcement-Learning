import gym
env = gym.make("CartPole-v0")
env.reset()
for e in range(100):
    env.render()
    env.step(env.action_space.sample())
    print(env.state)

env.close()

