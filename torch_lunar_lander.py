from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2', render_mode='human')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

agent.load_models()   # load models from ./tmp/ddpg/

score_history = []
n_games = 1000
for i in range(n_games):
    done = False
    score = 0
    obs = env.reset()[0]
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
            # we learn on every step (temporal difference learning method)
            # instead of at the end of every episode (Monte Carlo method)
        score += reward
        obs = new_state
    
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
          '100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()
    
filename = 'lunar-lander.png'
plotLearning(score_history, filename, window=100)