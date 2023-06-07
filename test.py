import torch
import gymnasium as gym

env = gym.make('Ant-v4', render_mode='human')
globalActor = torch.load('TIF360_Project/Actor')

state = torch.tensor(env.reset()[0])
done = False
final_pol_reward = 0

for _ in range(1000):
    env.render()
    action = globalActor.select_action(state.float())
    print(action, type(action), action.shape)
    state, reward, done, _, _ = env.step(action)
    state = torch.tensor(state)
    final_pol_reward += reward

    if done:
        break
    