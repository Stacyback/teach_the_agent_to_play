from stable_baselines3 import PPO
from car_env import CarEnv
import time

env = CarEnv()
model = PPO.load("ppo_self_driving_car")

obs, _ = env.reset()
print("AI КЕРУЄ МАШИНОЮ. ДИВІТЬСЯ НА ЕКРАН!")

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        print("Аварія! Спроба №", int(time.time())%100)
        obs, _ = env.reset()