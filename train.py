from stable_baselines3 import PPO
from car_env import CarEnv
import os

# Створюємо середовище
env = CarEnv()

# Створюємо модель
# Використовуємо вищий learning_rate для швидкого навчання за 20 хв
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005)

print("--- НАВЧАННЯ РОЗПОЧАТО ---")
print("AI вчиться керувати за допомогою 5 сенсорів...")

# 200,000 кроків - це приблизно 15-20 хвилин на середньому ПК
model.learn(total_timesteps=200000)

# Зберігаємо результат
model.save("trained_car_model")
print("--- ГОТОВО! Модель збережена як trained_car_model.zip ---")