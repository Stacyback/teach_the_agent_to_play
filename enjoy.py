from stable_baselines3 import PPO
from car_env import CarEnv
import time

# 1. Створюємо середовище
env = CarEnv()

# 2. Завантажуємо навчену модель (впевнись, що назва файлу збігається)
# Якщо файл називається trained_car_model.zip, пиши "trained_car_model"
try:
    model = PPO.load("trained_car_model")
    print("Модель успішно завантажена! Починаємо демонстрацію...")
except:
    print("Помилка: Файл моделі не знайдено. Перевір назву .zip файлу!")
    exit()

# 3. Цикл демонстрації
obs, _ = env.reset()
run = True

while run:
    # Запитуємо у AI дію на основі сенсорів
    # deterministic=True означає, що AI буде діяти максимально впевнено
    action, _states = model.predict(obs, deterministic=True)
    
    # Виконуємо дію в середовищі
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Візуалізуємо
    env.render()
    
    # Якщо сталася аварія — просто скидаємо і починаємо знову
    if terminated or truncated:
        time.sleep(0.5) # Пауза на півсекунди, щоб ми побачили момент аварії
        obs, _ = env.reset()

    # Можливість закрити вікно
    import pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()