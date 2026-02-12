import gymnasium as gym
import pygame

# Створюємо середовище
env = gym.make("CarRacing-v3", render_mode="human", continuous=False)
obs, info = env.reset()

run = True
print("Керування: Стрілки клавіатури")

while run:
    action = 0 # Нічого не робити (Discrete 0)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]: action = 1 # Вправо
    if keys[pygame.K_LEFT]:  action = 2 # Вліво
    if keys[pygame.K_UP]:    action = 3 # Газ
    if keys[pygame.K_DOWN]:  action = 4 # Гальмо

    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()