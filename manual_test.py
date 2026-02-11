from car_env import CarEnv
import pygame

env = CarEnv()
obs, _ = env.reset()
run = True

print("Керування: Стрілки або WASD. Закрийте вікно, щоб вийти.")

while run:
    # 1. Малюємо гру
    env.render()
    
    # 2. Обробка натискань клавіш (ручне керування)
    action = 3 # За замовчуванням стоїмо/гальмуємо
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]: action = 0
    if keys[pygame.K_RIGHT]: action = 1
    if keys[pygame.K_UP]: action = 2
    
    # 3. Крок гри
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 4. Перевірка виходу
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    if terminated:
        print("Аварія! Рестарт...")
        env.reset()

pygame.quit()