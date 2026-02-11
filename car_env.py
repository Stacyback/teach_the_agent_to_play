import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Траса
        self.track = pygame.Surface((self.width, self.height))
        self.track.fill((0, 0, 0))
        pygame.draw.circle(self.track, (255, 255, 255), (400, 300), 250)
        pygame.draw.circle(self.track, (0, 0, 0), (400, 300), 150)

        # Спрайт машинки (малюємо прямокутник, якщо немає файлу)
        self.car_img = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(self.car_img, (255, 200, 0), (0, 0, 30, 15), border_radius=3)
        pygame.draw.rect(self.car_img, (0, 0, 0), (20, 2, 8, 11)) # Скло

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_pos = [400.0, 100.0]
        self.car_angle = 0.0
        self.speed = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        distances = []
        angles = [-45, -22.5, 0, 22.5, 45]
        for a in angles:
            dist = 0
            rad = math.radians(self.car_angle + a)
            while dist < 150:
                dist += 2
                x = int(self.car_pos[0] + math.cos(rad) * dist)
                y = int(self.car_pos[1] - math.sin(rad) * dist)
                if 0 <= x < 800 and 0 <= y < 600:
                    if self.track.get_at((x, y))[0] == 0: break
                else: break
            distances.append(dist / 150.0)
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        if action == 0: self.car_angle += 7
        if action == 1: self.car_angle -= 7
        if action == 2: self.speed = min(self.speed + 0.3, 5)
        if action == 3: self.speed = max(self.speed - 0.3, 0)

        rad = math.radians(self.car_angle)
        self.car_pos[0] += math.cos(rad) * self.speed
        self.car_pos[1] -= math.sin(rad) * self.speed

        obs = self._get_obs()
        reward = self.speed * 0.5
        done = False
        
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        if x < 0 or x >= 800 or y < 0 or y >= 600 or self.track.get_at((x,y))[0] == 0:
            done = True
            reward = -50

        return obs, reward, done, False, {}

    def render(self):
        # Обробка подій, щоб вікно не зависало
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((50, 50, 50))
        self.screen.blit(self.track, (0, 0))

        # ВІЗУАЛІЗАЦІЯ СЕНСОРІВ
        obs = self._get_obs()
        angles = [-45, -22.5, 0, 22.5, 45]
        for i, a in enumerate(angles):
            rad = math.radians(self.car_angle + a)
            dist = obs[i] * 150
            # Перетворюємо в int, щоб не було TypeError
            start_p = (int(self.car_pos[0]), int(self.car_pos[1]))
            end_x = int(self.car_pos[0] + math.cos(rad) * dist)
            end_y = int(self.car_pos[1] - math.sin(rad) * dist)
            pygame.draw.line(self.screen, (0, 255, 0), start_p, (end_x, end_y), 1)

        # МАШИНКА
        rotated_car = pygame.transform.rotate(self.car_img, self.car_angle)
        rect = rotated_car.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
        self.screen.blit(rotated_car, rect.topleft)
        
        pygame.display.flip()
        self.clock.tick(60)