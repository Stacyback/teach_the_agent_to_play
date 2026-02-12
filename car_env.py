import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        # Дії: 0: Вліво, 1: Вправо, 2: Газ, 3: Гальмо
        self.action_space = spaces.Discrete(4)
        # 5 променів-сенсорів
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Створення стабільної траси
        self.track = pygame.Surface((self.width, self.height))
        self.track.fill((34, 139, 34)) # Трава (зелена)
        # Малюємо дорогу (біле коло на чорному фоні)
        pygame.draw.circle(self.track, (200, 200, 200), (400, 300), 250)
        pygame.draw.circle(self.track, (34, 139, 34), (400, 300), 150)
        # Асфальтова лінія
        pygame.draw.circle(self.track, (50, 50, 50), (400, 300), 240, 80)

        # Малюнок машинки
        self.car_img = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(self.car_img, (255, 50, 50), (0, 0, 30, 15), border_radius=3)
        pygame.draw.rect(self.car_img, (50, 50, 80), (20, 2, 8, 11)) # Лобове скло

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
                dist += 3
                x = int(self.car_pos[0] + math.cos(rad) * dist)
                y = int(self.car_pos[1] - math.sin(rad) * dist)
                if 0 <= x < 800 and 0 <= y < 600:
                    # Якщо колір НЕ асфальт (50, 50, 50) - це перешкода
                    if self.track.get_at((x, y))[0] != 50: break
                else: break
            distances.append(dist / 150.0)
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        if action == 0: self.car_angle += 7
        if action == 1: self.car_angle -= 7
        if action == 2: self.speed = min(self.speed + 0.4, 5.0)
        if action == 3: self.speed = max(self.speed - 0.5, 0.0)

        rad = math.radians(self.car_angle)
        self.car_pos[0] += math.cos(rad) * self.speed
        self.car_pos[1] -= math.sin(rad) * self.speed

        obs = self._get_obs()
        # Винагорода за швидкість, якщо ми на трасі
        reward = self.speed * 0.2
        done = False
        
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        if not (0 <= x < 800 and 0 <= y < 600) or self.track.get_at((x,y))[0] != 50:
            done = True
            reward = -100 # Штраф за аварію

        return obs, reward, done, False, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()

        self.screen.blit(self.track, (0, 0))
        # Малюємо сенсори
        obs = self._get_obs()
        angles = [-45, -22.5, 0, 22.5, 45]
        for i, a in enumerate(angles):
            rad = math.radians(self.car_angle + a)
            dist = obs[i] * 150
            end_x = int(self.car_pos[0] + math.cos(rad) * dist)
            end_y = int(self.car_pos[1] - math.sin(rad) * dist)
            pygame.draw.line(self.screen, (0, 255, 0), (int(self.car_pos[0]), int(self.car_pos[1])), (end_x, end_y), 1)

        # Малюємо машинку
        rotated_car = pygame.transform.rotate(self.car_img, self.car_angle)
        rect = rotated_car.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
        self.screen.blit(rotated_car, rect.topleft)
        pygame.display.flip()
        self.clock.tick(60)