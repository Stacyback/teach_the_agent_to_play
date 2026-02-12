import time
import os
from stable_baselines3 import PPO
from car_env import CarEnv

def main():
    env = CarEnv()

    # Перевірка, чи модель вже існує
    model_path = "trained_car_model.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("Модель успішно завантажена! Дивись, як AI веде машину...")
    else:
        print("Помилка: Файл моделі не знайдено! Спочатку запусти train.py.")
        return

    obs, _ = env.reset()
    
    while True:
        # AI прогнозує дію на основі сенсорів
        action, _states = model.predict(obs, deterministic=True)
        
        # Крок у середовищі
        obs, reward, done, truncated, info = env.step(action)
        
        # Візуалізація (малювання)
        env.render()
        
        if done:
            print("Рестарт після виїзду за межі...")
            time.sleep(0.5)
            obs, _ = env.reset()

if __name__ == "__main__":
    main()