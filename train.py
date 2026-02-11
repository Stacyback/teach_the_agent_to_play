from stable_baselines3 import PPO
from car_env import CarEnv

# Створюємо середовище
env = CarEnv()

# Створюємо модель (мозок)
# MlpPolicy - це тип нейромережі, яка підходить для наших 5 сенсорів
model = PPO("MlpPolicy", env, verbose=1)

print("Навчання почалося... AI зараз почне робити тисячі спроб.")
# Навчаємо 100 000 кроків. Це приблизно 2-5 хвилин на звичайному ноуті.
model.learn(total_timesteps=100000)

# Зберігаємо "інтелект" у файл
model.save("trained_car_model")
print("Готово! AI навчився. Тепер можна запускати демо.")