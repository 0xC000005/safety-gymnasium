import safety_gymnasium
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("ppo_safety_point_circle0.zip")

# Create the environment with rendering
env = safety_gymnasium.make("SafetyPointCircle0-v0", render_mode="human")

env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)


obs, info = env.reset()
terminated, truncated = False, False

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()