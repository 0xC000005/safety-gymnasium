import safety_gymnasium
from stable_baselines3 import PPO
env = safety_gymnasium.make('SafetyPointCircle0-v0')

env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1, device="cuda", progress_bar=True)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the model
model.save("ppo_safety_point_circle0")
env.close()



