import safety_gymnasium
from stable_baselines3 import PPO
import os
env = safety_gymnasium.make('SafetyPointButtonMO0-v0')

# Disable reward clipping for vector rewards to avoid constant warnings
env.task.reward_conf.reward_clip = None

env = safety_gymnasium.wrappers.SafetyMOGymnasium2SafetyGymnasium(env)  
env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1, device="cuda")

# Train the agent
model.learn(total_timesteps=100_000, progress_bar=True)

# Save the model
model.save("SafetyPointButtonMO0-v0")
env.close()



