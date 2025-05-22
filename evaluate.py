import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load the trained model
model = PPO.load("ppo_safety_point_circle0.zip")

# Create the environment
env = safety_gymnasium.make("SafetyPointCircle0-v0")

env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

env.close()