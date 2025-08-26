import robosuite as suite
import numpy as np

env = suite.make(
    env_name="Lift",
    robots=["Panda"],
    has_renderer=True,
    use_camera_obs=False,
    use_banana=True, 
)

obs = env.reset()
low, high = env.action_spec 
for _ in range(1000):
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
