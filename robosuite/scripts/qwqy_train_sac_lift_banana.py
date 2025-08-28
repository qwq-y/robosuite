import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import robosuite as suite
from robosuite.controllers import controller_factory
import os

import wandb
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def record_video(env, actor, max_len=200, video_path="rollout.gif"):
    frames = []
    obs = env.reset()
    obs_vec = obs["robot0_proprio-state"]
    for _ in range(max_len):
        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = actor.sample(obs_tensor)
        obs, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
        obs_vec = obs["robot0_proprio-state"]

        frame = env.sim.render(width=640, height=480, camera_name="agentview", depth=False)
        frames.append(frame)
        if done:
            break
    imageio.mimsave(video_path, frames, fps=20)

    # upload video to wandb
    wandb.log({"eval_video": wandb.Video(video_path, fps=20, format="mp4")})

# === ENVIRONMENT SETUP ===
env = suite.make(
    env_name="Lift",
    robots=["Panda"],
    use_banana=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,       
    camera_names="agentview",   
    camera_heights=256,
    camera_widths=256,
    render_camera="agentview",  
    reward_shaping=True,
    control_freq=20,
)

obs_dim = env.observation_spec()["robot0_proprio-state"].shape[0]
act_dim = env.action_spec[0].shape[0]
print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

# === REPLAY BUFFER ===
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

# === ACTOR NETWORK ===
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

actor = GaussianPolicy(obs_dim, act_dim).to(device)
print("Actor network initialized.")

# === CRITIC NETWORKS ===
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        q1_val = self.q1(xu)
        q2_val = self.q2(xu)
        return q1_val, q2_val

critic = QNetwork(obs_dim, act_dim).to(device)
critic_target = QNetwork(obs_dim, act_dim).to(device)
critic_target.load_state_dict(critic.state_dict())
print("Critic networks initialized.")

# === OPTIMIZERS & ALPHA === 
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

target_entropy = -act_dim
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

def get_alpha():
    return log_alpha.exp().item()

print(f"Optimizers initialized. Target entropy: {target_entropy}")

# === TRAINING SETTINGS ===
total_steps = int(1e5)
start_steps = 1000
update_after = 1000
update_every = 50
batch_size = 256
gamma = 0.99
polyak = 0.995

replay_buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6))

wandb.init(
    project="sac-lift-banana",
    name="run-01",
    mode="offline", 
    dir="/home/qwqy/Data/rlgs",
    config={
        "total_steps": total_steps,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "env": "Lift",
        "robot": "Panda",
    }
)
wandb.init(project="sac-lift-banana", name=f"mesh_low_run{wandb.run.id}") 

# === MAIN TRAINING LOOP ===
obs, ep_ret, ep_len = env.reset()["robot0_proprio-state"], 0, 0

for t in range(total_steps):
    # === SELECT ACTION ===
    if t < start_steps:
        act = np.random.uniform(env.action_spec[0], env.action_spec[1])
    else:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_tensor, _ = actor.sample(obs_tensor)
            act = act_tensor.cpu().numpy()[0]

    # === STEP ENV ===
    next_obs_all, reward, done, _ = env.step(act)
    next_obs = next_obs_all["robot0_proprio-state"]
    ep_ret += reward
    ep_len += 1

    # === STORE INTO BUFFER ===
    replay_buffer.store(obs, act, reward, next_obs, done)
    obs = next_obs

    # === END OF EPISODE ===
    if done:
        print(f"[step={t}] Episode done. Return: {ep_ret:.2f}, Length: {ep_len}")
        obs, ep_ret, ep_len = env.reset()["robot0_proprio-state"], 0, 0

    # === UPDATE NETWORKS ===
    if t >= update_after and t % update_every == 0:
        for _ in range(update_every):
            batch = replay_buffer.sample_batch(batch_size)
            obs_b = batch["obs"]
            act_b = batch["act"]
            rew_b = batch["rew"].unsqueeze(1)
            next_obs_b = batch["next_obs"]
            done_b = batch["done"].unsqueeze(1)

            # --- Critic update ---
            with torch.no_grad():
                next_act, next_logp = actor.sample(next_obs_b)
                target_q1, target_q2 = critic_target(next_obs_b, next_act) 
                target_q = torch.min(target_q1, target_q2) - get_alpha() * next_logp
                target = rew_b + gamma * (1 - done_b) * target_q

            current_q1, current_q2 = critic(obs_b, act_b)
            q1_loss = F.mse_loss(current_q1, target)
            q2_loss = F.mse_loss(current_q2, target)
            critic_loss = q1_loss + q2_loss

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # --- Actor update ---
            act_new, logp_new = actor.sample(obs_b)
            q1_pi, q2_pi = critic(obs_b, act_new)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (get_alpha() * logp_new - q_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Alpha update ---
            alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- Target network update ---
            with torch.no_grad():
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.mul_(polyak).add_((1 - polyak) * param.data)
    
        if t % 1000 == 0:
            print(f"[step={t}] actor_loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}, alpha: {get_alpha():.4f}")
        
            wandb.log({
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "alpha": get_alpha(),
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "step": t
            })

        if t % 10000 == 0 or t == total_steps - 1:    
            run_dir = f"/home/qwqy/Data/rlgs/train/lift_banana/mesh/{wandb.run.id}"
            # run_dir = f"/home/qwqy/Data/rlgs/train/lift_banana/mesh/test"
            os.makedirs(run_dir, exist_ok=True)
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_optimizer.state_dict(),
                "critic_opt": critic_optimizer.state_dict(),
                "log_alpha": log_alpha,
                "alpha_opt": alpha_optimizer.state_dict(),
            }, f"{run_dir}/sac_banana_step{t}.pt")

            record_video(env, actor)

