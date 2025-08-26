import numpy as np
import robosuite as suite

# --- env ---
CTRL = {"type": "OSC_POSE"}  # remove this line if unsupported
try:
    env = suite.make(
        env_name="Lift",
        robots=["Panda"],
        controller_configs=CTRL,
        has_renderer=True,
        use_camera_obs=False,
        use_banana=True,
        horizon=500,
        control_freq=20,
    )
except Exception:
    env = suite.make(
        env_name="Lift",
        robots=["Panda"],
        has_renderer=True,
        use_camera_obs=False,
        use_banana=True,
        horizon=500,
        control_freq=20,
    )

obs = env.reset()
low, high = env.action_spec
act_dim = low.shape[0]

# --- dims ---
try:
    gripper_dof = int(getattr(env.robots[0].gripper, "dof", 1))
except Exception:
    gripper_dof = 1
gripper_dof = max(0, min(gripper_dof, act_dim))
arm_dim = act_dim - gripper_dof
pos_dim = min(3, arm_dim)

# --- helpers ---
def clamp(a): return np.minimum(np.maximum(a, low), high)
def names(xs): return [x.decode("utf-8") if isinstance(x, bytes) else x for x in xs]

def ee_pos_from_obs(o):
    if "robot0_eef_pos" in o: return o["robot0_eef_pos"]
    n = names(env.sim.model.site_names)
    for i, s in enumerate(n):
        if "eef" in s.lower():
            return env.sim.data.site_xpos[i].copy()
    return np.zeros(3)

# def banana_pos_now():
#     m, d = env.sim.model, env.sim.data
#     for i, s in enumerate(names(m.site_names)):
#         if "banana" in s.lower(): return d.site_xpos[i].copy()
#     for i, b in enumerate(names(m.body_names)):
#         if "banana" in b.lower():
#             return (d.xpos[i].copy() if hasattr(d, "xpos") else d.body_xpos[i].copy())
#     raise RuntimeError("Add a site named like 'banana_center' or ensure body name contains 'banana'.")

def banana_pos_now():
    m, d = env.sim.model, env.sim.data

    # 1) 优先：site 恰好叫 "origin" 且挂在 "banana_align" 这个 body 上
    try:
        sid = m.site_name2id("origin")
        try:
            bid_align = m.body_name2id("banana_align")
        except Exception:
            bid_align = None
        if bid_align is None or m.site_bodyid[sid] == bid_align:
            return d.site_xpos[sid].copy()
    except Exception:
        pass

    # 2) 次优：找挂在 banana_align 上且名字里含 "origin" 的 site
    try:
        bid_align = m.body_name2id("banana_align")
        # 有的 mujoco 版本 m.site_names 需要 decode；沿用你原来的 names() 也可以
        names_list = [n.decode() if isinstance(n, (bytes, bytearray)) else n for n in m.site_names]
        for sid, bodyid in enumerate(m.site_bodyid):
            if bodyid == bid_align and "origin" in names_list[sid].lower():
                return d.site_xpos[sid].copy()
    except Exception:
        pass

    # 3) 回退：你的原有启发式
    try:
        names_list = [n.decode() if isinstance(n, (bytes, bytearray)) else n for n in m.site_names]
        for i, s in enumerate(names_list):
            if "banana" in s.lower():
                return d.site_xpos[i].copy()
    except Exception:
        pass

    try:
        body_names = [n.decode() if isinstance(n, (bytes, bytearray)) else n for n in m.body_names]
        for i, b in enumerate(body_names):
            if "banana" in b.lower():
                return (d.xpos[i].copy() if hasattr(d, "xpos") else d.body_xpos[i].copy())
    except Exception:
        pass

    raise RuntimeError(
        "找不到香蕉位置：请确保存在 site name='origin'（或名中含 banana 的 site），"
        "或者 body 名字里包含 banana。"
    )

def grip_metric(o):
    for k in o.keys():
        if "gripper" in k and "qpos" in k:
            v = np.array(o[k]).ravel()
            return float(v.mean())
    return None

# --- auto-calib gripper direction ---
def calib_grip_dir(steps=20):
    base = np.zeros_like(low)
    m0 = grip_metric(obs)

    a_high = base.copy()
    if gripper_dof > 0: a_high[-gripper_dof:] = 0.95 * high[-gripper_dof:]
    o = obs
    for _ in range(steps): o,_,_,_ = env.step(clamp(a_high))
    m_high = grip_metric(o)

    a_zero = base.copy()
    if gripper_dof > 0: a_zero[-gripper_dof:] = 0.0
    for _ in range(10): env.step(clamp(a_zero))

    a_low = base.copy()
    if gripper_dof > 0: a_low[-gripper_dof:] = 0.95 * low[-gripper_dof:]
    o2 = o
    for _ in range(steps): o2,_,_,_ = env.step(clamp(a_low))
    m_low = grip_metric(o2)

    if None in (m0, m_high, m_low) or gripper_dof == 0:
        return 0.95 * high[-gripper_dof:], 0.95 * low[-gripper_dof:]
    if m_high > m_low:
        return 0.95 * high[-gripper_dof:], 0.95 * low[-gripper_dof:]
    else:
        return 0.95 * low[-gripper_dof:], 0.95 * high[-gripper_dof:]

OPEN_VEC, CLOSE_VEC = calib_grip_dir()

def set_grip(a, open_hand: bool):
    if gripper_dof <= 0: return
    a[-gripper_dof:] = OPEN_VEC if open_hand else CLOSE_VEC

# --- motion tuning (slower & smoother) ---
K_POS = 8              # lower = slower approach
XY_STEP_CAP = 0.08     # m per step cap (x,y)
Z_STEP_CAP  = 0.04     # m per step cap (z)
SMOOTH = 0.1           # EMA on actions: a = SMOOTH*a_prev + (1-SMOOTH)*a_cmd
a_prev = np.zeros_like(low)

# collect banana / gripper geom ids for contact check
GEOM_NAMES = [n.decode("utf-8") if isinstance(n, bytes) else n for n in env.sim.model.geom_names]
BANANA_IDS = {i for i, n in enumerate(GEOM_NAMES) if "banana" in n.lower()}
GRIPPER_IDS = {i for i, n in enumerate(GEOM_NAMES) if ("gripper" in n.lower()) or ("finger" in n.lower())}

def touching_banana():
    d = env.sim.data
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in BANANA_IDS and g2 in GRIPPER_IDS) or (g2 in BANANA_IDS and g1 in GRIPPER_IDS):
            return True
    return False

def set_dpos(a, dpos):
    # cap per-step translation
    if pos_dim >= 1: a[0] = np.clip(K_POS * dpos[0], -XY_STEP_CAP, XY_STEP_CAP)
    if pos_dim >= 2: a[1] = np.clip(K_POS * dpos[1], -XY_STEP_CAP, XY_STEP_CAP)
    if pos_dim >= 3: a[2] = np.clip(K_POS * dpos[2], -Z_STEP_CAP,  Z_STEP_CAP)

# --- scripted pick ---
banana0 = banana_pos_now()
hover_h = 0.22
contact_z = max(0.07, banana0[2] + 0.01)
lift_h = 0.32
phase, steps, closed_steps = 0, 0, 0
success = False
banana_last = banana0.copy()

while True:
    steps += 1
    a_cmd = np.zeros_like(low)

    ee = ee_pos_from_obs(obs)
    try:
        banana = banana_pos_now(); banana_last = banana.copy()
    except Exception:
        banana = banana_last.copy()

    hover = np.array([banana[0], banana[1], max(banana[2] + hover_h, 0.22)])

    if phase == 0:
        # open + move above
        dpos = hover - ee
        set_dpos(a_cmd, dpos)
        set_grip(a_cmd, open_hand=True)
        if np.linalg.norm(dpos[:2]) < 0.02 and abs(dpos[2]) < 0.03:
            phase = 1

    elif phase == 1:
        # open + descend until first touch (or reach a floor)
        target_z = banana[2] - 0.02     # a bit below banana center
        dpos = np.array([0.0, 0.0, target_z - ee[2]])
        set_dpos(a_cmd, dpos)
        set_grip(a_cmd, open_hand=True)
        # trigger on touch (preferred) or close-to-target fallback
        if touching_banana() or abs(dpos[2]) < 0.002:
            phase = 2


    elif phase == 2:
        # close
        set_dpos(a_cmd, np.zeros(3))
        set_grip(a_cmd, open_hand=False)
        closed_steps += 1
        if closed_steps >= 500:
            phase = 3

    elif phase == 3:
        # lift
        dpos = np.array([0.0, 0.0, lift_h - ee[2]])
        set_dpos(a_cmd, dpos)
        set_grip(a_cmd, open_hand=False)
        if banana[2] - banana0[2] > 0.06:
            success = True
            break

    # smooth + clamp
    a = SMOOTH * a_prev + (1 - SMOOTH) * a_cmd
    a = clamp(a)
    a_prev = a.copy()

    obs, _, done, _ = env.step(a)
    env.render()
    if done or steps >= env.horizon:
        break

raise_h = banana_last[2] - banana0[2]
print(f"[result] success={success}, steps={steps}, raise={raise_h:.3f} m")
env.close()
