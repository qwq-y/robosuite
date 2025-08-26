import trimesh, numpy as np
m = trimesh.load("/home/qwqy/Documents/RL-GSBridge/robosuite/robosuite/models/assets/objects/banana/banana_collision.obj", force='mesh')

mins, maxs = m.bounds  # shape (3,)
x0, y0, z0 = mins
x1, y1, z1 = maxs

rad = max(abs(x0), abs(x1), abs(y0), abs(y1))
print("bottom_site:", [0, 0, float(z0)])
print("top_site:",    [0, 0, float(z1)])
print("horizontal_radius_site:", [float(rad), 0, 0])
height = z1 - z0
print("Estimated height (m):", float(height))
