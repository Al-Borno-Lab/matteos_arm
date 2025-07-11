import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
import math

# Load the XML model
model_path = "myoarm_centerreachoutbi1.xml"  # Replace with your XML file path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"XML model file '{model_path}' not found.")

model = mj.MjModel.from_xml_path(model_path)
data = mj.MjData(model)

def get_joint_qposadr(model, joint_name):
    for i in range(model.njnt):
        if model.names[model.name_jntadr[i]:].split(b'\x00', 1)[0].decode() == joint_name:
            return model.jnt_qposadr[i]
    raise ValueError(f"Joint '{joint_name}' not found in the model.")
def get_site_id(model, site_name):
    """Find the site ID by name."""
    for i in range(model.nsite):
        if model.names[model.name_siteadr[i]:].split(b'\x00', 1)[0].decode() == site_name:
            return i
    raise ValueError(f"Site '{site_name}' not found in the model.")

# Get joint IDs and set initial positions
shoulder_qposadr = get_joint_qposadr(model, 'shoulder')
elbow_qposadr = get_joint_qposadr(model, 'elbow')

# Initial joint positions
initial_positions = {
    #'shoulder': -0.8905542335,
    #'elbow': 1.781108467,
    'shoulder': -0.8905542335,
    'elbow': 1.781108467,
}

# Set initial joint positions
data.qpos[shoulder_qposadr] = initial_positions['shoulder']
data.qpos[elbow_qposadr] = initial_positions['elbow']

# Define target position and parameters
# Define target position and radius
target_position = np.array([0.552548, 0.0])  # Target coordinates in 2D
target_radius = 0.025  # Half the diameter

# Get the site ID for the hand
handsite_id = get_site_id(model, 'handsite')

# Initialize score
score = 0


# Create a GLFW window
if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")

window = glfw.create_window(1920, 1080, "MuJoCo Control", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Could not create GLFW window")

glfw.make_context_current(window)

# Set up MuJoCo visualization
camera = mj.MjvCamera()
camera.type = mj.mjtCamera.mjCAMERA_FREE  # Set camera to free mode
camera.lookat = np.array([0.25, 0, 0])  # Point camera at the origin (or adjust to your desired focus)
camera.distance = 1.5  # Distance from the origin (adjust as needed)
camera.elevation = -90  # Set elevation to -90 degrees for a top-down view
camera.azimuth = 0  # Keep azimuth aligned with the global frame

opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=1000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

# Define controls
actuator_mapping = {
    glfw.KEY_LEFT: 0,  # Control actuator 0
    glfw.KEY_DOWN: 2,  # Control actuator 1
    glfw.KEY_O: 4,  # Control actuator 2
    glfw.KEY_RIGHT: 1,  # Control actuator 3
    glfw.KEY_UP: 3,  # Control actuator 4
    glfw.KEY_L: 5,  # Control actuator 5
    
    glfw.KEY_A: 0,  # Control actuator 0
    glfw.KEY_S: 2,  # Control actuator 1
    glfw.KEY_D: 1,  # Control actuator 3
    glfw.KEY_W: 3,  # Control actuator 4
}

# Initialize control signals
ctrl_signals = np.zeros(model.nu)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key in actuator_mapping:
            actuator_id = actuator_mapping[key]
            ctrl_signals[actuator_id] += 0.1  # Increase control signal
        elif key == glfw.KEY_R:
            ctrl_signals.fill(0)  # Reset control signals
    elif action == glfw.RELEASE:
        if key in actuator_mapping:
            actuator_id = actuator_mapping[key]
            ctrl_signals[actuator_id] = 0.0  # Stop actuator

glfw.set_key_callback(window, key_callback)

tic = time.perf_counter()

# Main simulation loop
while not glfw.window_should_close(window):
    mj.mj_step(model, data)

    # Update controls
    data.ctrl[:] = ctrl_signals

    # Get current hand position
    hand_pos = data.site_xpos[handsite_id]

    # Check if the hand reaches the target
    if np.linalg.norm(hand_pos[:2] - target_position) <= target_radius:
        toc = time.perf_counter()
        print(f"Done in {toc - tic:0.4f} seconds")
        if (toc - tic) < 0.4:
            tt = 25
        elif (toc - tic) < 0.6:
            tt = 10
        elif (toc - tic) < 1.0:
            tt = 5
        else:
            tt = 1
        score += math.ceil(tt*score/150+1) # 150
        print(f"Score: {score}")
        tic = time.perf_counter()

        # Reset the arm to the initial position
        data.qpos[shoulder_qposadr] = initial_positions['shoulder']
        data.qpos[elbow_qposadr] = initial_positions['elbow']
        data.qvel[:] = 0  # Reset velocities

    # Render the scene
    viewport = mj.MjrRect(0, 0, glfw.get_framebuffer_size(window)[0], glfw.get_framebuffer_size(window)[1])
    mj.mjv_updateScene(model, data, opt, None, camera, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)

    # Display score on screen
    score_text = f"Score: {score}"

    # Swap buffers and poll for events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
glfw.terminate()
