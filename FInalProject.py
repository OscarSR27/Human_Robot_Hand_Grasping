# %%
import time
import mujoco
import mujoco.viewer
import numpy as np

# %%

# Path to the xml file
xml_path = "Scene/wonik_allegro/scene_right.xml"

# Load model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)

renderer = mujoco.Renderer(model)

# palm is the parent frame of all other bodies of the hand
hand_initial_pos = data.body('palm').xpos

target_initial_pos = data.body('cylinder_object').xpos

# PID controller parameters
kp = 0.2  # Proportional gain
ki = 0.001  # Integral gain
kd = 0.01  # Derivative gain
integral_error = np.zeros(3)  # Integral error initialization
prev_error = np.zeros(3)  # Previous error initialization

# Function that calculates the desired joint positions to move the hand towards the target.
def compute_control_signals(error):
    global integral_error, prev_error

    # Proportional term
    p = kp * error

    # Integral term
    integral_error += error
    i = ki * integral_error

    # Derivative term
    derivative_error = error - prev_error
    d = kd * derivative_error

    # Calculate the control signals (desired joint positions)
    control_signals = p + i + d

    # Update previous error
    prev_error = error

    return control_signals

mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 1000:
        step_start = time.time()
        # palm is the parent frame of all other bodies of the hand
        hand_initial_pos = data.body('palm').xpos

        target_initial_pos = data.body('cylinder_object').xpos

        # Calculate the distance to the target
        distance_to_target =  target_initial_pos - hand_initial_pos + [0,0,0.08]

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        #data.ctrl[:]= data.ctrl[:] + [0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001]
        data.ctrl[0:3] = compute_control_signals(distance_to_target)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        
        # Update the scene in the renderer
        renderer.update_scene(data)
        
        with viewer.lock():
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# %%
# Path to the xml file

'''
xml_path = "Scene/wonik_allegro/scene_right.xml"

# Load model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)

# palm is the parent frame of all other bodies of the hand
hand_initial_pos = data.body('palm').xpos

target_initial_pos = data.body('cylinder_object').xpos

# Calculate the distance to the target
distance_to_target = hand_initial_pos - target_initial_pos

renderer = mujoco.Renderer(model)

# Function that calculates the desired joint positions to move the hand towards the target.
def compute_control_signals(distance):
    # Define the proportional gain
    kp = 0.1

    # Calculate the control signals (desired joint positions)
    control_signals = kp * distance

    return control_signals

mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 1000:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        #data.ctrl[:]= data.ctrl[:] + [0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001, 0.002, 0.001,0.001]
        data.body('palm').xpos = data.body('palm').xpos - [0.1, 0.1, 0.1]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        
        # Update the scene in the renderer
        renderer.update_scene(data)
        
        with viewer.lock():
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
'''

# %%
'''
#Path to the xml file
xml_path = "Scene/wonik_allegro/scene_right.xml"

#Load model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 5:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
'''
