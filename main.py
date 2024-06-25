import numpy as np

from systems import *
from controller import *

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from collections import namedtuple

ControllerResult = namedtuple('Result', ['times', 'states', 'inputs', 'solver_time', 'computation_time'])

def eval(system, controller, x0, T_sim, dt_sim, N_sim_iter=10):
  xn = x0
  t0 = 0.
    
  states = [x0]
  inputs = []

  times = [0]
  computation_times_ms = []
  solve_times = []

  for j in range(int(T_sim / dt_sim)):
    # print("x0")
    # print(xn)
    # print(system.step(xn, np.zeros(system.input_dim), dt_sim, method='heun'))
    t = t0 + j * dt_sim

    start = time.process_time_ns()
    u = controller.compute(xn, t)
    end = time.process_time_ns()

    # finer simulation
    for i in range(N_sim_iter):
      xn = system.step(xn, u, dt_sim/N_sim_iter, method='heun')

    #xn += np.random.randn(4) * 0.0001

    states.append(xn)
    inputs.append(u)

    times.append(times[-1] + dt_sim)

    computation_times_ms.append((end - start) / 1e6)
    solve_times.append(controller.solve_time) 

  return ControllerResult(times, states, inputs, solve_times, computation_times_ms)

def double_cartpole_test():
  T_sim = 5

  sys = DoubleCartpole()
  x = np.zeros(6)
  x[1] = np.pi
  x[2] = 0
  u = np.zeros(1) - 0
  dt = 0.03

  Q = np.diag([1, 2, 2, 0.01, 0.01, 0.01])
  R = np.diag([.005])

  ref = np.zeros((6, 1))

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([2, 5, 2, 10, 10, 10]))
  input_scaling = 1 / (np.array([50]))
  
  # nmpc = NMPC(sys, 20, dt) # does not work
  # nmpc = NMPC(sys, 20, dt * 2) # does not work
  nmpc = NMPC(sys, 30, dt, quadratic_cost, ref) # does work
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 40 * dt, 20) # works
  # dts = get_linear_spacing(dt, 30 * dt, 20) # does not work
  # dts = get_linear_spacing(dt, 40 * dt, 15)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  # mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
  for res in [nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states, label=["x", "a1", "a2", "v", "a1_v", "a2_v"])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    for i in range(len(times)):
      if i % 1 == 0:
        rect = patches.Rectangle((states[i][0]-0.05, 0-0.05), width=0.1, height=0.1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        xpos = states[i][0]
        plt.plot([xpos, xpos + np.sin(states[i][1])], [0, np.cos(states[i][1])], color=(0, 0, i / len(times)))
        plt.plot([xpos + np.sin(states[i][1]), xpos + np.sin(states[i][1]) + np.sin(states[i][1] + states[i][2])], [np.cos(states[i][1]), np.cos(states[i][1]) + np.cos(states[i][1] + states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def cartpole_test():
  T_sim = 4
  sys = Cartpole()
  x = np.zeros(4)
  x[2] = 0
  u = np.zeros(1)
  dt = 0.05

  Q = np.diag([1, 0, 3, 0])
  ref = np.zeros((4, 1))
  ref[2, 0] = np.pi

  R = np.diag([.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    plt.figure()
    plt.plot(times, states, label=['x', 'v', 'theta', 'w'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    for i in range(len(times)):
      if i % 1 == 0:
        rect = patches.Rectangle((states[i][0]-0.05, 0-0.05), width=0.1, height=0.1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        plt.plot([states[i][0], states[i][0] + np.sin(states[i][2])], [0, -np.cos(states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_laplacian_dynamics():
  T_sim = 5

  sys = LaplacianDynamics()
  x = np.ones(3) * 2
  u = np.zeros(3)

  Q = np.diag([1, 1, 1])
  ref = np.ones((3, 1))
  R = np.diag([10, 10, 10])

  dt = 0.2

  quadratic_cost = QuadraticCost(Q, R, Q)

  nmpc = NMPC(sys, 10, dt*2, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states, label=['x', 'y', 'z'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_masspoint():
  T_sim = 3

  sys = Masspoint2D()
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([1, 0, 1, 0])
  ref = np.zeros((4, 1))
  R = np.diag([.1, .1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dt = 0.1
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  blocks = get_linear_blocking(20, 10)
  mb_mpc = MoveBlockingNMPC(sys, 20, dt, quadratic_cost, ref, blocks)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)
  mb_mpc_sol = eval(sys, mb_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol, mb_mpc_sol]:
  # for times, states, inputs, computation_times in [mb_mpc_sol]:
  # for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'y', 'v_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    plt.figure()
    plt.plot(res.solver_time)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def squircle(t):
  w = t * 0.8
  x = abs(np.cos(w)) ** (2/8) * np.sign(np.cos(w))
  y = abs(np.sin(w)) ** (2/8) * np.sign(np.sin(w))
  return np.array([x, 0, y, 0]).reshape((-1, 1))

def square(t):
  t = (float(t)) % 4
  side_length = 1  # Total side length of the square
  half_side = side_length / 2
  x = np.piecewise(t,
                [t < 1, (t >= 1) & (t < 2), (t >= 2) & (t < 3), t >= 3],
                [lambda t: half_side,
                lambda t: half_side - side_length * (t - 1),
                lambda t: -half_side,
                lambda t: -half_side + side_length * (t - 3)])
  y = np.piecewise(t,
                [t < 1, (t >= 1) & (t < 2), (t >= 2) & (t < 3), t >= 3],
                [lambda t: -half_side + side_length * t,
                lambda t: half_side,
                lambda t: half_side - side_length * (t - 2),
                lambda t: -half_side])
  
  return np.array([x, y])

def test_masspoint_ref_path():
  T_sim = 4

  sys = Masspoint2D()
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([10, 0.01, 10, 0.01])
  R = np.diag([.01, .01])

  ref = lambda t: np.array([square(t)[0], 0, square(t)[1], 0]).reshape(-1, 1)

  x = ref(0).flatten()

  quadratic_cost = QuadraticCost(Q, R, Q * 0.01)

  dt = 0.01
  nmpc = NMPC(sys, 10, dt*10, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 100 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][2] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    
    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'y', 'v_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref(times[i]))
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_jerk_masspoint():
  T_sim = 3
  dt = 0.05

  sys = JerkMasspoint2D()
  x = np.zeros(6)
  x[0] = 2
  x[1] = 0
  x[2] = 0
  x[3] = 5

  u = np.zeros(2)

  Q = np.diag([20, 0.01, 0.01, 20, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([.001, .001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([5, 5, 10, 5, 5, 10]))
  input_scaling = 1 / (np.array([50, 50]))
  
  # state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  # input_scaling = 1 / np.array([100, 100])

  nmpc = NMPC(sys, 10, dt*2, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_linear_spacing_v2(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    x = [states[i][0] for i in range(len(times))]
    y = [states[i][3] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)

    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'a_x', 'y', 'v_y', 'a_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_jerk_masspoint_ref_path():
  T_sim = 5
  dt = 0.01

  sys = JerkMasspoint2D()
  
  u = np.zeros(2)

  Q = np.diag([20, 0.01, 0.01, 20, 0.01, 0.01])
  R = np.diag([.0001, .0001])

  ref = lambda t: np.array([square(t)[0], 0, 0, square(t)[1], 0, 0]).reshape(-1, 1)
  x = ref(0).flatten()

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([5, 5, 10, 5, 5, 10]))
  input_scaling = 1 / (np.array([50, 50]))
  
  # state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  # input_scaling = 1 / np.array([100, 100])

  nmpc = NMPC(sys, 10, dt*10, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 100 * dt, 10)
  # dts = get_linear_spacing_v2(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][3] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)

    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'a_x', 'y', 'v_y', 'a_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref(times[i]))
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_racecar():
  T_sim = 1
  dt = 0.05

  sys = Racecar()
  x = np.zeros(6)
  x[0] = -5
  x[1] = -2
  x[2] = 0.0
  x[3] = 0.4

  u = np.zeros(2)

  # cost from liniger implementation
  Q = np.diag([0.1, 0.1, 1e-5, 0, 0, 0])
  ref = np.zeros((6, 1))
  # ref[3] = 0.3
  R = np.diag([0.01, 1])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 10, 10, 5]))
  input_scaling = 1 / (np.array([1, 1]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 40, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  # dts = get_linear_spacing(dt, 20 * dt, 10)
  dts = get_linear_spacing_v2(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    plt.figure()
    plt.plot(times, states, label=['x', 'y', 'omega', 'vx', 'vy', 'omega_dot'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['a', 'delta'])

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_racecar_ref_path():
  T_sim = 5
  dt = 0.05

  sys = Racecar()
  x = np.zeros(6)
  x[0] = -5
  x[1] = -2
  x[2] = 0.0
  x[3] = 0.4

  u = np.zeros(2)

  # cost from liniger implementation
  Q = np.diag([1, 1, 0.0000, 0.0000, 0.0000, 0.0000])
  # ref[3] = 0.3
  R = np.diag([0.01, 0.01])

  ref = lambda t: np.array([square(t)[0], square(t)[1], 0, 0, 0, 0]).reshape(-1, 1)

  x = ref(0).flatten()
  x = np.array([0.5, -0.5, np.pi/2, 2, 0, 0])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q * 0.01)

  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5]))
  input_scaling = 1 / (np.array([1, 0.35]))
  
  # state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5]))
  # input_scaling = 1 / (np.array([1, 0.35]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_linear_spacing_v2(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][1] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[1] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_ref, y_ref, '--', color='tab:orange')

    ax = plt.gca()

    triangle = np.array([[0.5, 0], [-0.5, 0], [0, 1]]) * 0.05

    for s in states[::5]:
      # Create a polygon patch
      polygon = patches.Polygon(triangle, closed=True, edgecolor='black', fill=True)

      # Add the polygon to the current axis
      ax.add_patch(polygon)

      # Create a transformation for rotating the polygon
      angle = (s[2] - np.pi/2) / np.pi * 180  # angle in degrees
      origin = (0.0, 0.0)  # rotation origin
      t = (transforms.Affine2D()
          .rotate_deg_around(origin[0], origin[1], angle)
          .translate(s[0], s[1]))
      # Apply the transformation to the polygon
      polygon.set_transform(t + ax.transData)

    plt.axis("equal")

    plt.figure()
    plt.plot(times, states, label=['x', 'y', 'omega', 'vx', 'vy', 'omega_dot'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['a', 'delta'])

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref(times[i]))
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()


def test_unicycle():
  T_sim = 1
  dt = 0.05

  sys = Unicycle()
  x = np.zeros(3)
  x[0] = -5
  x[1] = -2
  x[2] = 0.0

  u = np.zeros(2)

  # cost from liniger implementation
  Q = np.diag([1, 1, 1])
  ref = np.zeros((3, 1))
  # ref[3] = 0.3
  R = np.diag([0.01, 0.1])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([1, 1, 1]))
  input_scaling = 1 / (np.array([1, 1]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 15, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  # dts = get_linear_spacing(dt, 20 * dt, 10)
  dts = get_linear_spacing_v2(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][1] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    
    plt.figure()
    plt.plot(times, states, label=['x', 'y', 'omega'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['a', 'delta'])

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_unicycle_ref_path():
  T_sim = 5
  dt = 0.05

  sys = Unicycle()
  x = np.zeros(3)

  Q = np.diag([10, 10, 0])
  R = np.diag([0.001, 0.001])
  ref = lambda t: np.array([square(t)[0], square(t)[1], 0]).reshape(-1, 1)
  x = np.array([0.5, -0.5, np.pi/2])

  quadratic_cost = QuadraticCost(Q, R, 0.1 * Q)

  state_scaling = 1 / (np.array([1, 1, 1]))
  input_scaling = 1 / (np.array([5, 3]))

  # nmpc = NMPC(sys, 10, dt*2, quadratic_cost, ref)
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_power_spacing(dt, 20*dt, 10)

  # dts = [dt] + [19 * dt / 9 for _ in range(9)]
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][1] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[1] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_ref, y_ref, '--', color='tab:orange')

    ax = plt.gca()

    triangle = np.array([[0.5, 0], [-0.5, 0], [0, 1]]) * 0.05

    for s in states[::5]:
      # Create a polygon patch
      polygon = patches.Polygon(triangle, closed=True, edgecolor='black', fill=True)

      # Add the polygon to the current axis
      ax.add_patch(polygon)

      # Create a transformation for rotating the polygon
      angle = (s[2] - np.pi/2) / np.pi * 180  # angle in degrees
      origin = (0.0, 0.0)  # rotation origin
      t = (transforms.Affine2D()
          .rotate_deg_around(origin[0], origin[1], angle)
          .translate(s[0], s[1]))
      # Apply the transformation to the polygon
      polygon.set_transform(t + ax.transData)

    plt.axis("equal")

    plt.figure()
    plt.plot(times, states, label=['x', 'y', 'omega'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['a', 'delta'])

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref(times[i]))
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_quadcopter():
  T_sim = 3

  sys = Quadcopter2D()

  Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([.01, .01])

  x = np.zeros(6)
  x[0] = 3
  x[1] = 0
  x[2] = np.pi
  x[5] = -10

  u = np.zeros(2) - 0
  dt = 0.05

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / np.array([1., 1, 1, 10, 10, 10])
  input_scaling = 1 / np.array([2, 2])

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 40, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 40 * dt, 20)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states, label=["x", "y", "phi", "v_x", "v_y", "phi_dot"])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    l = 0.2
    for i in range(len(times)):
      if i % 1 == 0:
        xpos = states[i][0]
        ypos = states[i][1]
        plt.scatter([xpos], [ypos + 0.05*np.cos(states[i][2])], color=(0, 0, i / len(times)))
        plt.plot([xpos - l*np.cos(states[i][2]), xpos + l*np.cos(states[i][2])], [ypos - l*np.sin(states[i][2]), ypos + l*np.sin(states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_chain_of_masses():
  T_sim = 20

  N = 3
  sys = ChainOfMasses(N)
  x = np.zeros((N+1)*2+N*2)
  for i in range(N+1):
    x[i * 2] = (i+1) * 1

  # set v to zero for all
  x[(N+1) * 2: ] = 0

  u = np.zeros(2)

  # compute steady state
  dummy_control = lambda x: np.array([0, 0])
  dummy_sol = eval(sys, dummy_control, x, 0.1, 0.01)

  ref = dummy_sol[1][-1][:, None]

  print(ref)

  # perturb
  dummy_control = lambda x: np.array([1, 1])
  dummy_sol = eval(sys, dummy_control, ref[:, 0], 1, 0.01)
  
  ref = ref.at[(N+1)*2:].set(0)
  x = dummy_sol[1][-1]
  
  Q = np.diag([0]*N*2 + [25, 25] + [20] * N * 2)
  R = np.diag([.01, .01])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dt = 0.1
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states)
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    plt.figure()
    for i in range(len(states)):
      state = states[i]
      plt.plot([0] + [state[i*2] for i in range(N+1)], [0] + [state[1 + i*2] for i in range(N+1)], 'o-', color = (0, 0, i / len(states)))

    plt.axis('equal')

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_cstr():
  T_sim = 0.005*50

  sys = CSTR()
  # Set the initial state of mpc, simulator and estimator:
  C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
  C_b_0 = 0.5 # This is the controlled variable [mol/l]
  T_R_0 = 134.14 #[C]
  T_K_0 = 130.0 #[C]
  x = np.array([C_a_0, C_b_0, T_R_0, T_K_0])

  u = np.zeros(2)

  Q = np.diag([0, 10., 0.0, 0])
  ref = np.array([0, 0.6, 0, 0]).reshape(-1, 1)
  R = np.diag([.1, .001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dt = 0.005
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  state_scaling = (np.array([1, 1, 1/10., 1/10.]))
  input_scaling = (np.array([1, 1]))
  
  # state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  # input_scaling = 1 / np.array([1, 1])

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    plt.figure()
    plt.plot(times, states, label=['C_a', 'C_b', 'T_R', 'T_K'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['F', 'Q_dot'])

    plt.figure()
    plt.plot(computation_times)
    
    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def test_batch_reactor():
  T_sim = 100

  sys = BatchBioreactor()
  X_s_0 = 1.0 # Concentration biomass [mol/l]
  S_s_0 = 0.5 # Concentration substrate [mol/l]
  P_s_0 = 0.0 # Concentration product [mol/l]
  V_s_0 = 120.0 # Volume inside tank [m^3]
  x = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

  u = np.zeros(1)

  Q = np.diag([0, 0, 1., 0])
  ref = np.array([0, 0., 10, 0]).reshape(-1, 1)
  R = np.diag([1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / np.array([1., 0.5, 0.5, 100.])
  input_scaling = 1 / np.array([0.01])

  # state_scaling = 1 / np.array([1., 1, 1, 1.])
  # input_scaling = 1 / np.array([1])

  dt = 0.5
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    plt.figure()
    plt.plot(times, states, label=['X_S', 'S_s', 'P_s', 'V_s'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

  plt.show()

def main():
    pass

def test_linearization_batchreactor():
  sys = BatchBioreactor()
  X_s_0 = 1.0 # Concentration biomass [mol/l]
  S_s_0 = 0.5 # Concentration substrate [mol/l]
  P_s_0 = 0.0 # Concentration product [mol/l]
  V_s_0 = 120.0 # Volume inside tank [m^3]
  x = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

  u = np.zeros(1)

  Q = np.diag([0, 0, 1., 0])
  ref = np.array([0, 0., 10, 0]).reshape(-1, 1)
  R = np.diag([1])

  state_scaling = 1 / np.array([1., 0.5, 0.5, 100.])
  input_scaling = 1 / np.array([0.01])

  # state_scaling = 1 / np.array([1., 1, 1, 1.])
  # input_scaling = 1 / np.array([1])

  dt = 0.5

  quadratic_cost = QuadraticCost(Q, R, Q)

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  _ = nmpc.compute(x * 1., 0.)
  _ = nu_mpc.compute(x * 1., 0.)

  u_nmpc = nmpc.prev_u
  u_nu_mpc = nu_mpc.nmpc.prev_u

  # fwd simulate
  x_nmpc = [x * 1.]
  t_nmpc_gt = [0]
  t_nmpc = [dt * i for i in range(20+1)]
  for u in u_nmpc.T:
    # print(u)
    for _ in range(10):
      xn = sys.step(x_nmpc[-1], u* (1 / input_scaling), dt/10, 'euler')
      x_nmpc.append(xn)

      t_nmpc_gt.append(t_nmpc_gt[-1] + dt / 10)

  x_numpc = [x * 1.]
  t_numpc_gt = [0]
  t_numpc = [0]
  for delta in dts:
    t_numpc.append(t_numpc[-1] + delta)

  for u, delta in zip(u_nu_mpc.T, dts):
    print(u)
    for _ in range(50):
      xn = sys.step(x_numpc[-1], u * (1 / input_scaling), delta/50, 'euler')
      x_numpc.append(xn)

      t_numpc_gt.append(t_numpc_gt[-1] + delta / 50)

  plt.figure()
  plt.plot(t_nmpc_gt, np.array(x_nmpc))
  plt.plot(t_nmpc, np.array(nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(t_numpc_gt, np.array(x_numpc))
  plt.plot(t_numpc, np.array(nu_mpc.nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(dts)

  plt.show()


def test_linearization_cstr():
  sys = CSTR()
  # Set the initial state of mpc, simulator and estimator:
  C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
  C_b_0 = 0.5 # This is the controlled variable [mol/l]
  T_R_0 = 134.14 #[C]
  T_K_0 = 130.0 #[C]
  x = np.array([C_a_0, C_b_0, T_R_0, T_K_0])

  u = np.zeros(2)

  Q = np.diag([0, 10., 0.0, 0])
  ref = np.array([0, 0.6, 0, 0]).reshape(-1, 1)
  R = np.diag([.1, .001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dt = 0.005

  state_scaling = 1/(np.array([1, 1, 100., 100.]))
  input_scaling = 1/(np.array([2000, 100]))

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  _ = nmpc.compute(x * 1., 0)
  _ = nu_mpc.compute(x * 1., 0)

  u_nmpc = nmpc.prev_u
  u_nu_mpc = nu_mpc.nmpc.prev_u

  # fwd simulate
  x_nmpc = [x * 1.]
  t_nmpc_gt = [0]
  t_nmpc = [dt * i for i in range(20+1)]
  for u in u_nmpc.T:
    # print(u)
    for _ in range(10):
      xn = sys.step(x_nmpc[-1], u* (1 / input_scaling), dt/10, 'euler')
      x_nmpc.append(xn)

      t_nmpc_gt.append(t_nmpc_gt[-1] + dt / 10)

  x_numpc = [x * 1.]
  t_numpc_gt = [0]
  t_numpc = [0]
  for delta in dts:
    t_numpc.append(t_numpc[-1] + delta)

  for u, delta in zip(u_nu_mpc.T, dts):
    print(u)
    for _ in range(50):
      xn = sys.step(x_numpc[-1], u * (1 / input_scaling), delta/50, 'euler')
      x_numpc.append(xn)

      t_numpc_gt.append(t_numpc_gt[-1] + delta / 50)

  plt.figure()
  plt.plot(t_nmpc_gt, np.array(x_nmpc))
  plt.plot(t_nmpc, np.array(nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(t_numpc_gt, np.array(x_numpc))
  plt.plot(t_numpc, np.array(nu_mpc.nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(dts)

  plt.show()

def test_linearization_quadcopter():
  T_sim = 3

  sys = Quadcopter2D()

  Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([.01, .01])

  x = np.zeros(6)
  x[0] = 2
  x[1] = 2
  x[2] = np.pi
  x[5] = -5

  u = np.zeros(2) - 0
  dt = 0.05

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / np.array([1., 1, 1, 5, 5, 5])
  input_scaling = 1 / np.array([1, 1])

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 40, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 40 * dt, 20)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  _ = nmpc.compute(x * 1., 0)
  _ = nu_mpc.compute(x * 1., 0)

  u_nmpc = nmpc.prev_u * (1 / input_scaling)
  u_nu_mpc = nu_mpc.nmpc.prev_u * (1 / input_scaling)

  # fwd simulate
  x_nmpc = [x * 1.]
  t_nmpc_gt = [0]
  t_nmpc = [dt * i for i in range(40+1)]
  for u in u_nmpc.T:
    # print(u)
    for _ in range(10):
      xn = sys.step(x_nmpc[-1], u, dt/10, 'euler')
      x_nmpc.append(xn)

      t_nmpc_gt.append(t_nmpc_gt[-1] + dt / 10)

  x_numpc = [x * 1.]
  t_numpc_gt = [0]
  t_numpc = [0]
  for delta in dts:
    t_numpc.append(t_numpc[-1] + delta)

  for u, delta in zip(u_nu_mpc.T, dts):
    # print(u)
    for _ in range(50):
      xn = sys.step(x_numpc[-1], u, delta/50, 'euler')
      x_numpc.append(xn)

      t_numpc_gt.append(t_numpc_gt[-1] + delta / 50)

  plt.figure()
  plt.plot(t_nmpc_gt, np.array(x_nmpc))
  plt.plot(t_nmpc, np.array(nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(t_numpc_gt, np.array(x_numpc))
  plt.plot(t_numpc, np.array(nu_mpc.nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(dts)

  plt.show()

def test_linearization_jerk():
  dt = 0.1

  sys = JerkMasspoint2D()
  x = np.zeros(6)
  x[0] = 2
  x[1] = 0
  x[2] = 0
  x[3] = 5

  u = np.zeros(2)

  Q = np.diag([1, 0, 0, 1, 0, 0])
  ref = np.zeros((6, 1))
  R = np.diag([.0001, .0001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  # state_scaling = 1 / (np.array([1, 5, 20, 1, 5, 20]))
  # input_scaling = 1 / (np.array([10, 10]))
  
  state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  input_scaling = 1 / np.array([100, 100])

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 10, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 40 * dt, 20)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  _ = nmpc.compute(x * 1., 0)
  _ = nu_mpc.compute(x * 1., 0)

  u_nmpc = nmpc.prev_u
  u_nu_mpc = nu_mpc.nmpc.prev_u

  # fwd simulate
  x_nmpc = [x * 1.]
  t_nmpc_gt = [0]
  t_nmpc = [dt * i for i in range(40+1)]
  for u in u_nmpc.T:
    # print(u)
    for _ in range(10):
      xn = sys.step(x_nmpc[-1], u / input_scaling, dt/10, 'euler')
      x_nmpc.append(xn)

      t_nmpc_gt.append(t_nmpc_gt[-1] + dt / 10)

  x_numpc = [x * 1.]
  t_numpc_gt = [0]
  t_numpc = [0]
  for delta in dts:
    t_numpc.append(t_numpc[-1] + delta)

  for u, delta in zip(u_nu_mpc.T, dts):
    # print(u)
    for _ in range(50):
      xn = sys.step(x_numpc[-1], u / input_scaling, delta/50, 'euler')
      x_numpc.append(xn)

      t_numpc_gt.append(t_numpc_gt[-1] + delta / 50)

  plt.figure()
  plt.plot(t_nmpc_gt, np.array(x_nmpc))
  plt.plot(t_nmpc, np.array(nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(t_numpc_gt, np.array(x_numpc))
  plt.plot(t_numpc, np.array(nu_mpc.nmpc.prev_x).T * (1 / state_scaling), 'o-')

  plt.figure()
  plt.plot(dts)

  plt.show()

class Problem:
  def __init__(self, sys, Q, ref, R, x0, T):
    self.T = T
    self.sys = sys
    
    self.Q = Q
    self.ref = ref
    self.R = R

    self.x0 = x0

def make_cost_computation_curve():
  # run controllers with given
  #  - initial conditions
  #  - weight matrices
  # for various T/k combinations
  # ideally, we would have a 3d plot
  # instead we just make a bunch of slices

  if False:
    ks = [5, 7, 10, 12, 15, 20]

    T_sim = 3

    sys = Quadcopter2D()

    Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01])
    ref = np.zeros((6, 1))
    R = np.diag([.01, .01])

    x = np.zeros(6)
    x[0] = 3
    x[1] = 0
    x[2] = np.pi
    x[5] = -10

    u = np.zeros(2) - 0
    dt = 0.05

    state_scaling = 1 / np.array([1., 1, 1, 10, 10, 10])
    input_scaling = 1 / np.array([2, 2])
  elif False:
    T_sim = 3

    # ks = [5, 7, 10, 12, 15, 20]
    ks = [5, 7, 10]

    sys = Masspoint2D()
    x = np.zeros(4)
    x[2] = 5
    u = np.zeros(2)

    Q = np.diag([1, 0, 1, 0])
    ref = np.zeros((4, 1))
    R = np.diag([.1, .1])

    dt = 0.1

    state_scaling = 1 / np.array([1, 1, 1, 1])
    input_scaling = 1 / np.array([5, 5])
  else:
    ks = [5, 7, 10, 12, 15, 20]

    T_sim = 4
    sys = Cartpole()
    x = np.zeros(4)
    x[2] = 0
    u = np.zeros(1)
    dt = 0.05

    Q = np.diag([1, 0, 3, 0])
    ref = np.zeros((4, 1))
    ref[2, 0] = np.pi

    R = np.diag([.1])

    state_scaling = 1 / np.array([1, 1, 10, 10])
    input_scaling = 1 / np.array([50])

  quadratic_cost = QuadraticCost(Q, R, Q)

  for T in [1]:
    for k in ks:
      nmpc = NMPC(sys, k, T / k, quadratic_cost, ref)
      nmpc.state_scaling = state_scaling
      nmpc.input_scaling = input_scaling

      #dts = get_relu_spacing(dt, 30 * dt, 15)
      dts = get_linear_spacing(dt, T, k)
      nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

      nu_mpc.nmpc.state_scaling = state_scaling
      nu_mpc.nmpc.input_scaling = input_scaling

      mpc_sol = eval(sys, nmpc, x, T_sim, dt)
      nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

      for j, res in enumerate([mpc_sol, nu_mpc_sol]):
        times = res.times
        states = res.states
        inputs = res.inputs
        computation_times = res.computation_time

        cost = 0
        for i in range(len(states)-1):
          diff = (states[i][:, None] - ref)
          u = inputs[i][:, None]
          cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

        print(cost)
        print(sum(computation_times))

        if j==0:
          plt.scatter(sum(computation_times), cost, marker='o', color="tab:blue")
        else:
          plt.scatter(sum(computation_times), cost, marker='s', color="tab:orange")

  plt.xlabel("comp time")
  plt.ylabel("Cost")

  plt.show()

def test_dt_max():
  dts = get_linear_spacing_with_max_dt(1, 0.01, 0.1, 20)

  plt.plot(dts)
  plt.show()

if __name__ == "__main__":
  # test_dt_max()

  # test_quadcopter()
  # test_linearization_quadcopter()
  # test_linearization_cstr()
  # test_linearization_batchreactor()
  # test_linearization_jerk()

  # test_laplacian_dynamics()

  # test_racecar()
  # test_racecar_ref_path()
  
  # test_masspoint()
  # test_masspoint_ref_path()

  # test_unicycle()
  test_unicycle_ref_path()

  # test_jerk_masspoint()
  # test_jerk_masspoint_ref_path()

  # test_chain_of_masses()
  # test_cstr()
  # test_batch_reactor()

  # make_cost_computation_curve()

  # cartpole_test()
  # double_cartpole_test()
