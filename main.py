import numpy as np
import jax

from systems import *
from controller import *

from util import *
from controller_util import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib import cm

import matplotlib.animation as animation

def double_cartpole_test():
  T_sim = 4

  sys = DoubleCartpole()
  x = np.zeros(6)
  x[1] = np.pi
  x[2] = 0
  u = np.zeros(1) - 0
  dt = 0.025

  Q = np.diag([1, 5, 5, 0.01, 0.01, 0.01])
  R = np.diag([.001])

  ref = np.zeros((6, 1))

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([2, 5, 2, 10, 10, 10]))
  input_scaling = 1 / (np.array([50]))
  
  # nmpc = NMPC(sys, 20, dt) # does not work
  # nmpc = NMPC(sys, 20, dt * 2) # does not work
  # nmpc = NMPC(sys, 30, dt, quadratic_cost, ref) # does work
  nmpc = NMPC(sys, 10, 0.1, quadratic_cost, ref) # 
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  # dts = get_linear_spacing(dt, 40 * dt, 20) # works
  # dts = get_linear_spacing(dt, 30 * dt, 20) # does not work
  dts = get_linear_spacing(dt, 1, 40) # 10 steps: 355
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
    plt.plot(times, states, label=["x", "a1", "a2", "v", "a1_v", "a2_v"])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    for i in range(len(times)):
      if i % 1 == 0:
        xpos = states[i][0]

        rect = patches.Rectangle((xpos-0.05, 0-0.05), width=0.1, height=0.1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        a1 = states[i][1]
        a2 = states[i][2]

        l1 = 0.25
        l2 = 0.25
        
        x_pts = [xpos, xpos + l1 * np.sin(a1), xpos + l1 * np.sin(a1) + l2 * np.sin(a1 + a2)]
        y_pts = [0, l1 * np.cos(a1), l1*np.cos(a1) + l2*np.cos(a1 + a2)]
        plt.plot(x_pts, y_pts, color=(0, 0, i / len(times)))

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

  Q = np.diag([5, 0.1, 50, 0.1])
  ref = np.zeros((4, 1))
  ref[2, 0] = np.pi

  R = np.diag([.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  # dts = [dt] * 50
  dts = get_linear_spacing(dt, 25 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  mppi = MPPI(sys, dts, quadratic_cost, ref, var=np.array([5]), num_rollouts=10000)
  
  dts = get_linear_spacing(dt, 25 * dt, 25)
  # dts = [dt] * 10
  ilqr = PenaltyiLQR(sys, dts, quadratic_cost, ref)
  
  # mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)
  # mppi_sol = eval(sys, mppi, x, T_sim, dt)
  ilqr_sol = eval(sys, ilqr, x, T_sim, dt)

  # for res in [mpc_sol, nu_mpc_sol]:
  # for res in [mppi_sol]:
  # for res in [nu_mpc_sol, mppi_sol]:
  for res in [ilqr_sol, nu_mpc_sol]:
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

  sys = MasspointND(2)
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([1, 0, 1, 0])
  ref = np.zeros((4, 1))
  ref[0, 0] = 2
  R = np.diag([.1, .1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dt = 0.1
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  blocks = get_linear_blocking(20, 10)
  mb_mpc = MoveBlockingNMPC(sys, 20, dt, quadratic_cost, ref, blocks)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = [dt] * 10
  ilqr = PenaltyiLQR(sys, dts, quadratic_cost, ref)

  # mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  # nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)
  # mb_mpc_sol = eval(sys, mb_mpc, x, T_sim, dt)
  ilqr_sol = eval(sys, ilqr, x, T_sim, dt)

  # for res in [ilqr_sol, nu_mpc_sol]:
  for res in [ilqr_sol]:
  # for res in [mpc_sol, nu_mpc_sol, mb_mpc_sol]:
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

# def square(t, side_length=1):
#   t = (t * 1.0) % 4
#   half_side = side_length / 2
  
#   def constant(value):
#     return lambda t: jax.numpy.full_like(t, value)
  
#   x = jax.numpy.piecewise(t,
#               [t < 1, (t >= 1) & (t < 2), (t >= 2) & (t < 3), t >= 3],
#               [constant(half_side),
#                lambda t: constant(half_side)(t) - side_length * (t - 1),
#                constant(-half_side),
#                lambda t: constant(-half_side)(t) + side_length * (t - 3)])
  
#   y = jax.numpy.piecewise(t,
#               [t < 1, (t >= 1) & (t < 2), (t >= 2) & (t < 3), t >= 3],
#               [lambda t: constant(-half_side)(t) + side_length * t,
#                constant(half_side),
#                lambda t: constant(half_side)(t) - side_length * (t - 2),
#                constant(-half_side)])
  
#   return jax.numpy.asarray([x, y])

# plt.figure()
# figure_eight_knots = np.linspace(0, 2*np.pi, 500)
# figure_eight_pts = [figure_eight(t) for t in figure_eight_knots]
# inners, outers = figure_eight_bounds()

# plt.plot(*np.array(figure_eight_pts).T)
# plt.plot(*np.array(inners).T)
# plt.plot(*np.array(outers).T)

# plt.axis("equal")

# plt.show()

# plt.plot([square_track(i)[0] for i in np.linspace(0, 4, 100)], [square_track(i)[1] for i in np.linspace(0, 4, 100)], 'o', alpha=0.2)
# plt.show()

def test_masspoint_ref_path():
  T_sim = 7

  sys = MasspointND(2)
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([10, 0.01, 10, 0.01])
  R = np.diag([.01, .01])

  ref = lambda t: jax.numpy.asarray([squircle(t)[0], 0, squircle(t)[1], 0]).reshape(-1, 1)
  # ref = lambda t: jax.numpy.asarray([figure_eight(t)[0], 0, figure_eight(t)[1], 0]).reshape(-1, 1)
  # ref = lambda t: jax.numpy.asarray([square(t)[0], 0, square(t)[1], 0]).reshape(-1, 1)
  x = ref(0).flatten()

  quadratic_cost = QuadraticCost(Q, R, Q * 0.01)

  state_scaling = 1 / (np.array([1,1,1,1]))
  input_scaling = 1 / (np.array([5, 5]))
  
  # state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  # input_scaling = 1 / np.array([100, 100])

  dt = 0.05
  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = [dt*2] * 10
  # dts = get_linear_spacing(dt, 5 * dt, 5)
  # dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_power_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  pred_rnd = PredictiveRandomSampling(sys, dts, quadratic_cost, ref)

  # mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  # nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)
  rnd_mpc_sol = eval(sys, pred_rnd, x, T_sim, dt)

  for res in [rnd_mpc_sol]:
  # for res in [mpc_sol, nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][2] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[2] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_ref, y_ref, '--', color='tab:orange')
    
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
  T_sim = 4
  dt = 0.05

  sys = JerkMasspointND(2)
  x = np.zeros(6)
  x[0] = 2
  x[1] = 0
  x[2] = 0
  x[3] = 5

  u = np.zeros(2)

  Q = np.diag([10, 0.01, 0.01, 10, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([.001, .001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([5, 1, 3, 5, 1, 3]))
  input_scaling = 1 / (np.array([50, 50]))
  
  # state_scaling = 1 / np.array([1, 1, 1, 1, 1, 1])
  # input_scaling = 1 / np.array([100, 100])

  nmpc = NMPC(sys, 10, dt*2, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = [dt] * 20
  # dts = [dt] * 10
  # dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_linear_spacing_v2(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  pred_rnd = MPPI(sys, dts, quadratic_cost, ref)

  # mpc_sol = eval(sys, nmpc, x, T_sim, dt)
  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)
  rnd_mpc_sol = eval(sys, pred_rnd, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  # for res in [mpc_sol, nu_mpc_sol]:
  for res in [rnd_mpc_sol, nu_mpc_sol]:
  # for res in [rnd_mpc_sol]:
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
  dt = 0.05

  sys = JerkMasspointND(2)
  
  u = np.zeros(2)

  Q = np.diag([20, 0.01, 0.01, 20, 0.01, 0.01])
  R = np.diag([.000, .000])

  ref = lambda t: np.array([square(t)[0], 0, 0, square(t)[1], 0, 0]).reshape(-1, 1)
  x = ref(0).flatten()

  quadratic_cost = QuadraticCost(Q, R, Q * 0.05)

  # state_scaling = 1 / (np.array([5, 5, 10, 5, 5, 10]))
  # input_scaling = 1 / (np.array([10, 10]))
  
  state_scaling = 1 / np.array([1, 1, 3, 1, 1, 3])
  input_scaling = 1 / np.array([20, 20])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

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
  T_sim = 4
  dt = 0.05

  sys = Racecar()
  x = np.zeros(6)
  x[0] = -5
  x[1] = -2
  x[2] = 0.0
  x[3] = 0.4

  u = np.zeros(2)

  # cost from liniger implementation
  Q = np.diag([1, 1, 0.00001, 0.00001, 0.00001, 0.00001])
  # ref[3] = 0.3
  R = np.diag([0.01, 0.01])

  ref = lambda t: np.array([figure_eight(t*2)[0], figure_eight(t*2)[1], 0, 0, 0, 0]).reshape(-1, 1)

  x = ref(0).flatten()
  x[3] = 0.5
  # x = np.array([0.5, -0.5, np.pi/2, 0.5, 0, 0])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q * 0.01)

  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5]))
  input_scaling = 1 / (np.array([0.1, 0.35]))
  
  # state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5]))
  # input_scaling = 1 / (np.array([1, 0.35]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 40 * dt, 20)
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
  ref = lambda t: jnp.asarray([squircle(t*4)[0], squircle(t*4)[1], 0]).reshape(-1, 1)
  x = np.array([1, 0, np.pi/2])

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

def test_second_order_unicycle_ref_path():
  T_sim = 2
  dt = 0.05

  sys = Unicycle2ndOrder()
  x = np.zeros(5)

  Q = np.diag([100, 100, 0, 0.01, 0.01])
  R = np.diag([0.00001, 0.00001])
  ref = lambda t: jnp.asarray([squircle(t*4)[0], squircle(t*4)[1], 0, 0, 0]).reshape(-1, 1)
  x = np.array([1, 0, np.pi/2, 0, 0])

  quadratic_cost = QuadraticCost(Q, R, 0.1 * Q)

  state_scaling = 1 / (np.array([1, 1, 1, 1, 1]))
  input_scaling = 1 / (np.array([5, 3]))

  # nmpc = NMPC(sys, 10, dt*2, quadratic_cost, ref)
  nmpc = NMPC(sys, 20, dt*2, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 40 * dt, 20)
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
    plt.plot(times, states, label=['x', 'y', 'omega', 'v', 'delta'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['a', 'ddelta'])

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
  dt = 0.2

  quadratic_cost = QuadraticCost(Q, R, 0.1 * Q)

  state_scaling = 1 / np.array([1., 1, 1, 5, 5, 10])
  input_scaling = 1 / np.array([2, 2])

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 40, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  # dts = get_linear_spacing(dt, 40 * dt, 10)
  dts = get_power_spacing(dt, 40 * dt, 10)

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

def test_quadcopter_ref_path():
  T_sim = 3

  sys = Quadcopter2D()
# [[29.40315802]]
# [[29.27208035]]

  Q = np.diag([10, 10, 1, 0.01, 0.01, 0.01])
  ref = lambda t: np.array([square(t)[0], square(t)[1], 0, 0, 0, 0]).reshape(-1, 1)
  R = np.diag([.01, .01])

  x = np.zeros(6)
  x[0] = 3
  x[1] = 0
  x[2] = np.pi
  x[5] = -10

  u = np.zeros(2) - 0
  dt = 0.05

  quadratic_cost = QuadraticCost(Q, R, 0.1 * Q)

  state_scaling = 1 / np.array([1., 1, 1, 5, 5, 10])
  input_scaling = 1 / np.array([2, 2])

  # state_scaling = 1 / np.array([1., 1, 1, 1., 1., 1.])
  # input_scaling = 1 / np.array([1, 1])

  nmpc = NMPC(sys, 40, dt, quadratic_cost, ref)
  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  # dts = get_linear_spacing(dt, 40 * dt, 10)
  dts = get_power_spacing(dt, 40 * dt, 10)

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
      diff = (states[i][:, None] - ref(times[i]))
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

  sys = JerkMasspointND(2)
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

def make_T_curve():
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
  elif True:
    T_sim = 5
    dt = 0.05

    sys = JerkMasspointND(2)
    x = np.zeros(6)
    x[0] = 2
    x[1] = 0
    x[2] = 0
    x[3] = 5

    u = np.zeros(2)

    Q = np.diag([10, 0.01, 0.01, 10, 0.01, 0.01])
    ref = np.zeros((6, 1))
    R = np.diag([.001, .001])

    quadratic_cost = QuadraticCost(Q, R, Q)

    state_scaling = 1 / (np.array([5, 1, 3, 5, 1, 3]))
    input_scaling = 1 / (np.array([50, 50]))
  elif True:
    T_sim = 3

    sys = MasspointND(2)
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

  quadratic_cost = QuadraticCost(Q, R, Q * 0.1)

  for T in [0.2, 0.5, 0.7, 1, 2]:
    k = int(T / dt)
    nmpc = NMPC(sys, k, T / k, quadratic_cost, ref)
    nmpc.state_scaling = state_scaling
    nmpc.input_scaling = input_scaling

    #dts = get_relu_spacing(dt, 30 * dt, 15)
    dts = get_linear_spacing(dt, T, k)
    # nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

    # nu_mpc.nmpc.state_scaling = state_scaling
    # nu_mpc.nmpc.input_scaling = input_scaling

    mpc_sol = eval(sys, nmpc, x, T_sim, dt)
    # nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

    for j, res in enumerate([mpc_sol]):
    # for j, res in enumerate([mpc_sol, nu_mpc_sol]):
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
        plt.scatter(k, cost, marker='o', color="tab:blue")
      else:
        plt.scatter(k, cost, marker='s', color="tab:orange")

  plt.xlabel("comp time")
  plt.ylabel("Cost")

  plt.show()

def make_dt_curve():
  if True:
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
  elif True:
    T_sim = 5
    dt = 0.05

    sys = JerkMasspointND(2)
    x = np.zeros(6)
    x[0] = 2
    x[1] = 0
    x[2] = 0
    x[3] = 5

    u = np.zeros(2)

    Q = np.diag([10, 0.01, 0.01, 10, 0.01, 0.01])
    ref = np.zeros((6, 1))
    R = np.diag([.001, .001])

    quadratic_cost = QuadraticCost(Q, R, Q)

    state_scaling = 1 / (np.array([5, 1, 3, 5, 1, 3]))
    input_scaling = 1 / (np.array([50, 50]))
  elif False:
    T_sim = 3

    sys = MasspointND(2)
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

  quadratic_cost = QuadraticCost(Q, R, Q * 0.1)

  for dt in [0.02, 0.05, 0.1, 0.2, 0.5]:
    T = 1
    k = int(T / dt)
    nmpc = NMPC(sys, k, T / k, quadratic_cost, ref)
    nmpc.state_scaling = state_scaling
    nmpc.input_scaling = input_scaling

    # dts = get_relu_spacing(dt, 30 * dt, 15)
    # dts = get_linear_spacing(dt, T, k)
    # nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

    # nu_mpc.nmpc.state_scaling = state_scaling
    # nu_mpc.nmpc.input_scaling = input_scaling

    mpc_sol = eval(sys, nmpc, x, T_sim, dt, int(dt/0.02 * 10))
    # nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

    for j, res in enumerate([mpc_sol]):
    # for j, res in enumerate([mpc_sol, nu_mpc_sol]):
      times = res.times
      states = res.states
      inputs = res.inputs
      computation_times = res.computation_time

      plt.figure()
      plt.plot(times, states)
      plt.savefig(f"./img/states_{dt:.3f}.png", dpi=200)

      plt.figure()
      plt.plot(times[1:], inputs)
      plt.savefig(f"./img/inputs_{dt:.3f}.png", dpi=200)

      plt.figure('cost_dt_curve')
      state_cost = 0
      input_cost = 0
      for i in range(len(states)-1):
        diff = (states[i][:, None] - ref)
        u = inputs[i][:, None]
        state_cost += 0.02/10 * (diff.T @ Q @ diff)
        input_cost += 0.02/10 * (u.T @ R @ u)

      print(state_cost)
      print(input_cost)
      print(state_cost+input_cost)

      print(sum(computation_times))

      if j==0:
        plt.scatter(dt, state_cost, marker='>', color="tab:blue")
        plt.scatter(dt, input_cost, marker='<', color="tab:blue")
        plt.scatter(dt, state_cost + input_cost, marker='o', color="tab:blue")
      else:
        plt.scatter(dt, state_cost, marker='>', color="tab:orange")
        plt.scatter(dt, input_cost, marker='<', color="tab:orange")
        plt.scatter(dt, state_cost + input_cost, marker='s', color="tab:orange")

  plt.xlabel("dt")
  plt.ylabel("Cost")
  plt.savefig(f"./img/cost_dt_curve.png", dpi=300)


  plt.show()

def mpcc_debug():
  T_sim = 3

  sys = MasspointND(2)
  u = np.zeros(2)

  Q = np.diag([1, 0, 1, 0])
  R = np.diag([.1, .1])

  dt = 0.05

  state_scaling = 1 / np.array([1, 1, 1, 1])
  input_scaling = 1 / np.array([1, 1])

  ref = lambda t: np.array([square(t)[0], square(t)[1]]).reshape(-1, 1)

  mapping = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
  ])

  x = (mapping.T @ ref(0)).flatten()

  nmpcc = NMPCC(sys, [dt]*20, mapping, ref)

  u = nmpcc.compute(x, 0)

  states = nmpcc.prev_x
  input = nmpcc.prev_x

  plt.figure("progress")
  plt.plot(nmpcc.prev_p[0, :])
  plt.plot(nmpcc.prev_p[1, :])

  plt.figure()
  plt.axis("equal")
  x = states[0, :]
  y = states[2, :]
  plt.plot(x, y)

  plt.figure()
  plt.plot(input.T)

  plt.show()

  # t = np.linspace(0, 4, 1000)
  # pts = a.path(t)
  # x = [pts[i][0] for i in range(1000)]
  # y = [pts[i][1] for i in range(1000)]
  # print(pts)
  # plt.plot(x, y)
  # plt.show()

def mpcc_test():
  T_sim = 17

  sys = MasspointND(2)

  dt = 0.1
  # dt = 0.05

  state_scaling = 1 / np.array([1, 5, 1, 5])
  input_scaling = 1 / np.array([5, 5])

  # state_scaling = 1 / np.array([1, 1, 1, 1])
  # input_scaling = 1 / np.array([1, 1])

  ref = lambda t: np.array([square(t,3)[0], square(t,3)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([racetrack(t)[0], racetrack(t)[1]]).reshape(-1, 1)

  mapping = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
  ])

  x = (mapping.T @ ref(0)).flatten()

  # dts = [dt]*10
  # dts = [dt]*20
  # dts = get_linear_spacing(dt, 20 * dt, 10)
  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_power_spacing(dt, 10 * dt, 5)

  nmpcc = NMPCC(sys, dts, mapping, ref)
  nmpcc.state_scaling = state_scaling
  nmpcc.input_scaling = input_scaling

  nmpcc_sol = eval(sys, nmpcc, x, T_sim, dt, mpcc=True)

  for res in [nmpcc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][2] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[1] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_ref, y_ref, '--', color='tab:orange')
    plt.axis('equal')

    for s in states[::20]:
      x = s[0]
      y = s[2]

      plt.scatter(x, y, color='tab:blue')

    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'y', 'v_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    progress = res.progress

    finish = 0
    for i in range(len(progress)):
      if progress[i] / nmpcc.progress_scaling[0] > 4:
        finish = i
        break

    laptime = finish * dt
    print(laptime)

    # cost = 0
    # for i in range(len(states)-1):
    #   diff = (states[i][:, None] - ref(times[i]))
    #   u = inputs[i][:, None]
    #   cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    # print(cost)

  plt.show()

def mpcc_jerk_test():
  T_sim = 4

  sys = JerkMasspointND(2)

  dt = 0.05

  state_scaling = 1 / (np.array([5, 1, 3, 5, 1, 3]))
  input_scaling = 1 / (np.array([50, 50]))
  # state_scaling = 1 / (np.array([1, 1, 1, 1, 1, 1]))
  # input_scaling = 1 / (np.array([50, 50]))

  ref = lambda t: np.array([square(t)[0], square(t)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([racetrack(t)[0], racetrack(t)[1]]).reshape(-1, 1)

  mapping = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
  ])

  x = (mapping.T @ ref(0)).flatten()

  # dts = [dt]*30
  dts = get_linear_spacing(dt, 30 * dt, 15)

  nmpcc = NMPCC(sys, dts, mapping, ref)
  nmpcc.state_scaling = state_scaling
  nmpcc.input_scaling = input_scaling

  nmpcc_sol = eval(sys, nmpcc, x, T_sim, dt)

  for res in [nmpcc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][3] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[1] for i in range(len(times))]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_ref, y_ref, '--', color='tab:orange')
    plt.axis('equal')
    
    plt.figure()
    plt.plot(times, states)
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    # cost = 0
    # for i in range(len(states)-1):
    #   diff = (states[i][:, None] - ref(times[i]))
    #   u = inputs[i][:, None]
    #   cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    # print(cost)

  plt.show()

def mpcc_racecar_test():
  T_sim = 9
  dt = 0.05

  sys = Racecar()
  x = np.zeros(6)

  # ref = lambda t: np.array([figure_eight(t)[0], figure_eight(t)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([square(t)[0], square(t)[1]]).reshape(-1, 1)
  ref = lambda t: np.array([racetrack(t)[0], racetrack(t)[1]]).reshape(-1, 1)

  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 4, 2, 5]))
  input_scaling = 1 / (np.array([0.1, 0.35]))
  mapping = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
  ])

  x = (mapping.T @ ref(0)).flatten()
  # x[2] = np.pi/2
  # x[3] = 0.4
  x[2] = -np.pi/4
  x[3] = 0.

  # dts = [dt]*20
  # dts = [dt]*10
  dts = get_linear_spacing(dt, 20 * dt, 10)
  # dts = get_power_spacing(dt, 20 * dt, 10)

  nmpcc = NMPCC(sys, dts, mapping, ref)
  nmpcc.state_scaling = state_scaling
  nmpcc.input_scaling = input_scaling

  nmpcc.contouring_weight = .25
  nmpcc.cont_weight = .5
  nmpcc.progress_weight = 2
  
  nmpcc.diff_from_center = 0.15

  nmpcc.input_reg = 1e-5

  nmpcc_sol = eval(sys, nmpcc, x, T_sim, dt)

  for res in [nmpcc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][1] for i in range(len(times))]
    v = [states[i][3] for i in range(len(times))]

    x_ref = [ref(times[i])[0] for i in range(len(times))]
    y_ref = [ref(times[i])[1] for i in range(len(times))]

    x_inner = [racetrack(times[i], track_inner)[0] for i in range(len(times))]
    y_inner = [racetrack(times[i], track_inner)[1] for i in range(len(times))]
    
    x_outer = [racetrack(times[i], track_outer)[0] for i in range(len(times))]
    y_outer = [racetrack(times[i], track_outer)[1] for i in range(len(times))]

    plt.figure()
    plt.scatter(x, y, marker='.', c=cm.viridis(np.array(v)/max(v)), edgecolor='none')
    plt.plot(x_ref, y_ref, '--', color='tab:orange')
    
    plt.plot(x_inner, y_inner, '--', color='black')
    plt.plot(x_outer, y_outer, '--', color='black')
    plt.axis('equal')

    ax = plt.gca()

    triangle = np.array([[0.5, 0], [-0.5, 0], [0, 1]]) * 0.05

    for s in states[::20]:
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
    
    plt.figure()
    plt.plot(times, states, label=sys.state_names)
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

    # cost = 0
    # for i in range(len(states)-1):
    #   diff = (states[i][:, None] - ref(times[i]))
    #   u = inputs[i][:, None]
    #   cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    # print(cost)

  plt.show()

def mpcc_amzracecar_test(track='fig8'):
  if track == 'fig8':
    T_sim = 5
  else:
     T_sim = 9
  
  dt = 0.025

  sys = AMZRacecar()
  x = np.zeros(8)

  if track == 'fig8':
    ref = lambda t: np.array([figure_eight(t, 1.2)[0], figure_eight(t, 1.2)[1]]).reshape(-1, 1)
  else:
    ref = lambda t: np.array([racetrack(t)[0], racetrack(t)[1]]).reshape(-1, 1)
  
  # ref = lambda t: np.array([square_track(t)[0], square_track(t)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([squircle(t*4)[0], squircle(t*4)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([square(t)[0], square(t)[1]]).reshape(-1, 1)
  
  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5, 1, 0.35]))
  input_scaling = 1 / (np.array([5, 3]))
  mapping = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
  ])

  x = (mapping.T @ ref(0)).flatten()

  if track == 'fig8':
    x[2] = 0.5 * np.pi
  else:
    x[2] = -np.pi/4

  if track=='fig8':
    goal_progress = 2*np.pi
  else:
    goal_progress = 4

  # dts = [dt * 4] * 10 # does not work
  # dts = [dt]*40 # 7.9, 15
  # dts = [dt]*20 # 8.05, 15
  # dts = [1/15]*15 # 8.025, 15
  # dts = [dt]*30 # 7.9
  # dts = [dt] * 10 # does not work
  # dts = [dt * 2] * 20 # 8.15, 18
  # dts = [dt * 2] * 10 # 8.075, 47
  # dts = get_linear_spacing(dt, 40 * dt, 20) # 7.95, 13
  # dts = get_linear_spacing(dt, 40 * dt, 10) # does not work
  # dts = get_linear_spacing(dt, 30 * dt, 20) # 7.9
  dts = get_linear_spacing(dt, 40 * dt, 20) # 8.125, 25
  # dts = get_linear_spacing(dt, 40 * dt, 10) # 8.125, 25
  # plt.plot(dts)

  # dts = get_linear_spacing_with_max_dt(40 * dt, dt, 0.2, 10) # fails
  # dts = get_power_spacing(dt, 40 * dt, 20) # 8.05
  # dts = get_power_spacing(dt, 20 * dt, 10) # 8.075, 22

  # plt.plot(dts)
  # plt.show()

  nmpcc = NMPCC(sys, dts, mapping, ref)
  nmpcc.state_scaling = state_scaling
  nmpcc.input_scaling = input_scaling

  nmpcc.contouring_weight = 0.5
  nmpcc.cont_weight = 0.25
  nmpcc.progress_weight = 4

  nmpcc.diff_from_center = 0.15

  if track != 'fig8':
    nmpcc.track_constraint_linear_weight = 100
    nmpcc.track_constraint_quadratic_weight = 2000

  nmpcc.input_reg = 1e-3

  nmpcc_sol = eval(sys, nmpcc, x, T_sim, dt, mpcc=True)

  for res in [nmpcc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
    
    x = [states[i][0] for i in range(len(times))]
    y = [states[i][1] for i in range(len(times))]
    v = [states[i][3] for i in range(len(times))]

    x_ref = [ref(i)[0] for i in np.linspace(0, goal_progress, 500)]
    y_ref = [ref(i)[1] for i in np.linspace(0, goal_progress, 500)]

    if track == 'fig8':
      inners, outers = figure_eight_bounds(1.2)

      x_inner = np.array(inners)[:, 0]
      y_inner = np.array(inners)[:, 1]

      x_outer = np.array(outers)[:, 0]
      y_outer = np.array(outers)[:, 1]
    else:
      x_inner = [racetrack(i, track_inner)[0] for i in np.linspace(0, goal_progress, 500)]
      y_inner = [racetrack(i, track_inner)[1] for i in np.linspace(0, goal_progress, 500)]
      
      x_outer = [racetrack(i, track_outer)[0] for i in np.linspace(0, goal_progress, 500)]
      y_outer = [racetrack(i, track_outer)[1] for i in np.linspace(0, goal_progress, 500)]

    progress = res.progress

    finish_idx = 0
    for i in range(len(progress)):
      if progress[i] / nmpcc.progress_scaling[0] > goal_progress:
        finish_idx = i
        break

    laptime = finish_idx * dt
    print('laptime', laptime)

    violations = []
    violation = 0
    for i in range(len(progress)):
      scaled_progress = progress[i] / nmpcc.progress_scaling[0]
      delta = ref(scaled_progress) - np.array([x[i*10], y[i*10]]).reshape(-1, 1)
      violations.append(np.linalg.norm(delta))

      if np.linalg.norm(delta) > nmpcc.diff_from_center:
        violation += 1

    print(violation)

    plt.figure()
    plt.plot(violations)

    plt.figure()
    plt.scatter(x, y, marker='.', c=cm.viridis(np.array(v)/max(v)), edgecolor='none')
    plt.plot(x_ref, y_ref, '--', color='tab:orange')

    plt.axis('equal')

    # if track != 'fig8':
    plt.plot(x_inner, y_inner, '--', color='black')
    plt.plot(x_outer, y_outer, '--', color='black')

    ax = plt.gca()

    triangle = np.array([[0.5, 0], [-0.5, 0], [0, 1]]) * 0.05

    for s in states[::20]:
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
    
    plt.figure()
    plt.plot(times, states, label=sys.state_names)
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure('controller time')
    plt.plot(computation_times)

    plt.figure('solver time')
    plt.plot(res.solver_time)

    plt.style.use('default')
    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()
    
    ax.plot(x_ref, y_ref, '--', color='black', alpha=0.5)
    
    # if track != 'fig8':
    ax.plot(x_inner, y_inner, '-', color='black')
    ax.plot(x_outer, y_outer, '-', color='black')

    pred, = ax.plot([], [], '-')
    path, = ax.plot([], [], '--')

    rw = 0.1
    rh = 0.2
    rect = patches.Rectangle((-rw/2, -rh/2), rw, rh, fill=None, edgecolor='black', rotation_point='center')
    ax.add_patch(rect)

    scat = ax.scatter([0], [0])
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
      x_pred = [res.state_predictions[i][0, j] for j in range((res.state_predictions[i].shape[1]))]
      y_pred = [res.state_predictions[i][1, j] for j in range((res.state_predictions[i].shape[1]))]
      angle_pred = [res.state_predictions[i][2, j] / state_scaling[2] for j in range((res.state_predictions[i].shape[1]))]
      
      history_x = x[:i*10]
      history_y = y[:i*10]

      pred.set_data(x_pred, y_pred)
      path.set_data(history_x, history_y)

      ref_pos = ref(progress[i]/ nmpcc.progress_scaling[0])
      scat.set_offsets(np.array([ref_pos[0][0], ref_pos[1][0]]))

      # rect.set_xy((x_pred[0] - rw/2, y_pred[0]-rh/2))
      
      # Rotate the rectangle
      # rect.angle = angle_pred[0] / (2*np.pi) * 360 - 90.
      
      # Update the rectangle's transform to rotate around its center
      # trans = plt.matplotlib.transforms.Affine2D().rotate_deg_around(x_pred[0]-rw/2., y_pred[0] - rh/2., rect.angle)
      # rect.set_transform(trans + ax.transData)

      angle = (angle_pred[0] - np.pi/2) / np.pi * 180  # angle in degrees
      origin = (0.0, 0.0)  # rotation origin
      t = (transforms.Affine2D()
          .rotate_deg_around(origin[0], origin[1], angle)
          .translate(x_pred[0], y_pred[0]))
      # Apply the transformation to the polygon
      rect.set_transform(t + ax.transData)

      time_text.set_text(time_template % (i*dt))

      return pred, path, time_text
    
    ani = animation.FuncAnimation(
      fig, animate, len(computation_times), interval=dt*1000, blit=True)

    ani.save(f'amzracecar_animation_{track}.gif', writer='pillow', fps=30, savefig_kwargs={ "bbox_inches": "tight" })

    # cost = 0
    # for i in range(len(states)-1):
    #   diff = (states[i][:, None] - ref(times[i]))
    #   u = inputs[i][:, None]
    #   cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    # print(cost)

  plt.show()

if __name__ == "__main__":
  # motivation_horizon()
  # motivation_discretization()

  # make_cart_pole_animation()
  # make_quadcopter_animation()
  # make_double_cart_pole_animation()

  # make_masspoint_cost_comp_for_blog(num_runs=5)
  # make_quadcopter_cost_comp_for_blog(num_runs=5)
  # make_cartpole_cost_comp_for_blog(num_runs=5)
  # make_racecar_analysis(num_runs=3, track='fig8')
  # make_racecar_analysis(num_runs=1, track='race')

  # make_double_cartpole_cost_comp_for_blog()

  # test_quadcopter()
  # test_quadcopter_ref_path()
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
  # test_unicycle_ref_path()
  # test_second_order_unicycle_ref_path()

  # test_jerk_masspoint()
  # test_jerk_masspoint_ref_path()

  # test_chain_of_masses()
  # test_cstr()
  # test_batch_reactor()

  # mpcc_debug()
  # make_T_curve()
  # make_dt_curve()
  # make_cost_computation_curve()

  # cartpole_test()
  # double_cartpole_test()

  # mpcc_test()
  # mpcc_jerk_test()
  # mpcc_racecar_test()
  mpcc_amzracecar_test(track='race')
  # mpcc_amzracecar_test(track='fig8')

  plt.show()
