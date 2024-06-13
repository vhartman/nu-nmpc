import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacobian

from systems import *

import time

import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Controller:
  def __init__(self, system):
    self.system = system

  def compute(self, x):
    raise NotImplementedError("Implement me pls")

class NMPC(Controller):
  def __init__(self, system, N, dt):
    self.sys = system
    self.N = N
    self.dt = dt

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    self.prev_x = []
    self.prev_u = []

    self.dts = [self.dt] * self.N
    #self.dts = [self.dt + 0.005*(i+1) for i in range(self.N)]

    # solver params
    self.sqp_iters = 2

  def compute(self, state, Q, x_ref, R):
    x = cp.Variable((self.state_dim, self.N+1))
    u = cp.Variable((self.input_dim, self.N))

    if len(self.prev_x) == 0:
      for i in range(self.N+1):
        self.prev_x.append(state)

        if i < self.N:
          self.prev_u.append(np.zeros(self.input_dim))

      self.prev_x = np.array(self.prev_x).T
      self.prev_u = np.array(self.prev_u).T

    I = np.eye(self.sys.state_dim)

    # dynamics constraints
    for _ in range(self.sqp_iters):
      constraints = []
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]

        # this should be shifted by one step
        prev_state = self.prev_x[:, i].flatten()
        prev_input = self.prev_u[:, i].flatten()

        A, B, K = self.sys.linearization(prev_state, prev_input, dt)

        constraints.append(
            x[:, next_idx][:, None] == (I + A) @ x[:, idx][:, None] + B @ u[:, idx][:, None] - K
        )

      constraints.append(x[:, 0] == state)

      # state constraints
      # input constraints
      for i in range(self.N + 1):
        if i < self.N:
          constraints.append(u[:, i] >= self.sys.input_limits[:, 0])
          constraints.append(u[:, i] <= self.sys.input_limits[:, 1])

        constraints.append(x[:, i] >= self.sys.state_limits[:, 0])
        constraints.append(x[:, i] <= self.sys.state_limits[:, 1])

      cost = 0
      for i in range(self.N):
        dt = self.dts[i]
        cost += dt * cp.sum_squares(Q @ (x[:, i+1] [:, None] - x_ref))
        cost += dt * cp.sum_squares(R @ u[:,i][:, None])

      #cost = cp.sum_squares(10*(x[2, self.N] - target_angle))
      #cost += cp.sum_squares(10*(x[1, self.N] - target_angle))
      #cost += cp.sum_squares(1 * (x[0, self.N]))

      objective = cp.Minimize(cost)

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      # The optimal objective value is returned by `prob.solve()`.
      result = prob.solve(solver='OSQP', eps_abs=1e-5, eps_rel=1e-3, max_iter=10000)
      #result = prob.solve(solver='OSQP', verbose=True,eps_abs=1e-7, eps_rel=1e-5, max_iter=10000)
      #result = prob.solve(solver='ECOS', verbose=True)

      print(x.value)
      print(u.value)

      self.prev_x = x.value
      self.prev_u = u.value

    if prob.status not in ["infeasible", "unbounded"]:
        return u.value[:, 0]

    return np.zeros(self.sys.input_dim)
  
class NU_NMPC(Controller):
  def __init__(self, system, dts):
    self.nmpc = NMPC(system, len(dts), 0)

    self.nmpc.dts = dts

  def compute(self, state, Q, x_ref, R):
    return self.nmpc.compute(state, Q, x_ref, R)

def eval(system, controller, x0, T_sim, dt_sim):
  states = []
  inputs = []

  times = [0]
  computation_times_ms = []

  xn = x0

  N_sim_iter = 10
  for _ in range(int(T_sim / dt_sim)):
    start = time.process_time_ns()
    u = controller(xn)
    end = time.process_time_ns()

    # finer simulation
    for i in range(N_sim_iter):
      xn = system.step(xn, u, dt_sim/N_sim_iter)

    #xn += np.random.randn(4) * 0.0001

    states.append(xn)
    inputs.append(u)

    times.append(times[-1] + dt_sim)

    computation_times_ms.append((end - start) / 1e6)

  return times, states, inputs, computation_times_ms

#def get_relu_spacing(dt0, dt_max, T, steps):

def get_linear_spacing(dt0, T, steps):
  alpha = 2 *(T - steps * dt0) / (steps * (steps-1))
  return [dt0 + i * alpha for i in range(steps)]

def double_cartpole_test():
  T_sim = 5

  sys = DoubleCartpole()
  x = np.zeros(6)
  x[1] = np.pi
  x[2] = 0
  u = np.zeros(1) - 0
  dt = 0.03

  Q = np.diag([1, 2, 2, 0.1, 0.1, 0.1])
  ref = np.zeros((6, 1))

  R = np.diag([.1])

  nmpc = NMPC(sys, 40, dt)

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 40 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=["x", "a1", "a2", "v", "a1_v", "a2_v"])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    for i in range(len(times[1:])):
      if i % 1 == 0:
        rect = patches.Rectangle((states[i][0]-0.05, 0-0.05), width=0.1, height=0.1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        xpos = states[i][0]
        plt.plot([xpos, xpos + np.sin(states[i][1])], [0, np.cos(states[i][1])], color=(0, 0, i / len(times)))
        plt.plot([xpos + np.sin(states[i][1]), xpos + np.sin(states[i][1]) + np.sin(states[i][1] + states[i][2])], [np.cos(states[i][1]), np.cos(states[i][1]) + np.cos(states[i][1] + states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

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

  nmpc = NMPC(sys, 20, dt)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=['x', 'v', 'theta', 'w'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    for i in range(len(times[1:])):
      if i % 1 == 0:
        rect = patches.Rectangle((states[i][0]-0.05, 0-0.05), width=0.1, height=0.1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        plt.plot([states[i][0], states[i][0] + np.sin(states[i][2])], [0, -np.cos(states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)):
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
  nmpc = NMPC(sys, 10, dt)

  dts = get_linear_spacing(dt, 40 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=['x', 'y', 'z'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

  plt.show()

def test_masspoint():
  T_sim = 5

  sys = Masspoint2D()
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([1, 0, 1, 0])
  ref = np.zeros((4, 1))
  R = np.diag([.1, .1])

  dt = 0.05
  nmpc = NMPC(sys, 10, dt*2)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=['x', 'v_x', 'y', 'v_y'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    plt.plot(computation_times)

  plt.show()

def test_quadcopter():
  T_sim = 3

  sys = Quadcopter2D()

  Q = np.diag([1, 1, 1, 0., 0., 0.])
  ref = np.zeros((6, 1))
  R = np.diag([.01, .01])

  x = np.zeros(6)
  x[0] = 2
  x[1] = 2
  x[2] = np.pi
  x[5] = -5

  u = np.zeros(2) - 0
  dt = 0.05

  nmpc = NMPC(sys, 20, dt)

  #dts = get_relu_spacing(dt, 30 * dt, 15)
  dts = get_linear_spacing(dt, 20 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=["x", "y", "phi", "v_x", "v_y", "phi_dot"])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs)

    plt.figure()
    l = 0.2
    for i in range(len(times[1:])):
      if i % 1 == 0:
        xpos = states[i][0]
        ypos = states[i][1]
        plt.scatter([xpos], [ypos + 0.05*np.cos(states[i][2])], color=(0, 0, i / len(times)))
        plt.plot([xpos - l*np.cos(states[i][2]), xpos + l*np.cos(states[i][2])], [ypos - l*np.sin(states[i][2]), ypos + l*np.sin(states[i][2])], color=(0, 0, i / len(times)))

    plt.axis('equal')

    plt.figure()
    plt.plot(computation_times)

    cost = 0
    for i in range(len(states)):
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

  dt = 0.1
  nmpc = NMPC(sys, 20, dt)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states)
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
    for i in range(len(states)):
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

  dt = 0.001
  nmpc = NMPC(sys, 20, dt)

  dts = get_linear_spacing(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts)

  nmpc_control = lambda x: nmpc.compute(x, Q, ref, R)
  numpc_control = lambda x: nu_mpc.compute(x, Q, ref, R)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times[1:], states, label=['C_a', 'C_b', 'T_R', 'T_K'])
    plt.legend()

    plt.figure()
    plt.plot(times[1:], inputs, label=['F', 'Q_dot'])

    plt.figure()
    plt.plot(computation_times)

  plt.show()

def main():
    pass

if __name__ == "__main__":
  #test_quadcopter()
  #test_laplacian_dynamics()
  #test_masspoint()
  # test_chain_of_masses()
  test_cstr()

  # cartpole_test()
  #double_cartpole_test()
