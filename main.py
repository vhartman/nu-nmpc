import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacobian

from scipy.interpolate import interp1d

from systems import *

import time

import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

class Controller:
  def __init__(self, system):
    self.system = system

  def compute(self, x):
    raise NotImplementedError("Implement me pls")

class QuadraticCost:
  def __init__(self, Q, R, QN, q = None, qn=None, r=None):
    self.Q = Q
    self.QN = QN
    self.q = q # linear part, not used atm
    self.qn = qn # linear part, not used atm

    self.R = R
    self.w = r # linear part, not used atm

class NMPC(Controller):
  def __init__(self, system, N, dt, quadratic_cost, reference):
    self.sys = system
    self.N = N
    self.dt = dt

    self.cost = quadratic_cost

    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    self.prev_x = []
    self.prev_u = []

    self.dts = [self.dt] * self.N
    #self.dts = [self.dt + 0.005*(i+1) for i in range(self.N)]

    self.input_scaling = np.array([1] * self.sys.input_dim)
    self.state_scaling = np.array([1] * self.sys.state_dim)

    # solver params
    self.sqp_iters = 4
    self.sqp_mixing = 0.8

    self.first_run = True

  def compute_initial_state_guess(self, x):
    if self.first_run:
      for i in range(self.N+1):
        self.prev_x.append(x)

      self.prev_x = np.array(self.prev_x).T
    else:
      # shift solution by one
      # self.prev_x = np.hstack((self.prev_x[:, 1:], self.prev_x[:, -1][:, None]))
      # self.prev_x[:, 0] = x

      times = np.array([0] + list(np.cumsum(self.dts)))

      f = interp1d(times, self.prev_x, axis=1, bounds_error=False, fill_value=self.prev_x[:, -1])
      new_x = f(times + self.dts[0])
      
      self.prev_x = new_x
      self.prev_x[:, 0] = x

  def compute_initial_control_guess(self):
    if self.first_run:
      for i in range(self.N):
        self.prev_u.append(np.zeros(self.input_dim))

      self.prev_u = np.array(self.prev_u).T
    else:
      # shift solution by one
      self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

  def compute(self, state, t):
    x = cp.Variable((self.state_dim, self.N+1))
    u = cp.Variable((self.input_dim, self.N))

    s = cp.Variable((self.state_dim, (self.N+1)*2))

    I = np.eye(self.sys.state_dim)

    S = np.diag(self.state_scaling)
    Sinv = np.diag(1 / self.state_scaling)

    W = np.diag(self.input_scaling)
    Winv = np.diag(1 / self.input_scaling)

    self.compute_initial_state_guess((S @ state[:, None]).flatten())
    self.compute_initial_control_guess()

    if not self.first_run:
      print(self.prev_x[:, 0])
    
    print("state")
    print(self.prev_x)

    # scaling:
    # xs = S @ x
    # us = W @ u
    # S @ xn = s @ (I + A) @ x + s @ B @ u + s @ K
    # xns = S @ (I + A) @ Sinv @ S @ x + S @ B @ Winv @ W @ s + S @ K
    # xns = S @ (I + A) @ Sinv @ xs + S @ B @ Winv @ us + S @ K
    # l <= Cx <= u
    # l*s <+ Cxs <= u*s
    # x^T Q x + u^T R u
    # Sinv @ S @ x)^T Q @ Sinv @ S @ x
    # (Sinv @ xs)^T @ Q @ Sinv @ xs
    # xs^T @ Sinv^T @ Q @ Sinv @ xs

    # A' = S @ (I + A) @ Sinv
    # B' = S @ B @ Winv
    # K' = S @ K

    # Q' = Sinv^T @ Q @ Sinv
    # R' = Winv^T @ R # Winv

    # dynamics constraints
    for k in range(self.sqp_iters):
      constraints = []
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]
        print(i, dt)

        # this should be shifted by one step
        prev_state = (Sinv @ self.prev_x[:, i][:, None]).flatten()
        prev_input = (Winv @ self.prev_u[:, i][:, None]).flatten()

        # print("state")
        # print(prev_state)
        # next_state = self.sys.step(prev_state, prev_input, dt, 'euler')
        # print(next_state)
        # next_state = self.sys.step(next_state, prev_input, dt, 'euler')
        # print(next_state)

        A, B, K = self.sys.linearization(prev_state, prev_input, dt)

        constraints.append(
            x[:, next_idx][:, None] == S @ (I + A) @ Sinv @ x[:, idx][:, None] + S @ B @ Winv @ u[:, idx][:, None] - S @ K
        )

        # print("A")
        # print(A)
        # print("scaled a")
        # print(S @ (I + A) @ Sinv )
        # print(S @ B @ Winv)
        # print("K")
        # print(S @ K)
        # print((S @ K))

        # print(self.sys.f(prev_state, prev_input))

      constraints.append(x[:, 0][:, None] == S @ state[:, None])

      # input constraints
      for i in range(self.N):
          constraints.append(u[:, i][:, None] >= W @ self.sys.input_limits[:, 0][:, None])
          constraints.append(u[:, i][:, None] <= W @ self.sys.input_limits[:, 1][:, None])

      # state constraints
      if not self.first_run:
        for i in range(self.N + 1):
          constraints.append(x[:, i][:, None] >= S @ self.sys.state_limits[:, 0][:, None] - s[:, i*2][:, None])
          constraints.append(x[:, i][:, None] <= S @ self.sys.state_limits[:, 1][:, None] + s[:, i*2+1][:, None])

      # slack variables
      for i in range((self.N + 1)*2):
        constraints.append(s[:, i] >= 0)

      # print("lims")
      # print(S @ self.sys.state_limits[:, 0][:, None])
      # print(S @ self.sys.state_limits[:, 1][:, None])

      # trust region constraints
      # if k > 0:
      #   for i in range(self.N + 1):
      #     if i < self.N:
      #       constraints.append(cp.norm(u[:, i] - self.prev_u[:, i], "inf") <= 2)

      #     constraints.append(cp.norm(x[:, i] - self.prev_x[:, i], "inf") <= 2)

      Qs = Sinv @ self.cost.Q @ Sinv
      QNs = Sinv @ self.cost.QN @ Sinv

      Rs = Winv @ self.cost.R @ Winv

      # print('costs')
      # print(Qs)
      # print(Rs)

      cost = 0
      curr_time = t
      for i in range(self.N): 
        dt = self.dts[i]
        x_ref = self.ref(curr_time)
        cost += dt * cp.quad_form((x[:, i] [:, None] - S @ x_ref), Qs)
        cost += dt * cp.quad_form(u[:,i][:, None], Rs)

        curr_time += dt

      # terminal cost
      cost += cp.quad_form((x[:, -1] [:, None] - S @ x_ref), QNs)

      # slack variables
      # for i in range((self.N + 1)*2):
      cost += 50 * cp.sum(s)

      cost = cost / 100

      #cost = cp.sum_squares(10*(x[2, self.N] - target_angle))
      #cost += cp.sum_squares(10*(x[1, self.N] - target_angle))
      #cost += cp.sum_squares(1 * (x[0, self.N]))

      objective = cp.Minimize(cost)

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      # The optimal objective value is returned by `prob.solve()`.
      result = prob.solve(solver='OSQP', eps_abs=1e-4, eps_rel=1e-6, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(solver='OSQP', verbose=True,eps_abs=1e-7, eps_rel=1e-5, max_iter=10000)
      # result = prob.solve(solver='ECOS', verbose=True, max_iters=1000, feastol=1e-5, reltol=1e-4, abstol_inacc=1e-5, reltol_inacc=1e-5, feastol_inacc=1e-5)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-8)
      # result = prob.solve(solver='PIQP', verbose=True)

      # options_cvxopt = {
      #     "max_iters": 5000,
      #     "verbose": True,
      #     "abstol": 1e-21,
      #     "reltol": 1e-11,
      #     "refinement": 2, # higher number seemed to make things worse
      #     "kktsolver": "robust"
      # }
      # result = prob.solve(solver='CVXOPT', **options_cvxopt)

      print("Sols")
      print(x.value)
      print(u.value)

      # update solutions for linearization
      # TODO: should really by a line search
      if self.first_run:
        self.prev_x = x.value
        self.prev_u = u.value
      else:
        self.prev_x = x.value * self.sqp_mixing + self.prev_x * (1 - self.sqp_mixing)
        self.prev_u = u.value * self.sqp_mixing + self.prev_u * (1 - self.sqp_mixing)

      self.first_run = False

    if prob.status not in ["infeasible", "unbounded"]:
        print("U")
        print(u.value[:, 0])
        print( Winv @ u.value[:, 0])
        return Winv @ u.value[:, 0]
        # return u.value[:, 0]

    return np.zeros(self.sys.input_dim)
  
class NU_NMPC(Controller):
  def __init__(self, system, dts, quadratic_cost, reference):
    self.nmpc = NMPC(system, len(dts), 0,quadratic_cost, reference)

    self.nmpc.dts = dts

  def compute(self, state, t):
    return self.nmpc.compute(state, t)

def constant_cost(state, i, Q, ref, R):
  return Q, ref, R

def path_following_cost(state, i, Q, ref, R):
  return Q, ref[i], R

def eval(system, controller, x0, T_sim, dt_sim, N_sim_iter=10):
  xn = x0
  t0 = 0.
    
  states = [x0]
  inputs = []

  times = [0]
  computation_times_ms = []

  for j in range(int(T_sim / dt_sim)):
    # print("x0")
    # print(xn)
    # print(system.step(xn, np.zeros(system.input_dim), dt_sim, method='heun'))
    t = t0 = j * dt_sim

    start = time.process_time_ns()
    u = controller(xn, t)
    end = time.process_time_ns()

    # finer simulation
    for i in range(N_sim_iter):
      xn = system.step(xn, u, dt_sim/N_sim_iter, method='heun')

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

def get_linear_spacing_v2(dt0, T, steps):
  alpha = 2 *(T-dt0 - (steps-1) * dt0) / ((steps-1) * (steps-2))
  return [dt0] + [dt0 + i * alpha for i in range(steps)]

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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  # mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
  for times, states, inputs, computation_times in [nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
    plt.figure()
    plt.plot(times, states, label=['x', 'v_x', 'y', 'v_y'])
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

def squircle(t):
  w = t * 0.8
  x = abs(np.cos(w)) ** (2/8) * np.sign(np.cos(w))
  y = abs(np.sin(w)) ** (2/8) * np.sign(np.sin(w))
  return np.array([x, 0, y, 0]).reshape((-1, 1))

def square(t):
  t = (t) % 4
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
  # return np.array([x, 0, 0, y, 0, 0]).reshape((-1, 1))
  # return np.array([x, y, 0]).reshape((-1, 1))
  # return np.array([x, y, 0, 0, 0, 0]).reshape((-1, 1))

def test_masspoint_ref_path():
  T_sim = 8

  sys = Masspoint2D()
  x = np.zeros(4)
  x[2] = 5
  u = np.zeros(2)

  Q = np.diag([1, 0, 1, 0])
  R = np.diag([.1, .1])

  ref = lambda t: np.array([square(t)[0], 0, square(t)[1], 0]).reshape(-1, 1)

  x = ref(0).flatten()

  quadratic_cost = QuadraticCost(Q, R, Q * 0.01)

  dt = 0.01
  nmpc = NMPC(sys, 5, dt*20, quadratic_cost, ref)

  dts = get_linear_spacing(dt, 100 * dt, 5)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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
  Q = np.diag([10, 10, 0.0001, 0.0001, 0.0001, 0.0001])
  # ref[3] = 0.3
  R = np.diag([0.001, 0.001])

  ref = lambda t: np.array([square(t)[0], square(t)[1], 0, 0, 0, 0]).reshape(-1, 1)

  x = ref(0).flatten()
  x = np.array([0.5, -0.5, np.pi/2, 0.5, 0, 0])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5]))
  input_scaling = 1 / (np.array([1, 0.35]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  dts = get_linear_spacing(dt, 30 * dt, 10)
  # dts = get_linear_spacing_v2(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  x = ref(0).flatten()
  x = np.array([0.5, -0.5, np.pi/2])

  # Q = np.diag([1, 1, 0, 0, 0, 0])
  # ref = np.zeros((6, 1))
  # ref[3] = 0.3
  # R = np.diag([0.1, 0.1])

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / (np.array([1, 1, 1]))
  input_scaling = 1 / (np.array([1, 1]))
  
  # state_scaling = 1 / np.array([1, 1, 2*np.pi, 10, 10, 5])
  # input_scaling = 1 / np.array([1, 0.5])

  nmpc = NMPC(sys, 20, dt, quadratic_cost, ref)

  nmpc.state_scaling = state_scaling
  nmpc.input_scaling = input_scaling

  # dts = get_linear_spacing(dt, 20 * dt, 10)
  dts = get_linear_spacing_v2(dt, 20 * dt, 10)
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  # for times, states, inputs, computation_times in [mpc_sol]:
  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  print("NMPC")
  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  print("NU-MPC")
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

  nmpc_control = lambda x, t: nmpc.compute(x, t)
  numpc_control = lambda x, t: nu_mpc.compute(x, t)

  print("Starting sim for nmpc")
  mpc_sol = eval(sys, nmpc_control, x, T_sim, dt, 100)

  print("Starting sim for numpc")
  nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt, 100)

  for times, states, inputs, computation_times in [mpc_sol, nu_mpc_sol]:
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

      nmpc_control = lambda x, t: nmpc.compute(x, t)
      numpc_control = lambda x, t: nu_mpc.compute(x, t)

      print("NMPC")
      mpc_sol = eval(sys, nmpc_control, x, T_sim, dt)
      print("NU-MPC")
      nu_mpc_sol = eval(sys, numpc_control, x, T_sim, dt)

      for j, (times, states, inputs, computation_times) in enumerate([mpc_sol, nu_mpc_sol]):
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

if __name__ == "__main__":
  # test_quadcopter()
  # test_linearization_quadcopter()
  # test_linearization_cstr()
  # test_linearization_batchreactor()
  # test_linearization_jerk()

  # test_racecar()
  # test_racecar_ref_path()

  # test_laplacian_dynamics()
  # test_masspoint()
  # test_masspoint_ref_path()
  # test_jerk_masspoint_ref_path()

  # test_unicycle()
  test_unicycle_ref_path()

  # test_jerk_masspoint()
  # test_chain_of_masses()
  # test_cstr()
  # test_batch_reactor()

  # make_cost_computation_curve()

  # cartpole_test()
  # double_cartpole_test()
