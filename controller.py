import numpy as np
import math

import jax
import jax.numpy as jnp

import time

import cvxpy as cp
from scipy.interpolate import interp1d, make_interp_spline

from systems import MasspointND

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
    self.sqp_iters = 3
    self.sqp_mixing = 0.8

    self.first_run = True

    self.linearization = jax.jit(self.sys.linearization)

    self.constraints_with_slack = True

    self.move_blocking = False
    self.blocks = None

    self.prev_t = 0

  def compute_initial_state_guess(self, x, t):
    if self.first_run:
      for i in range(self.N+1):
        self.prev_x.append(x)

      self.prev_x = np.array(self.prev_x).T
    else:
      # shift solution by one
      # self.prev_x = np.hstack((self.prev_x[:, 1:], self.prev_x[:, -1][:, None]))
      # self.prev_x[:, 0] = x

      times = np.array([0] + list(np.cumsum(self.dts)))

      # f = interp1d(times, self.prev_x, axis=1, bounds_error=False, fill_value=self.prev_x[:, -1])
      f = interp1d(times, self.prev_x, kind='linear', axis=1, bounds_error=False, fill_value=self.prev_x[:, -1])
      diff = t - self.prev_t
      new_x = f(times + diff)
      
      self.prev_x = new_x
      self.prev_x[:, 0] = x

  def compute_initial_control_guess(self):
    if self.first_run:
      for _ in range(self.N):
        self.prev_u.append(np.zeros(self.input_dim))

      self.prev_u = np.array(self.prev_u).T
    else:
      # shift solution by one
      # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      f = interp1d(times, self.prev_u, kind='zero', axis=1, bounds_error=False, fill_value=self.prev_u[:, -1])
      new_u = f(times + self.dts[0])

      self.prev_u = new_u

  def compute(self, state, t):
    x = cp.Variable((self.state_dim, self.N+1))
    u = cp.Variable((self.input_dim, self.N))

    s = cp.Variable((self.state_dim, (self.N+1)*2))

    S = np.diag(self.state_scaling)
    Sinv = np.diag(1 / self.state_scaling)

    W = np.diag(self.input_scaling)
    Winv = np.diag(1 / self.input_scaling)

    self.compute_initial_state_guess((S @ state[:, None]).flatten(), t)
    self.compute_initial_control_guess()

    if not self.first_run:
      print(self.prev_x[:, 0])
    
    print("state")
    print(self.prev_x)

    self.solve_time = 0

    # dynamics constraints
    for k in range(self.sqp_iters):
      start = time.process_time_ns()

      constraints = []
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]
        print(i, dt, t + sum(self.dts[:i]))

        # this should be shifted by one step
        prev_state = (Sinv @ self.prev_x[:, i][:, None]).flatten()
        prev_input = (Winv @ self.prev_u[:, i][:, None]).flatten()

        # print("state")
        # print(prev_state)
        # next_state = self.sys.step(prev_state, prev_input, dt, 'euler')
        # print(next_state)
        # next_state = self.sys.step(next_state, prev_input, dt, 'euler')
        # print(next_state)

        A, B, K = self.linearization(prev_state, prev_input, dt)

        constraints.append(
            x[:, next_idx][:, None] == S @ (A) @ Sinv @ x[:, idx][:, None] + S @ B @ Winv @ u[:, idx][:, None] - S @ K
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

      if self.move_blocking:
        idx = 0
        for block_len in self.blocks:
          if block_len > 1:
            for i in range(1, block_len):
              constraints.append(u[:, idx] == u[:, idx + i])
              print(idx, idx + i)
          
          idx += block_len

      # state constraints
      if self.constraints_with_slack:
        if not self.first_run:
          for i in range(self.N + 1):
            constraints.append(x[:, i][:, None] >= S @ self.sys.state_limits[:, 0][:, None] - s[:, i*2][:, None])
            constraints.append(x[:, i][:, None] <= S @ self.sys.state_limits[:, 1][:, None] + s[:, i*2+1][:, None])

        # slack variables
        for i in range((self.N + 1)*2):
          constraints.append(s[:, i] >= 0)
      else:
        if not self.first_run:
          for i in range(self.N + 1):
            constraints.append(x[:, i][:, None] >= S @ self.sys.state_limits[:, 0][:, None])
            constraints.append(x[:, i][:, None] <= S @ self.sys.state_limits[:, 1][:, None])

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

      # TODO: normalize cost by total prediction horizn?
      cost = 0
      curr_time = t
      for i in range(self.N): 
        dt = self.dts[i]
        x_ref = self.ref(curr_time)
        cost += dt * cp.quad_form((x[:, i] [:, None] - S @ x_ref), Qs)
        cost += dt * cp.quad_form(u[:,i][:, None], Rs)

        curr_time += dt

      # terminal cost
      x_ref = self.ref(curr_time)
      cost += cp.quad_form((x[:, -1] [:, None] - S @ x_ref), QNs)

      # cost = cost * 10
      # cost = cost / 1000
      # cost = cost * 100
      # cost = cost / 10

      # slack variables
      if self.constraints_with_slack:
        for i in range(self.N):
          dt = self.dts[i]
          cost += dt * 50 * cp.sum(s[:, (i+1)*2])
          cost += dt * 50 * cp.sum(s[:, (i+1)*2+1])

          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2])
          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2+1])

      objective = cp.Minimize(cost)

      end = time.process_time_ns()

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      # The optimal objective value is returned by `prob.solve()`.
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-6, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(solver='OSQP', verbose=True,eps_abs=1e-7, eps_rel=1e-5, max_iter=10000)
      # result = prob.solve(solver='ECOS', verbose=True, max_iters=1000, feastol=1e-5, reltol=1e-4, abstol_inacc=1e-5, reltol_inacc=1e-5, feastol_inacc=1e-5)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-8)
      # result = prob.solve(solver='PIQP', verbose=True)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-8)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-2)

      result = prob.solve(solver='CLARABEL', verbose=True, max_iter=5000)
      # result = prob.solve(solver='QPALM', verbose=True, max_iter=5000)

      # options_cvxopt = {
      #     "max_iters": 5000,
      #     "verbose": True,
      #     "abstol": 1e-21,
      #     "reltol": 1e-11,
      #     "refinement": 2, # higher number seemed to make things worse
      #     "kktsolver": "robust"
      # }
      # result = prob.solve(solver='CVXOPT', **options_cvxopt)

      if prob.status in ["infeasible", "unbounded"]:
        infeasible_res = np.ones(self.sys.input_dim)
        infeasible_res[0] = np.nan
        return infeasible_res

      print("Sols")
      print(x.value)
      print(u.value)

      # update solutions for linearization
      # TODO: should really be a line search
      if self.first_run:
        self.prev_x = x.value
        self.prev_u = u.value
      else:
        self.prev_x = x.value * self.sqp_mixing + self.prev_x * (1 - self.sqp_mixing)
        self.prev_u = u.value * self.sqp_mixing + self.prev_u * (1 - self.sqp_mixing)

      self.first_run = False

      self.solve_time += prob.solver_stats.solve_time
      # print('s', prob.solver_stats.solve_time)
      # lin_sys_time = prob.solver_stats.extra_stats["info"]["lin_sys_time"]
      # cone_time = prob.solver_stats.extra_stats["info"]["cone_time"]
      # accel_time = prob.solver_stats.extra_stats["info"]["accel_time"]
      # setup_time = prob.solver_stats.extra_stats["info"]["setup_time"]

      # self.solve_time += (lin_sys_time + cone_time + accel_time + setup_time) / 1000

      # self.solve_time += (end - start) / 1e9

      # data, _, _= prob.get_problem_data(cp.OSQP)

      # print(data)

      # print("A")
      # print(data['A'].todense())
      # np.savetxt("foo.csv", data["A"].todense(), delimiter=",")

      # print("B")
      # print(data['b'])

    self.prev_t = t

    if prob.status not in ["infeasible", "unbounded"]:
      print("U")
      print(u.value[:, 0])
      print( Winv @ u.value[:, 0])

      return Winv @ u.value[:, 0]
      # return u.value[:, 0]

    return np.zeros(self.sys.input_dim)
  
def get_linear_spacing_with_max_dt(T, dt0, dt_max, steps):
  # copy and paste from claude
  def calculate_T(alpha, dt_max, dt, N):
    if alpha == 0:
      return N * min(dt_max, dt)
    
    ts = [np.min([dt + alpha * i, dt_max]) for i in range(N)]
    return sum(ts)

  def solve_for_alpha(target_T, dt0, dt_max, N, tolerance=1e-6, max_iterations=1000):
    alpha_min = 0
    alpha_max = (dt_max - dt0) * 2 / N  # An initial guess for upper bound
    print(alpha_max)

    for _ in range(max_iterations):
      alpha = (alpha_min + alpha_max) / 2
      calculated_T = calculate_T(alpha, dt_max, dt0, N)
      
      if abs(calculated_T - target_T) < tolerance:
        return alpha
      
      if calculated_T < target_T:
        alpha_min = alpha
      else:
        alpha_max = alpha
    
    return alpha  # Return best approximation

  alpha = solve_for_alpha(T, dt0, dt_max, steps)
  print(alpha)

  return [min(dt0 + i * alpha, dt_max) for i in range(steps)]

def get_linear_spacing(dt0, T, steps):
  alpha = 2 *(T - steps * dt0) / (steps * (steps-1))
  return [dt0 + i * alpha for i in range(steps)]

def get_linear_spacing_v2(dt0, T, steps):
  alpha = 2 *(T-dt0 - (steps-1) * dt0) / ((steps-1) * (steps-2))
  return [dt0] + [dt0 + i * alpha for i in range(steps)]

def get_power_spacing(dt0, T, steps):
  def solve_for_alpha(T, dt, N, max_iterations=100, tolerance=1e-6):
    def f(alpha):
      return dt * (((1 + alpha) ** (N + 1) - 1) / alpha) - T

    def f_prime(alpha):
      return dt * ((N + 1) * (1 + alpha)**N / alpha - ((1 + alpha)**(N + 1) - 1) / alpha**2)

    # Initial guess
    alpha = 0.1

    for _ in range(max_iterations):
      f_value = f(alpha)
      if abs(f_value) < tolerance:
          return alpha

      f_prime_value = f_prime(alpha)
      if f_prime_value == 0:
          return None  # To avoid division by zero

      alpha = alpha - f_value / f_prime_value

    return None  # If no solution found within max_iterations
  
  alpha = solve_for_alpha(T, dt0, steps-1, 1000, 1e-6)
  dts = [dt0 * (1+alpha)**i for i in range(steps)]

  return dts

class NU_NMPC(Controller):
  def __init__(self, system, dts, quadratic_cost, reference):
    self.nmpc = NMPC(system, len(dts), 0, quadratic_cost, reference)

    self.nmpc.dts = dts

  def compute(self, state, t):
    res = self.nmpc.compute(state, t)

    self.solve_time = self.nmpc.solve_time
    self.prev_x = self.nmpc.prev_x
    self.prev_u = self.nmpc.prev_u
    
    return res

class MoveBlockingNMPC(Controller):
  def __init__(self, system, N, dt, quadratic_cost, reference, blocks):
    self.nmpc = NMPC(system, N, dt, quadratic_cost, reference)

    self.nmpc.blocks = blocks
    self.nmpc.move_blocking = True

  def compute(self, state, t):
    res = self.nmpc.compute(state, t)
    self.solve_time = self.nmpc.solve_time
    return res
  
def get_linear_blocking(steps, num_free_vars):
  dts = get_linear_spacing(1, steps, num_free_vars)
  blocks = [int(np.floor(dt)) for dt in dts]
  sum_naive = sum(blocks)
  blocks[-1] = blocks[-1] + (steps - sum_naive)

  return blocks

class NMPCC(Controller):
  def __init__(self, system, dts, H, reference):
    self.prev_x = []
    self.prev_u = []

    self.dts = dts
    self.N = len(dts)
    self.H = H

    self.sys = system
    self.linearization = jax.jit(self.sys.linearization)

    # make spline approximation of reference path
    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    ts = np.linspace(0, 20, 10000)
    pts = [self.ref(t) for t in ts]
    self.path = make_interp_spline(ts, pts)
    self.path_derivative = self.path.derivative(1)
    self.path_second_derivative = self.path.derivative(2)

    self.progress_system = MasspointND(1)
    self.progress_linearization = jax.jit(self.progress_system.linearization)

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    self.prev_x = []
    self.prev_u = []

    self.prev_p = []
    self.prev_up = []

    self.state_scaling = np.array([1] * self.sys.state_dim)
    self.input_scaling = np.array([1] * self.sys.input_dim)

    self.progress_bounds = np.array([[0, 20], [0, 3.5]])
    self.progress_input_bounds = np.array([[-5, 5]])

    self.progress_scaling = 1 / np.array([10, 5])
    self.progress_acc_scaling = 1 / np.array([5])

    self.contouring_weight = 0.5
    self.progress_weight = 0.1

    self.input_reg = 1e-4
    self.progress_acc_reg = 1e-2

    self.lag_weight = 2000
    self.cont_weight = 0.05

    # solver params
    self.sqp_iters = 3
    self.sqp_mixing = 0.5

    self.first_run = True

    self.state_constraints_with_slack = True

    self.diff_from_center = 0.15
    self.track_constraints = True
    self.track_constraints_with_slack = True

    self.track_constraint_linear_weight = 1000
    self.track_constraint_quadratic_weight = 2000

    self.move_blocking = False
    self.blocks = None

    self.prev_t = 0

  def compute_initial_state_guess(self, x, t):
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
      
      diff = t - self.prev_t
      new_x = f(times + diff)
      
      self.prev_x = new_x
      self.prev_x[:, 0] = x

  def compute_initial_control_guess(self, t):
    if self.first_run:
      for _ in range(self.N):
        self.prev_u.append(np.zeros(self.input_dim))

      self.prev_u = np.array(self.prev_u).T
    else:
      # shift solution by one
      # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      f = interp1d(times, self.prev_u, axis=1, bounds_error=False, fill_value=self.prev_u[:, -1])
      
      diff = t - self.prev_t
      new_u = f(times + diff)

      self.prev_u = new_u

  def project_state_to_progess(self, state, time):
    if self.first_run:
      return 0

    diff = time - self.prev_t
    initial_theta_guess = 1. / self.progress_scaling[0] * self.prev_p[0, 0] + \
        diff * 1. / self.progress_scaling[1] * self.prev_p[1, 0] + \
        diff ** 2 / 2. * 1. / self.progress_acc_scaling[0] * self.prev_up[0, 0]

    if True:
      # temporary for testing
      return 1. / self.progress_scaling[0] * self.prev_p[0, 1]

      return initial_theta_guess

    min_error = 1e6
    theta_best = 0
    max_diff = self.progress_bounds[1,1] * diff

    print(max_diff)

    for t in np.linspace(0, 20, 1000):
      e = self.error(state, t)
      if abs(t - initial_theta_guess) < max_diff and e < min_error:
      # if e < min_error:
        min_error = e
        theta_best = t

    return theta_best

  def compute_initial_progress_guess(self, state, T, t):
    theta = self.project_state_to_progess(state, t)

    if self.first_run:
      for i in range(self.N+1):
        self.prev_p.append((T @ np.array([theta, 0.05])[:, None]).flatten())

      self.prev_p = np.array(self.prev_p).T
    else:
      times = np.array([0] + list(np.cumsum(self.dts)))

      f = interp1d(times, self.prev_p, axis=1, bounds_error=False, fill_value=self.prev_p[:, -1])

      diff = t - self.prev_t
      new_p = f(times + diff)
      
      self.prev_p = new_p
      self.prev_p[0, 0] = T[0,0] * theta
    
  def compute_initial_progress_input_guess(self, t):
    if self.first_run:
      for i in range(self.N):
        self.prev_up.append(np.ones(1)*0.1)

      self.prev_up = np.array(self.prev_up).T
    else:
      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      # print(times)
      # print(len(times))
      # print(self.prev_up)
      # print(len(self.prev_up))
      f = interp1d(times, self.prev_up, axis=1, bounds_error=False, fill_value=self.prev_up[:, -1])

      diff = t - self.prev_t
      new_up = f(times + diff)
      
      self.prev_up = new_up

  def error(self, q, theta):
    residual = self.path(theta) - self.H @ q[:, None]
    return residual.T @ residual

  def error_quad_approx(self, q, theta, proj = None):
    if proj is None:
      proj = np.eye(len(self.path(theta)))

    residual = (self.path(theta) - self.H @ q[:, None])

    # print('state', q)
    # print('path', self.path(theta))
    # print('theta', theta)
    # print("residual:", residual)
    
    path_jac = self.path_derivative(theta)
    path_hess = self.path_second_derivative(theta)

    if True: #np.linalg.norm(path_hess) < 1e-3:
      path_hess *= 0.

    # print("path gradient", path_jac)
    # print("path hessian", path_hess)

    # Compute the gradient
    grad_theta = 2 * (self.path_derivative(theta).T @ proj.T @ proj @ residual)
    grad_q = - 2 * (self.H.T @ proj.T @ proj @ residual)

    # Compute the Hessian
    # hess_theta = (path_jac.T @ proj.T @ proj @ path_jac + path_hess.T @ proj.T @ proj @ residual)
    hess_theta = 2 * (path_jac.T @ proj.T @ proj @ path_jac)
    # hess_theta = 2 * (path_jac.T @ path_jac)
    hess_q = 2 * (self.H.T @ proj.T @ proj @ self.H)
    hess_theta_q = - 2*(path_jac.T @ proj.T @ proj @ self.H)

    # print("other part", path_jac.T @ path_jac)
    # print("residual thingy:", path_hess.T @ residual)

    return hess_q, hess_theta, hess_theta_q, grad_q, grad_theta

  def lag_error(self, q, theta):
    tangent = self.path_derivative(theta)
    tangent = tangent / np.linalg.norm(tangent)
    mat = tangent @ tangent.T

    mat = 0.5 * mat + 0.5 * mat.T

    # print('proj mat')
    # print(mat)

    return self.error_quad_approx(q, theta, mat)

  def contouring_error(self, q, theta):
    tangent = self.path_derivative(theta)
    tangent = tangent / np.linalg.norm(tangent)

    mat = np.eye(len(tangent)) - tangent @ tangent.T

    # print('other proj mat')
    # print(mat)
    
    return self.error_quad_approx(q, theta, mat)

  def compute(self, state, t):
    x = cp.Variable((self.state_dim, self.N+1))
    u = cp.Variable((self.input_dim, self.N))

    # state slack variables
    s = cp.Variable((self.state_dim, self.N+1))

    # reference slack variables
    rs = cp.Variable((1, (self.N+1)))

    p = cp.Variable((2, self.N+1))
    up = cp.Variable((1, self.N))

    # project state to path to obtain progress var

    # S @ x = x_norm
    S = np.diag(self.state_scaling)
    Sinv = np.diag(1 / self.state_scaling)

    # W @ u = u_norm
    W = np.diag(self.input_scaling)
    Winv = np.diag(1 / self.input_scaling)

    # T @ prog = p_norm
    T = np.diag(self.progress_scaling)
    Tinv = np.diag(1 / self.progress_scaling)

    # Tu @ pu = pu_norm
    Tu = np.diag(self.progress_acc_scaling)
    Tuinv = np.diag(1 / self.progress_acc_scaling)

    self.compute_initial_state_guess((S @ state[:, None]).flatten(), t)
    self.compute_initial_control_guess(t)

    self.compute_initial_progress_guess(state, T, t)
    self.compute_initial_progress_input_guess(t)

    if not self.first_run:
      print(self.prev_x[:, 0])

    self.solve_time = 0

    for k in range(self.sqp_iters):
      constraints = []
      # dynamics constraints
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]
        print(i, dt, t + np.sum(self.dts[:i]))

        # this should be shifted by one step
        prev_state = (Sinv @ self.prev_x[:, i][:, None]).flatten()
        prev_input = (Winv @ self.prev_u[:, i][:, None]).flatten()

        A, B, K = self.linearization(prev_state, prev_input, dt)

        constraints.append(
          x[:, next_idx][:, None] == S @ (A) @ Sinv @ x[:, idx][:, None] + S @ B @ Winv @ u[:, idx][:, None] - S @ K
        )

      constraints.append(x[:, 0][:, None] == S @ state[:, None])

      # add dynamics for progress
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]

        prev_progress = (Tinv @ self.prev_p[:, i][:, None]).flatten()
        prev_input = (Tuinv @ self.prev_up[:, i][:, None]).flatten()

        # print(i)
        # print('progress at time ', prev_progress[0])
        # print('prog vel at time ', prev_progress[1])
        # print('prog. acc. at time ', prev_input[0])
        # print('delta t (progress)', dt)

        A, B, K = self.progress_linearization(prev_progress, prev_input, dt)

        # print(A)
        # print(B)

        constraints.append(
          p[:, next_idx][:, None] == T @ A @ Tinv @ p[:, idx][:, None] + T @ B @ Tuinv @ up[:, idx][:, None] - T @ K
        )
      
      theta = self.project_state_to_progess(state, t)
      constraints.append(p[0, 0] == T[0, 0] * theta)

      # input constraints
      for i in range(self.N):
        constraints.append(u[:, i][:, None] >= W @ self.sys.input_limits[:, 0][:, None])
        constraints.append(u[:, i][:, None] <= W @ self.sys.input_limits[:, 1][:, None])

      # progress constraints
      for i in range(self.N):
        constraints.append(up[:, i][:, None] >= Tu @ self.progress_input_bounds[:, 0][:, None])
        constraints.append(up[:, i][:, None] <= Tu @ self.progress_input_bounds[:, 1][:, None])
    
      for i in range(self.N+1):
        constraints.append(p[:, i][:, None] >= T @ self.progress_bounds[:, 0][:, None])
        constraints.append(p[:, i][:, None] <= T @ self.progress_bounds[:, 1][:, None])

      if self.move_blocking:
        idx = 0
        for block_len in self.blocks:
          if block_len > 1:
            for i in range(1, block_len):
              constraints.append(u[:, idx] == u[:, idx + i])
              print(idx, idx + i)
          
          idx += block_len

      # state constraints
      if self.state_constraints_with_slack:
        if not self.first_run:
          for i in range(self.N + 1):
            constraints.append(x[:, i][:, None] >= S @ self.sys.state_limits[:, 0][:, None] - s[:, i][:, None])
            constraints.append(x[:, i][:, None] <= S @ self.sys.state_limits[:, 1][:, None] + s[:, i][:, None])

        # slack variables
        for i in range((self.N + 1)):
          constraints.append(s[:, i] >= 0)
      else:
        if not self.first_run:
          for i in range(self.N + 1):
            constraints.append(x[:, i][:, None] >= S @ self.sys.state_limits[:, 0][:, None])
            constraints.append(x[:, i][:, None] <= S @ self.sys.state_limits[:, 1][:, None])

      if self.track_constraints:
        for i in range(1, self.N+1):
          # print(self.path(self.prev_p[0, i]))
          residual = self.H @ Sinv @ x[:, i][:, None] - self.path(Tinv[0,0]*self.prev_p[0, i])
          # constraints.append(residual.T @ residual <= 0.5)

          if self.track_constraints_with_slack:
            constraints.append(cp.sum_squares(residual) <= self.diff_from_center**2 + rs[0, i])
            constraints.append(rs[0, i] >= 0)
          else:
            constraints.append(cp.sum_squares(residual) <= self.diff_from_center**2)

      cost = 0

      # TODO: normalize cost by total prediction horizn?
      Rs = self.input_reg * Winv @ np.eye(self.input_dim) @ Winv

      # curr_time = t
      for i in range(self.N): 
        dt = self.dts[i]
        cost += dt * cp.quad_form(u[:,i][:, None], Rs)
        cost += dt * self.progress_acc_reg * Tuinv[0, 0]**2 * up[0, i]**2

        # curr_time += dt

      for i in range(1, self.N+1):
        dt = self.dts[i-1]

        # quadratize error
        prev_progress = (Tinv @ self.prev_p[:, i][:, None]).flatten()
        prev_state = (Sinv @ self.prev_x[:, i][:, None]).flatten()

        # print("Step", i)
        # print(prev_progress)

        if False:
          # hess_q, hess_theta, hess_theta_q, grad_q, grad_theta = self.error_quad_approx(prev_state, prev_progress[0])
          
          hess_lag_q, hess_lag_theta, hess_lag_theta_q, grad_lag_q, grad_lag_theta = self.lag_error(prev_state, prev_progress[0])
          hess_cont_q, hess_cont_theta, hess_cont_theta_q, grad_cont_q, grad_cont_theta = self.contouring_error(prev_state, prev_progress[0])

          hess_q = self.lag_weight * hess_lag_q + self.cont_weight * hess_cont_q
          hess_theta = self.lag_weight * hess_lag_theta + self.cont_weight * hess_cont_theta
          hess_theta_q = self.lag_weight * hess_lag_theta_q + self.cont_weight * hess_cont_theta_q

          grad_q = self.lag_weight * grad_lag_q + self.cont_weight * grad_cont_q
          grad_theta = self.lag_weight * grad_lag_theta + self.cont_weight * grad_cont_theta

          hess = np.block([[hess_q, hess_theta_q.T],
                           [hess_theta_q, hess_theta]])
          
          hess  = 0.5 * hess + 0.5 * hess.T + np.eye(hess.shape[0]) * 1e-4
          
          p_reshaped = cp.reshape(p[0, i], (1, 1))
          x_reshaped = cp.reshape(Sinv @ x[:, i][:, None], (self.state_dim, 1))
          tmp = cp.vstack([x_reshaped, p_reshaped])

          offset = np.vstack((prev_state[:, None], prev_progress[0]))

          cost += dt * self.contouring_weight * 0.5 * cp.quad_form(tmp - offset, hess)

          cost += dt * self.contouring_weight * grad_q.T @ Sinv @ (x[:, i][:, None] - prev_state[:, None])
          cost += dt * self.contouring_weight * grad_theta.flatten() * (p[0, i] - prev_progress[0])
        else:
          tangent = self.path_derivative(prev_progress[0])
          tangent_norm = tangent / np.linalg.norm(tangent)
          lag = tangent_norm @ tangent_norm.T
          lag = 0.5 * lag + 0.5 * lag.T

          cont = np.eye(len(tangent_norm)) - lag
          cont = 0.5 * cont + 0.5 * cont.T

          cost += self.contouring_weight * dt * cp.quad_form(
            (self.path(prev_progress[0]) + tangent * (Tinv[0,0] * p[0, i] - prev_progress[0])) - self.H @ Sinv @ x[:, i][:, None]
            , self.cont_weight * cont @ cont.T + self.lag_weight * lag @ lag.T + np.eye(lag.shape[0]) * 1e-3)

      for i in range(0, self.N):
        dt = self.dts[i]
        cost -= dt * self.progress_weight * Tinv[1, 1] * p[1, i]

      # cost = cost * 10
      # cost = cost / 1000
      # cost = cost * 10
      # cost = cost / 10

      # slack variables
      if self.state_constraints_with_slack:
        for i in range(self.N):
          dt = self.dts[i]
          cost += dt * 10 * cp.sum(s[:, (i+1)])
          cost += dt * 500 * cp.sum_squares(s[:, (i+1)])

      if self.track_constraints and self.track_constraints_with_slack:
        for i in range(self.N):
          dt = self.dts[i]
          cost += dt * self.track_constraint_linear_weight * cp.sum(rs[0, (i+1)])
          cost += dt * self.track_constraint_quadratic_weight * cp.sum_squares(rs[0, (i+1)])

      objective = cp.Minimize(cost)

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      start = time.process_time_ns()
      # The optimal objective value is returned by `prob.solve()`.
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-4, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(solver='OSQP', verbose=True,eps_abs=1e-7, eps_rel=1e-5, max_iter=10000)
      # result = prob.solve(solver='ECOS', verbose=True, max_iters=1000, feastol=1e-5, reltol=1e-4, abstol_inacc=1e-5, reltol_inacc=1e-5, feastol_inacc=1e-5)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-4)
      # result = prob.solve(solver='SCS', verbose=True, eps=1e-8, normalize=False, acceleration_lookback=-10)
      # result = prob.solve(solver='PIQP', verbose=True)
      result = prob.solve(solver='CLARABEL', verbose=True, max_iter=500)

      # options_cvxopt = {
      #     "max_iters": 5000,
      #     "verbose": True,
      #     "abstol": 1e-21,
      #     "reltol": 1e-11,
      #     "refinement": 2, # higher number seemed to make things worse
      #     "kktsolver": "robust"
      # }
      # result = prob.solve(solver='CVXOPT', **options_cvxopt)

      end = time.process_time_ns()

      print("Sols")
      print(x.value)
      print(u.value)

      # update solutions for linearization
      # TODO: should really be a line search
      if self.first_run:
        self.prev_x = x.value
        self.prev_u = u.value

        self.prev_p = p.value
        self.prev_up = up.value
      else:
        self.prev_x = x.value * self.sqp_mixing + self.prev_x * (1 - self.sqp_mixing)
        self.prev_u = u.value * self.sqp_mixing + self.prev_u * (1 - self.sqp_mixing)

        self.prev_p = p.value * self.sqp_mixing + self.prev_p * (1 - self.sqp_mixing)
        self.prev_up = up.value * self.sqp_mixing + self.prev_up * (1 - self.sqp_mixing)

      self.first_run = False

      # self.solve_time += prob.solver_stats.solve_time
      self.solve_time += (end - start) / 1e9

      # data, _, _= prob.get_problem_data(cp.OSQP)

      # print(data)

      # print("A")
      # print(data['A'].todense())
      # np.savetxt("foo.csv", data["A"].todense(), delimiter=",")

      # print("B")
      # print(data['b'])
      print('progress', p.value[:, -1])

    if prob.status not in ["infeasible", "unbounded"]:
      print("U")
      print(u.value[:, 0])
      print( Winv @ u.value[:, 0])

      self.prev_t = t

      return Winv @ u.value[:, 0]
      # return u.value[:, 0]

    return np.zeros(self.sys.input_dim)

class PredictiveRandomSampling(Controller):
  def eval_cost(self, state, input, t):
    x = state
    # jax.debug.print("{x}", x=x)
    cost = 0
    for i, dt in enumerate(self.dts):
      diff = (x[:, None] - self.ref(t))

      # print(diff)

      cost += dt * diff.T @ self.cost.Q @ diff
      cost += dt * input[:, i][:, None].T @ self.cost.R @ input[:, i][:, None]

      state_lb_violation = (x[:, None] - self.sys.state_limits[:, 0][:, None]) > 0
      state_ub_violation = (x[:, None] - self.sys.state_limits[:, 1][:, None]) < 0

      w_violation = 100
      lb_violation = (jax.numpy.sum(x[:, None] - self.sys.state_limits[:, 0][:, None], where=state_lb_violation))**2
      ub_violation = (jax.numpy.sum(x[:, None] - self.sys.state_limits[:, 1][:, None], where=state_ub_violation))**2

      cost += dt * w_violation * (lb_violation + ub_violation)
      x = self.sys.step(x, input[:, i], dt, "rk4")

      t += dt

    diff = (x[:, None] - self.ref(t))
    cost += diff.T @ self.cost.QN @ diff

    return cost

  def __init__(self, system, dts, quadratic_cost, reference, var=None, num_rollouts=10000):
    self.sys = system
    self.N = len(dts)
    self.dts = dts

    self.num_rollouts = num_rollouts

    self.cost = quadratic_cost

    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    if var is None:
      self.var = np.eye(self.input_dim)
    else:
      self.var = var

    self.prev_x = []
    self.prev_u = []

    self.vectorized_rollout = jax.vmap(self.eval_cost, in_axes=(None, 0, None))
    self.jitted_rollout = jax.jit(self.vectorized_rollout)
    # self.jitted_rollout = self.vectorized_rollout

    self.key = jax.random.PRNGKey(0)

    self.prev_best_control = jax.numpy.zeros(shape=(self.input_dim, self.N))

  def compute(self, state, t):
    # shift prev best control
    # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

    times = np.array([0] + list(np.cumsum(self.dts[:-1])))

    f = interp1d(times, self.prev_best_control, axis=1, bounds_error=False, fill_value=self.prev_best_control[:, -1])
    new_u = f(times + self.dts[0])

    self.prev_best_control = new_u

    # random_parameters = self.var @ jax.random.normal(key=self.key, shape=(self.num_rollouts, self.input_dim))
    diff_random_parameters = jax.numpy.zeros(shape=(self.num_rollouts * 2, self.input_dim, self.N))
    
    diff_random_parameters = diff_random_parameters.at[0:self.num_rollouts].set(
      1. * jax.random.normal(key=self.key, shape=(self.num_rollouts, self.input_dim, self.N)))
    diff_random_parameters = diff_random_parameters.at[self.num_rollouts:].set(
      jax.random.uniform(key=self.key, minval=-2, maxval=2, shape=(self.num_rollouts, self.input_dim, self.N)))

    random_parameters = self.prev_best_control + diff_random_parameters
    # random_parameters = diff_random_parameters

    random_parameters = jax.numpy.where(random_parameters < 10, random_parameters, 10)
    random_parameters = jax.numpy.where(random_parameters > -10, random_parameters, -10)

    # control_parameters_vec = best_control_parameters + additional_random_parameters

    costs = self.jitted_rollout(state, random_parameters, t)
    
    best_index = jax.numpy.nanargmin(costs)
    best_cost = costs.take(best_index)
    best_control_parameters = random_parameters[best_index]

    self.solve_time = 0
    self.prev_x = 0
    self.prev_u = 0

    self.prev_best_control = best_control_parameters

    return best_control_parameters[:, 0]

class MPPI(Controller):
  def eval_cost(self, state, input, t):
    x = state
    # jax.debug.print("{x}", x=x)
    cost = 0
    for i, dt in enumerate(self.dts):
      diff = (x[:, None] - self.ref(t))
      # jax.debug.print("{diff}", diff=diff)
      # print(diff)

      cost += dt * diff.T @ self.cost.Q @ diff
      cost += dt * input[:, i][:, None].T @ self.cost.R @ input[:, i][:, None]

      state_lb_violation = (x[:, None] - self.sys.state_limits[:, 0][:, None]) > 0
      state_ub_violation = (x[:, None] - self.sys.state_limits[:, 1][:, None]) < 0

      w_violation = 10
      lb_violation = (jax.numpy.sum(x[:, None] - self.sys.state_limits[:, 0][:, None], where=state_lb_violation))**2
      ub_violation = (jax.numpy.sum(x[:, None] - self.sys.state_limits[:, 1][:, None], where=state_ub_violation))**2

      cost += dt * w_violation * (lb_violation + ub_violation)
      x = self.sys.step(x, input[:, i], dt, "euler")

      t += dt

    diff = (x[:, None] - self.ref(t))
    cost += diff.T @ self.cost.QN @ diff

    return cost

  def __init__(self, system, dts, quadratic_cost, reference, var=None, num_rollouts=10000):
    self.sys = system
    self.N = len(dts)
    self.dts = dts

    self.num_rollouts = num_rollouts

    self.cost = quadratic_cost

    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    if var is None:
      self.var = np.ones(self.input_dim) * 20
    else:
      self.var = var

    self.prev_x = []
    self.prev_u = []

    self.vectorized_rollout = jax.vmap(self.eval_cost, in_axes=(None, 0, None))
    self.jitted_rollout = jax.jit(self.vectorized_rollout)
    # self.jitted_rollout = self.vectorized_rollout

    self.key = jax.random.PRNGKey(0)

    self.prev_best_control = jax.numpy.zeros(shape=(self.input_dim, self.N))

  def compute(self, state, t):
    # shift prev best control
    # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

    times = np.array([0] + list(np.cumsum(self.dts[:-1])))

    f = interp1d(times, self.prev_best_control, axis=1, bounds_error=False, fill_value=self.prev_best_control[:, -1])
    new_u = f(times + self.dts[0])

    self.prev_best_control = new_u

    # random_parameters = self.var @ jax.random.normal(key=self.key, shape=(self.num_rollouts, self.input_dim))
    diff_random_parameters = jax.numpy.zeros(shape=(self.num_rollouts * 2, self.input_dim, self.N))
    
    diff_random_parameters = diff_random_parameters.at[0:self.num_rollouts].set(
      self.var[:, None] * jax.random.normal(key=self.key, shape=(self.num_rollouts, self.input_dim, self.N)))
    diff_random_parameters = diff_random_parameters.at[self.num_rollouts:].set(
      jax.random.uniform(key=self.key, minval=-40, maxval=40, shape=(self.num_rollouts, self.input_dim, self.N)))

    random_parameters = self.prev_best_control + diff_random_parameters
    # random_parameters = diff_random_parameters

    random_parameters = jnp.clip(random_parameters, self.sys.input_limits[:, 0][:, None], self.sys.input_limits[:, 1][:, None])

    costs = self.jitted_rollout(state, random_parameters, t)
    
    best_index = jax.numpy.nanargmin(costs)
    best_cost = costs.take(best_index)
    best_control_parameters = random_parameters[best_index]

    beta = best_cost
    temp = 1.
    exp_costs = jnp.exp((-1./temp) * (costs - beta))
    denom = np.sum(exp_costs)
    weights = exp_costs/denom
    # print(weights)
    # print(random_parameters)
    # print(weights.shape)
    # print(random_parameters.shape)
    weighted_inputs = weights * random_parameters
    best_control_parameters = jnp.sum(weighted_inputs, axis=0)

    print(t)
    print(best_control_parameters)

    self.solve_time = 0
    self.prev_x = 0
    self.prev_u = 0

    self.prev_best_control = best_control_parameters

    return best_control_parameters[:, 0]

class PenaltyiLQR(Controller):
  def __init__(self, system, dts, quadratic_cost, reference):
    self.sys = system
    self.N = len(dts)
    self.dts = dts

    self.cost = quadratic_cost

    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    self.state_dim = self.sys.state_dim
    self.input_dim = self.sys.input_dim

    self.prev_u = [np.zeros(self.input_dim)] * len(self.dts)
    self.prev_x = []

    self.max_iters = 10

    self.linearization = jax.jit(self.sys.linearization)

    self.solve_time = 0
    self.first_run = True

    def barrier(state, inp, q1, q2):
      sum = 0

      for i in range(self.state_dim):
        sum += q1 * jnp.exp(q2 * (-state[i] + self.sys.state_limits[i, 0]))
        sum += q1 * jnp.exp(q2 * (state[i] - self.sys.state_limits[i, 1]))

      for i in range(self.input_dim):
        sum += q1 * jnp.exp(q2 * (-inp[i] + self.sys.input_limits[i, 0]))
        sum += q1 * jnp.exp(q2 * (inp[i] - self.sys.input_limits[i, 1]))

      return sum
    
    self.b = barrier
    
    self.bx = jax.jit(jax.jacfwd(barrier, argnums=0))
    self.bxx = jax.jit(jax.jacfwd(self.bx, argnums=0))

    self.bu = jax.jit(jax.jacfwd(barrier, argnums=1))
    self.buu = jax.jit(jax.jacfwd(self.bu, argnums=1))

  def compute_initial_control_guess(self):
    if self.first_run:
      pass
    else:
      # shift solution by one
      # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      print(np.array(self.prev_u).T)

      f = interp1d(times, np.array(self.prev_u).T, axis=1, bounds_error=False, fill_value=self.prev_u[-1])
      new_u = f(times + self.dts[0])

      for i in range(len(self.dts)):
        self.prev_u[i] = new_u[:, i]
      # self.prev_u = new_u

  def cost_w_penalty(self, xs, us, t0):
    c = 0
    t = t0

    for i in range(len(us)):
      dt = self.dts[i]
      diff = (xs[i][:, None] - self.ref(t))
      
      c += dt * diff.T @ self.cost.Q @ diff
      c += dt * us[i][:, None].T @ self.cost.R @ us[i][:, None]
      
      c += dt * self.b(xs[i], us[i], 0.1, 2)
      
      t += dt
      
    return c

  def fwd_pass(self, xs, us, ks, Ks, t0, alpha = 0.5):
    # rollout of trajectory with control computed by backward pass
    xs_new = [xs[0]]
    us_new = []
    for i, u in enumerate(us):
      # print('u', u[:, None])
      # print('k', ks[i])
      
      for k in ks[i]:
        if np.isnan(k):
          raise "A"

      us_new.append(u[:, None] + alpha * ks[i])

    print('rolling out with control')

    for i in range(len(self.dts)):
      dt = self.dts[i]

      diff = xs_new[i] - xs[i]
      us_new[i] += (Ks[i] @ diff[:, None])
      us_new[i] = us_new[i].flatten()

      xn = self.sys.step(xs_new[i], us_new[i], dt, "euler")
      xs_new.append(xn)

    cost = self.cost_w_penalty(xs_new, us_new, t0)

    return xs_new, us_new, cost

  def quadratized_penalty(self, x, u, q1=0.1, q2=2):
    q = self.bx(x, u, q1, q2)
    Q = self.bxx(x, u, q1, q2)

    r = self.bu(x, u, q1, q2)
    R = self.buu(x, u, q1, q2)
    
    # print('Q\n', Q)
    # print('q\n', q)

    return Q, q, R, r
  
  def bwd_pass(self, xs, us, t0):
    # compute affine control matrix with riccatti equation
    ks = [None] * len(self.dts)
    Ks = [None] * len(self.dts)

    sk = jnp.zeros(self.state_dim)[:, None]
    Sk = self.cost.QN

    t = t0 + sum(self.dts)

    for i in range(len(self.dts)):
      x_idx = len(self.dts) - i - 1
      u_idx = len(self.dts) - i - 1
      dt = self.dts[u_idx]

      t -= dt

      prev_state = xs[x_idx][:, None]
      prev_input = us[u_idx][:, None]
      A, B, F = self.linearization(prev_state.flatten(), prev_input.flatten(), dt)

      # print('x prev\n', prev_state)
      # print('u_prev\n', prev_input)

      bQ, bq, bR, br = self.quadratized_penalty(prev_state.flatten(), prev_input.flatten())

      Q = self.cost.Q + bQ
      R = self.cost.R + bR

      Q = dt * Q
      R = dt * R

      # print(Q)
      # print(R)

      # print(bQ)
      # print(bR)

      lin_ref_tracking = -self.cost.Q @ self.ref(t)

      lx = Q @ prev_state + dt * lin_ref_tracking + dt * bq[:, None]
      lu = R @ prev_input + dt * br[:, None]

      # print('lx\n', lx)
      # print('lu\n', lu)

      lxx = Q
      luu = R
      lux = np.zeros((self.input_dim, self.state_dim))

      Qx = lx + A.T @ sk
      Qu = lu + B.T @ sk
      Qxx = lxx + A.T @ Sk @ A
      Quu = luu + B.T @ Sk @ B
      Qux = lux + B.T @ Sk @ A

      Quu_inv = jnp.linalg.inv(Quu)

      # print('Quu\n', Quu)
      # print('Quu_inv\n', Quu_inv)
      # print('Qu\n', Qu)

      d = -Quu_inv @ Qu
      K = -Quu_inv @ Qux

      ks[u_idx] = d 
      Ks[u_idx] = K

      # print('d\n', d)
      # print('K\n', K)

      sk = Qx + K.T @ Quu @ d + K.T @ Qu + Qux.T @ d
      Sk = Qxx + K.T @ Quu @ K  + K.T @ Qux + Qux.T @ K

    return ks, Ks

  def rollout(self, x0, us, t0):
    xs = [x0]

    for i in range(len(self.dts)):
      dt = self.dts[i]
      xs.append(self.sys.step(xs[i], us[i], dt, "euler"))

    cost = self.cost_w_penalty(xs, us, t0)
    return xs, cost

  def compute(self, state, t):
    # shift solutions
    self.compute_initial_control_guess()

    us = self.prev_u    
    xs, cost = self.rollout(state, us, t)

    for i in range(self.max_iters):
      print('iter', i)

      # print('backward pass')
      ks, Ks = self.bwd_pass(xs, us, t)

      # print('fwd pass')
      for alpha in [0.5**i for i in range(3)]:
        xs_new, us_new, cost_new = self.fwd_pass(xs, us, ks, Ks, t, alpha)

        # print(xs_new)

        print('costs', cost, cost_new)

        if cost_new < cost:
          xs = xs_new
          us = us_new

          # if (abs(cost_new - cost)/cost) < self.converge_thresh:
            # break

          cost = cost_new
          break

    self.first_run = False

    return us[0]

class ALiLQR(Controller):
  pass