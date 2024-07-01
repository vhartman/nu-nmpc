import numpy as np
import jax

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
      for _ in range(self.N):
        self.prev_u.append(np.zeros(self.input_dim))

      self.prev_u = np.array(self.prev_u).T
    else:
      # shift solution by one
      # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      f = interp1d(times, self.prev_u, axis=1, bounds_error=False, fill_value=self.prev_u[:, -1])
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

    self.compute_initial_state_guess((S @ state[:, None]).flatten())
    self.compute_initial_control_guess()

    if not self.first_run:
      print(self.prev_x[:, 0])
    
    print("state")
    print(self.prev_x)

    self.solve_time = 0

    # dynamics constraints
    for k in range(self.sqp_iters):
      constraints = []
      for i in range(self.N):
        idx = i
        next_idx = i+1

        dt = self.dts[i]
        print(i, dt, t)

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
          cost += dt * 10 * cp.sum(s[:, (i+1)*2])
          cost += dt * 10 * cp.sum(s[:, (i+1)*2+1])

          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2])
          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2+1])

      objective = cp.Minimize(cost)

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      # The optimal objective value is returned by `prob.solve()`.
      result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
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
      # TODO: should really be a line search
      if self.first_run:
        self.prev_x = x.value
        self.prev_u = u.value
      else:
        self.prev_x = x.value * self.sqp_mixing + self.prev_x * (1 - self.sqp_mixing)
        self.prev_u = u.value * self.sqp_mixing + self.prev_u * (1 - self.sqp_mixing)

      self.first_run = False

      self.solve_time += prob.solver_stats.solve_time

      # data, _, _= prob.get_problem_data(cp.OSQP)

      # print(data)

      # print("A")
      # print(data['A'].todense())
      # np.savetxt("foo.csv", data["A"].todense(), delimiter=",")

      # print("B")
      # print(data['b'])

    if prob.status not in ["infeasible", "unbounded"]:
      print("U")
      print(u.value[:, 0])
      print( Winv @ u.value[:, 0])
      return Winv @ u.value[:, 0]
      # return u.value[:, 0]

    return np.zeros(self.sys.input_dim)
  
#def get_relu_spacing(dt0, dt_max, T, steps):

# def get_linear_spacing_with_max_dt(T, dt0, dt_max, steps):
#   def find_k_and_alpha(T, dt_0, dt_max, n, initial_alpha_guess=1.0, tolerance=1e-6, max_iterations=1000):
#     def calculate_k(alpha):
#       return int(dt_max // alpha)

#     def calculate_alpha(k):
#       return (2 * (T - n * dt_0 - (n - k - 1) * dt_max)) / (k * (k + 1))

#     alpha = initial_alpha_guess
#     for iteration in range(max_iterations):
#       k = calculate_k(alpha)
#       new_alpha = calculate_alpha(k)
      
#       print(k , alpha, new_alpha)

#       if abs(new_alpha - alpha) < tolerance:
#         return k, new_alpha

#       alpha = new_alpha

#     raise ValueError("Convergence not achieved within the maximum number of iterations")
  
#   assert(dt_max >= dt0)

#   _, alpha = find_k_and_alpha(T, dt0, (dt_max - dt0), steps, 0.01)
#   return [min(dt0 + i * alpha, dt_max) for i in range(steps)]

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

    self.N = len(dts)

    self.sys = system
    self.dts = dts

    self.H = H

    # make spline approximation of reference path
    if callable(reference):
      self.ref = reference
    else:
      self.ref = lambda t: reference

    ts = np.linspace(0, 4, 100)
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
      for _ in range(self.N):
        self.prev_u.append(np.zeros(self.input_dim))

      self.prev_u = np.array(self.prev_u).T
    else:
      # shift solution by one
      # self.prev_u = np.hstack((self.prev_u[:, 1:], self.prev_u[:, -1][:, None]))

      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      f = interp1d(times, self.prev_u, axis=1, bounds_error=False, fill_value=self.prev_u[:, -1])
      new_u = f(times + self.dts[0])

      self.prev_u = new_u

  def project_state_to_progess(self, state):
    min_error = 1e6
    theta_best = 0
    for t in np.linspace(0, 4, 1000):
      e = self.error(state, t)
      if e < min_error:
        min_error = e
        theta_best = t

    return theta_best

  def compute_initial_progress_guess(self, state):
    theta = self.project_state_to_progess(state)

    if self.first_run:
      for i in range(self.N+1):
        self.prev_p.append(np.array([theta + i*0.01, 0.01]))

      self.prev_p = np.array(self.prev_p).T
    else:
      times = np.array([0] + list(np.cumsum(self.dts)))

      f = interp1d(times, self.prev_p, axis=1, bounds_error=False, fill_value=self.prev_p[:, -1])
      new_p = f(times + self.dts[0])
      
      self.prev_p = new_p
      self.prev_p[:, 0] = theta
    
  def compute_initial_progress_input_guess(self):
    if self.first_run:
      for i in range(self.N):
        self.prev_up.append(np.ones(1)*0.1)

      self.prev_up = np.array(self.prev_up).T
    else:
      times = np.array([0] + list(np.cumsum(self.dts[:-1])))

      print(times)
      print(len(times))
      print(self.prev_up)
      print(len(self.prev_up))
      f = interp1d(times, self.prev_up, axis=1, bounds_error=False, fill_value=self.prev_up[:, -1])
      new_up = f(times + self.dts[0])
      
      self.prev_up = new_up

  def error(self, q, theta):
    residual = self.path(theta) - self.H @ q[:, None]
    return residual.T @ residual

  def error_quad_approx(self, q, theta):
    residual = self.path(theta) - self.H @ q[:, None]

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
    grad_theta = 2 * (self.path_derivative(theta).T @ residual)
    grad_q = -2 * (self.H.T @ residual)

    # Compute the Hessian
    hess_theta = 2 * (path_jac.T @ path_jac + path_hess.T @ residual)
    # hess_theta = 2 * (path_jac.T @ path_jac)
    hess_q = 2 * (self.H.T @ self.H)
    hess_theta_q = -2 * (path_jac.T @ self.H)

    print("other part", path_jac.T @ path_jac)
    print("residual thingy:", path_hess.T @ residual)

    return hess_q, hess_theta, hess_theta_q, grad_q, grad_theta

  def compute(self, state, t):
    x = cp.Variable((self.state_dim, self.N+1))
    u = cp.Variable((self.input_dim, self.N))

    s = cp.Variable((self.state_dim, (self.N+1)*2))

    p = cp.Variable((2, self.N+1))
    up = cp.Variable((1, self.N))

    # project state to path to obtain progress var

    S = np.diag(self.state_scaling)
    Sinv = np.diag(1 / self.state_scaling)

    W = np.diag(self.input_scaling)
    Winv = np.diag(1 / self.input_scaling)

    self.compute_initial_state_guess((S @ state[:, None]).flatten())
    self.compute_initial_control_guess()

    self.compute_initial_progress_guess(state)
    self.compute_initial_progress_input_guess()

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
        print(i, dt, t)

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

        prev_progress = self.prev_p[:, i]
        prev_input = self.prev_up[:, i]

        A, B, K = self.progress_linearization(prev_progress, prev_input, dt)

        constraints.append(
          p[:, next_idx][:, None] == A @ p[:, idx][:, None] + B @ up[:, idx][:, None] - K
        )

      # input constraints
      for i in range(self.N):
        constraints.append(u[:, i][:, None] >= W @ self.sys.input_limits[:, 0][:, None])
        constraints.append(u[:, i][:, None] <= W @ self.sys.input_limits[:, 1][:, None])

      # progress constraints
      for i in range(self.N):
        constraints.append(up[0, i] >= -10)
        constraints.append(up[0, i] <= 10)
    
      for i in range(self.N+1):
        constraints.append(p[1, i] >= 0)
        constraints.append(p[1, i] <= 2)

        constraints.append(p[0, i] >= 0)
        constraints.append(p[0, i] <= 3)

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

      cost = 0

      # TODO: normalize cost by total prediction horizn?
      Rs = 0.000001 * Winv @ np.eye(self.input_dim) @ Winv

      curr_time = t
      for i in range(self.N): 
        dt = self.dts[i]
        cost += dt * cp.quad_form(u[:,i][:, None], Rs)

        curr_time += dt

      for i in range(1, self.N+1):
        dt = self.dts[i-1]

        # quadratize error
        prev_progress = self.prev_p[:, i]
        prev_state = (Sinv @ self.prev_x[:, i][:, None]).flatten()

        print("Step", i)
        hess_q, hess_theta, hess_theta_q, grad_q, grad_theta = self.error_quad_approx(prev_state, prev_progress[0])

        # print(hess_q)
        # print(hess_theta)
        # print(hess_theta_q)
        # print(grad_q)
        # print(grad_theta)

        hess = np.block([[hess_q, hess_theta_q.T],
                         [hess_theta_q, hess_theta]])
        
        hess  = 0.5 * hess + 0.5 * hess.T #+ np.eye(hess.shape[0]) * 0.001
        
        p_reshaped = cp.reshape(p[0, i], (1, 1))
        x_reshaped = cp.reshape(Sinv @ x[:, i][:, None], (self.state_dim, 1))
        tmp = cp.vstack([x_reshaped, p_reshaped])

        offset = np.vstack((prev_state[:, None], prev_progress[0]))

        w = 50
        cost += dt * w * 0.5 * cp.quad_form(tmp - offset, hess)

        # np.set_printoptions(precision=3)
        # print(hess)

        # def is_pos_def(x):
        #   print("EV:", np.linalg.eigvals(x))
        #   return np.all(np.linalg.eigvals(x) > 0)
        
        # print(is_pos_def(hess))

        # cost += 0.5 * cp.quad_form(x[:, i] [:, None], hess_q)
        # cost += 0.5 * hess_theta * p[0, i]**2
        # cost += p[0, i] * hess_theta_q @ x[:, i] [:, None]

        cost += dt * w * grad_q.T @ Sinv @ x[:, i][:, None]
        cost += dt * w * grad_theta.flatten() * p[0, i]

        cost -= dt * 0.1 * p[1, i]

      # cost = cost * 10
      # cost = cost / 1000
      cost = cost * 10
      # cost = cost / 10

      # slack variables
      if self.constraints_with_slack:
        for i in range(self.N):
          dt = self.dts[i]
          cost += dt * 10 * cp.sum(s[:, (i+1)*2])
          cost += dt * 10 * cp.sum(s[:, (i+1)*2+1])

          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2])
          cost += dt * 500 * cp.sum_squares(s[:, (i+1)*2+1])

      objective = cp.Minimize(cost)

      prob = cp.Problem(objective, constraints)

      # warm start
      x.value = self.prev_x
      u.value = self.prev_u

      # The optimal objective value is returned by `prob.solve()`.
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-4, eps_rel=1e-8, max_iter=100000, scaling=False, verbose=True, polish_refine_iter=10)
      # result = prob.solve(solver='OSQP', verbose=True,eps_abs=1e-7, eps_rel=1e-5, max_iter=10000)
      # result = prob.solve(solver='ECOS', verbose=True, max_iters=1000, feastol=1e-5, reltol=1e-4, abstol_inacc=1e-5, reltol_inacc=1e-5, feastol_inacc=1e-5)
      result = prob.solve(solver='SCS', verbose=True, eps=1e-8)
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

      self.solve_time += prob.solver_stats.solve_time

      # data, _, _= prob.get_problem_data(cp.OSQP)

      # print(data)

      # print("A")
      # print(data['A'].todense())
      # np.savetxt("foo.csv", data["A"].todense(), delimiter=",")

      # print("B")
      # print(data['b'])

    if prob.status not in ["infeasible", "unbounded"]:
      print("U")
      print(u.value[:, 0])
      print( Winv @ u.value[:, 0])
      return Winv @ u.value[:, 0]
      # return u.value[:, 0]

    return np.zeros(self.sys.input_dim)
  