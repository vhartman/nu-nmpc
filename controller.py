import numpy as np

import cvxpy as cp
from scipy.interpolate import interp1d

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
