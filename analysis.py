import numpy as np

from collections import namedtuple
import time

from controller import *
from controller_util import *

ControllerResult = namedtuple('Result', ['times', 'states', 'inputs', 'solver_time', 'computation_time', 'state_predictions', 'control_predictions', 'progress'])

def eval(system, controller, x0, T_sim, dt_sim, N_sim_iter=10, mpcc=False, noise_on_obs=False, noise_on_input=False):
  xn = x0
  t0 = 0.
    
  states = [x0]
  inputs = []

  progress = []

  state_pred = []
  control_pred = []

  times = [0]
  computation_times_ms = []
  solve_times = []

  fwd = jax.jit(lambda x, u, dt: system.step(x, u, dt, method="rk4"))

  for j in range(int(T_sim / dt_sim)):
    # print("x0")
    # print(xn)
    # print(system.step(xn, np.zeros(system.input_dim), dt_sim, method='heun'))
    t = t0 + j * dt_sim

    if np.isnan(xn).any():
      solve_times[-1] = 1e10
      break

    if noise_on_obs:
      xn += (np.random.rand(xn.shape[0]) - 0.5) * 0.1

    start = time.process_time_ns()
    u = controller.compute(xn, t)
    end = time.process_time_ns()

    if np.isnan(u).any():
      solve_times[-1] = 1e10
      break

    if noise_on_input:
      u += (np.random.rand(u.shape[0]) - 0.5) * 0.1

    # finer simulation
    for i in range(N_sim_iter):
      xn = fwd(xn, u, dt_sim/N_sim_iter)

      #xn += np.random.randn(4) * 0.0001

      states.append(xn)
      inputs.append(u)

      times.append(times[-1] + dt_sim / N_sim_iter)

    computation_times_ms.append((end - start) / 1e6)
    solve_times.append(controller.solve_time) 

    state_pred.append(controller.prev_x)
    control_pred.append(controller.prev_u)

    if mpcc:
      progress.append(controller.prev_p[0, 0])

  return ControllerResult(times, states, inputs, solve_times, computation_times_ms, state_pred, control_pred, progress)

class Problem:
  def __init__(self, T_sim, dt_sim, sys, x0, cost, ref, state_scaling=None, input_scaling=None):
    self.T_sim = T_sim
    self.dt_sim = dt_sim
    self.sys = sys
    
    self.x0 = x0

    self.cost = cost
    self.ref = ref

    self.state_scaling = state_scaling
    self.input_scaling = input_scaling

def make_cost_computation_curve(problem, ks, T_pred=1, noise=False):
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

  T_sim = problem.T_sim
  sys = problem.sys
  dt = problem.dt_sim

  x0 = problem.x0
  ref = problem.ref

  quadratic_cost = problem.cost

  state_scaling = problem.state_scaling
  input_scaling = problem.input_scaling

  data = {}

  for T in [T_pred]:
    for k in ks:
      # nmpc = NMPC(sys, k, T / k, quadratic_cost, ref)
      nmpc = ParameterizedNMPC(sys, k, T / k, quadratic_cost, ref)
      nmpc.state_scaling = state_scaling
      nmpc.input_scaling = input_scaling
      nmpc.setup_problem()

      lin_dts = get_linear_spacing(dt, T, k)
      # nu_mpc_lin = NU_NMPC(sys, lin_dts, quadratic_cost, ref)
      nu_mpc_lin = Parameterized_NU_NMPC(sys, lin_dts, quadratic_cost, ref)
      nu_mpc_lin.nmpc.state_scaling = state_scaling
      nu_mpc_lin.nmpc.input_scaling = input_scaling
      nu_mpc_lin.setup_problem()

      # exp_dts = get_power_spacing(dt, T, k)
      # nu_mpc_exp = NU_NMPC(sys, exp_dts, quadratic_cost, ref)

      # nu_mpc_exp.nmpc.state_scaling = state_scaling
      # nu_mpc_exp.nmpc.input_scaling = input_scaling

      solvers = [
        ("NMPC", nmpc), 
        ("NU_MPC_linear", nu_mpc_lin), 
        # ("NU_MPC_exp", nu_mpc_exp)
      ]

      for j, (name, solver) in enumerate(solvers):
        res = eval(sys, solver, x0, T_sim, dt, noise_on_obs=noise, noise_on_input=noise)
        # mpc_sol = eval(sys, nmpc, x0, T_sim, dt)
        # nu_mpc_lin_sol = eval(sys, nu_mpc_lin, x, T_sim, dt)
        # nu_mpc_exp_sol = eval(sys, nu_mpc_exp, x, T_sim, dt)

        # for j, res in enumerate([mpc_sol, nu_mpc_lin_sol, nu_mpc_exp_sol]):
        times = res.times
        states = res.states
        inputs = res.inputs
        computation_times = res.computation_time
        solve_times = res.solver_time

        cost = 0
        for i in range(len(states)-1):
          diff = (states[i][:, None] - ref)
          u = inputs[i][:, None]
          cost += dt * (diff.T @ quadratic_cost.Q @ diff + u.T @ quadratic_cost.R @ u)

        print(cost)
        print(sum(computation_times))

        solver_data = [sum(computation_times), sum(solve_times), cost[0][0], k, res]
        data.setdefault(name, []).append(solver_data)

        # if j==0:
        #   plt.scatter(sum(solve_times), cost, marker='o', color="tab:blue")
        # elif j==1:
        #   plt.scatter(sum(solve_times), cost, marker='s', color="tab:orange")
        # elif j==2:
        #   plt.scatter(sum(solve_times), cost, marker='s', color="tab:green")

  # for solver_name, value in data.items():
  #   x = [value[i][1] for i in range(len(value))]
  #   y = [value[i][2] for i in range(len(value))]

  #   plt.plot(x, y, label=solver_name)

  # plt.xlabel("Solver time [ms]")
  # plt.ylabel("Cost")
  # plt.legend()

  # plt.show()

  return data