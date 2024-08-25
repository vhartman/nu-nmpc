import numpy as np
import jax

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib import cm

import matplotlib.animation as animation

from systems import *
from controller import *

from analysis import *

from util import *
from controller_util import *

plt.style.use('./blog.mplstyle')

def motivation_horizon():
  T_sim = 3
  dt = 0.025

  sys = JerkMasspointND(1)
  x0 = np.zeros(3)
  x0[0] = 4
  x0[1] = 0
  x0[2] = 0

  Q = np.diag([10, 0.01, 0.01])
  ref = np.zeros((3, 1))
  R = np.diag([.0001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  for N_steps in [5, 10, 20, 40, 60]:
    # nmpc = NMPC(sys, N_steps, dt, quadratic_cost, ref)
    nmpc = ParameterizedNMPC(sys, N_steps, dt, quadratic_cost, ref)
    nmpc.sqp_iters = 1
    nmpc.sqp_mixing = 1

    state_scaling = 1 / np.array([1, 1, 1])
    input_scaling = 1 / np.array([20])

    nmpc.state_scaling = state_scaling
    nmpc.input_scaling = input_scaling

    mpc_sol = eval(sys, nmpc, x0, T_sim, dt)

    for res in [mpc_sol]:
      times = res.times
      states = res.states
      inputs = res.inputs
      computation_times = res.computation_time

      x = [states[i][0] for i in range(len(times))]
      v = [states[i][1] for i in range(len(times))]
      a = [states[i][2] for i in range(len(times))]
      u = [inputs[i][0] for i in range(len(times)-1)]

      plt.figure(f"N={N_steps}")
      plt.plot(times, x)
      plt.plot(times, v)
      plt.plot(times, a)

      plt.plot(times, [sys.state_limits[0, 0]]*len(times), '--', c='tab:blue')
      plt.plot(times, [sys.state_limits[1, 0]]*len(times), '--', c='tab:orange')
      plt.plot(times, [sys.state_limits[2, 0]]*len(times), '--', c='tab:green')

      plt.plot(times, [sys.state_limits[0, 1]]*len(times), '--', c='tab:blue')
      plt.plot(times, [sys.state_limits[1, 1]]*len(times), '--', c='tab:orange')
      plt.plot(times, [sys.state_limits[2, 1]]*len(times), '--', c='tab:green')

      plt.title(f"T_p={N_steps*dt}")
      plt.xlabel('T')

      plt.savefig(f'./img/blog/lin_sys_n_{N_steps}.png', format='png', dpi=300, bbox_inches = 'tight')

      plt.figure(f"input for N={N_steps}")
      plt.plot(times[:-1], u)

      plt.title(f"T_p={N_steps*dt}")
      plt.xlabel('T')

      plt.savefig(f'./img/blog/input_lin_sys_n_{N_steps}.png', format='png', dpi=300, bbox_inches = 'tight')

  plt.show()

def motivation_discretization():
  T_sim = 3
  dt_sim = 0.02

  sys = JerkMasspointND(1)
  x0 = np.zeros(3)
  x0[0] = 4
  x0[1] = 0
  x0[2] = 0

  Q = np.diag([10, 0.01, 0.01])
  ref = np.zeros((3, 1))
  R = np.diag([0.00001])

  quadratic_cost = QuadraticCost(Q, R, Q)

  discretizations = [5, 10, 20, 30, 40]
  # discretizations = [5, 10, 15]
  costs = []
  comp_times = []

  T_pred = 40 * dt_sim

  for N_disc in discretizations:
    dt_disc = T_pred / N_disc

    nmpc = NMPC(sys, N_disc, dt_disc, quadratic_cost, ref)
    nmpc.sqp_iters = 1
    nmpc.sqp_mixing = 1

    state_scaling = 1 / np.array([1, 1, 1])
    input_scaling = 1 / np.array([20])

    nmpc.state_scaling = state_scaling
    nmpc.input_scaling = input_scaling

    mpc_sol = eval(sys, nmpc, x0, T_sim, dt_sim)

    for res in [mpc_sol]:
      times = res.times
      states = res.states
      inputs = res.inputs
      computation_times = res.computation_time
      solve_times = res.solver_time

      # comp_times.append(sum(computation_times))
      comp_times.append(sum(solve_times) * 1000)

      cost = 0
      for i in range(len(states)-1):
        diff = (states[i][:, None] - ref)
        u = inputs[i][:, None]
        cost += dt_sim * (diff.T @ Q @ diff + u.T @ R @ u)

      costs.append(cost[0])

      x = [states[i][0] for i in range(len(times))]
      v = [states[i][1] for i in range(len(times))]
      a = [states[i][2] for i in range(len(times))]
      u = [inputs[i][0] for i in range(len(times)-1)]

      plt.figure(f"dt={dt_disc}")
      plt.plot(times, x)
      plt.plot(times, v)
      plt.plot(times, a)

      plt.plot(times, [sys.state_limits[0, 0]]*len(times), '--', c='tab:blue')
      plt.plot(times, [sys.state_limits[1, 0]]*len(times), '--', c='tab:orange')
      plt.plot(times, [sys.state_limits[2, 0]]*len(times), '--', c='tab:green')

      plt.plot(times, [sys.state_limits[0, 1]]*len(times), '--', c='tab:blue')
      plt.plot(times, [sys.state_limits[1, 1]]*len(times), '--', c='tab:orange')
      plt.plot(times, [sys.state_limits[2, 1]]*len(times), '--', c='tab:green')

      print(cost)

  plt.figure("Costs")
  plt.plot(discretizations, costs)
  plt.xlabel("N")
  plt.ylabel("Cost")

  plt.savefig(f'./img/blog/lin_sys_cost.png', format='png', dpi=300, bbox_inches = 'tight')

  plt.figure("Computation times")
  plt.plot(discretizations, np.array(comp_times))
  plt.xlabel("N")
  plt.ylabel("Solver time [ms]")

  plt.savefig(f'./img/blog/lin_sys_comp_time.png', format='png', dpi=300, bbox_inches = 'tight')

  plt.show()

def make_cart_pole_animation():
  plt.style.use('default')

  T_sim = 4
  sys = Cartpole()
  x = np.zeros(4)
  x[2] = 0
  u = np.zeros(1)
  dt = 0.05

  Q = np.diag([5, 0.1, 50, 0.1])
  ref = np.zeros((4, 1))
  ref[2, 0] = np.pi

  R = np.diag([.01])

  quadratic_cost = QuadraticCost(Q, R, Q)

  dts = get_linear_spacing(0.05, 1, 20)
  # nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)
  nu_mpc = Parameterized_NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    plt.figure()
    plt.plot([i[0] for i in inputs])

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], '-')

    rw = 0.1
    rh = 0.1
    rect = patches.Rectangle((-rw/2, -rh/2), rw, rh, fill=None, edgecolor='black', rotation_point='center')
    ax.add_patch(rect)
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
      x = states[i][0]
      angle = states[i][2]

      x_line = [x, x + 0.5 * np.sin(angle)]
      y_line = [0, -0.5 * np.cos(angle)]

      line.set_data(x_line, y_line)

      t = (transforms.Affine2D()
          .translate(x, 0))
      # Apply the transformation to the polygon
      rect.set_transform(t + ax.transData)

      time_text.set_text(time_template % (times[i]))

      return line, time_text
    
    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)
    
    ani = animation.FuncAnimation(
      fig, animate, np.arange(0, len(states), 4), interval=dt*1000/10*4, blit=True)

    ani.save('cartpole_anim.gif', writer='pillow', fps=1./(dt/10.)/4)

  plt.show()

def make_double_cart_pole_animation():
  plt.style.use('default')

  T_sim = 4

  sys = DoubleCartpole()
  x0 = np.zeros(6)
  x0[1] = np.pi
  x0[2] = 0
  dt = 0.025

  Q = np.diag([1, 2, 2, 0.1, 0.1, 0.1])
  R = np.diag([.001])

  # Q = np.diag([1, 4, 4, 0.01, 0.01, 0.01])
  # R = np.diag([.05])

  ref = np.zeros((6, 1))

  state_scaling = 1 / (np.array([2, 5, 2, 10, 10, 10]))
  input_scaling = 1 / (np.array([50]))
  
  quadratic_cost = QuadraticCost(Q, R, Q)

  dts = get_linear_spacing(0.025, 1, 20)
  # dts = [0.05] * 20 # 205
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)
  
  nu_mpc.state_scaling = state_scaling
  nu_mpc.input_scaling = input_scaling

  nu_mpc_sol = eval(sys, nu_mpc, x0, T_sim, dt)

  for res in [nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time
  
    plt.figure()
    plt.plot([i[0] for i in inputs])

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-1.2, 1.2))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], '-')

    rw = 0.1
    rh = 0.1
    rect = patches.Rectangle((-rw/2, -rh/2), rw, rh, fill=None, edgecolor='black', rotation_point='center')
    ax.add_patch(rect)
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
      xpos = states[i][0]

      a1 = states[i][1]
      a2 = states[i][2]

      l1 = 0.5
      l2 = 0.5
      
      x_pts = [xpos, xpos + l1 * np.sin(a1), xpos + l1 * np.sin(a1) + l2 * np.sin(a1 + a2)]
      y_pts = [0, l1 * np.cos(a1), l1*np.cos(a1) + l2*np.cos(a1 + a2)]

      line.set_data(x_pts, y_pts)

      t = (transforms.Affine2D()
          .translate(xpos, 0))
      # Apply the transformation to the polygon
      rect.set_transform(t + ax.transData)

      time_text.set_text(time_template % (times[i]))

      return line, time_text
    
    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)
    
    ani = animation.FuncAnimation(
      fig, animate, np.arange(0, len(states), 4), interval=dt*1000/10, blit=True)

    ani.save('double_cartpole_anim.gif', writer='pillow', fps=1./(dt/10.)/4)

  plt.show()

def make_quadcopter_animation():
  plt.style.use('default')

  T_sim = 3

  sys = Quadcopter2D()

  Q = np.diag([10, 10, 50, 0.01, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([0.001, 0.001])

  x = np.zeros(6)
  x[0] = 2
  x[1] = 1
  x[2] = np.pi
  x[3] = -2
  x[4] = 4

  dt = 0.05

  quadratic_cost = QuadraticCost(Q, R, Q)

  state_scaling = 1 / np.array([1., 1, 1, 5, 5, 10])
  input_scaling = 1 / np.array([2, 2])

  dts = get_linear_spacing(0.05, 1, 10)
  # N = 10
  # dts = [1/N] * N
  nu_mpc = NU_NMPC(sys, dts, quadratic_cost, ref)

  nu_mpc.nmpc.state_scaling = state_scaling
  nu_mpc.nmpc.input_scaling = input_scaling

  nu_mpc_sol = eval(sys, nu_mpc, x, T_sim, dt)

  for res in [nu_mpc_sol]:
    times = res.times
    states = res.states
    inputs = res.inputs
    computation_times = res.computation_time

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-0.5, 2.5), ylim=(-1, 2))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], '-')
    scat = ax.scatter([0], [0])

    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    l = 0.2
    def animate(i):
      xpos = states[i][0]
      ypos = states[i][1]
      angle = states[i][2]
        
      x_line = [xpos - l*np.cos(angle), xpos + l*np.cos(angle)]
      y_line = [ypos - l*np.sin(angle), ypos + l*np.sin(angle)]
      line.set_data(x_line, y_line)

      scat.set_offsets(np.array([xpos, ypos + 0.05*np.cos(states[i][2])]))

      time_text.set_text(time_template % (times[i]))

      return line, time_text
    
    cost = 0
    for i in range(len(states)-1):
      diff = (states[i][:, None] - ref)
      u = inputs[i][:, None]
      cost += dt * (diff.T @ Q @ diff + u.T @ R @ u)

    print(cost)

    ani = animation.FuncAnimation(
      fig, animate, np.arange(0, len(states), 4), interval=dt*1000/10*4, blit=True)

    ani.save('quadcopter_animation.gif', writer='pillow', fps=1./(dt/10.)/4)

  
  plt.show()

def make_racecar_animation():
  pass

solver_to_color_map = {
  "NMPC": "tab:blue",
  "NU_MPC_linear": "tab:orange",
  "NU_MPC_exp": "tab:green",
}

def plot_data_from_cost_comp(all_data, system_name, plot_normalized_data=True, save=True):
  solvers = set(k for d in all_data for k in d)

  plt.figure()
  for solver_name in solvers:
    times = []
    costs = []

    for data in all_data:
      value = data[solver_name]
      x = [value[i][1] for i in range(len(value))]
      y = [value[i][2] for i in range(len(value))]
      y_normalized = [value[i][-1] for i in range(len(value))]
      
      times.append(x)

      if plot_normalized_data:
        costs.append(y_normalized)
      else:
        costs.append(y)

    median_times = np.median(times, axis=0)
    median_costs = np.median(costs, axis=0)

    plt.errorbar(median_times, median_costs,
                 xerr=[median_times - np.min(times, axis=0), np.max(times, axis=0) - median_times], 
                 yerr=[median_costs - np.min(costs, axis=0), np.max(costs, axis=0) - median_costs],
                 label=solver_name, color=solver_to_color_map[solver_name],
                 marker='o')

  plt.xlabel("Solver time [s]")
  plt.ylabel("Cost")
  plt.legend()

  if save:
    plt.savefig(f'./img/blog/{system_name}_cost_comp.png', format='png', dpi=300, bbox_inches = 'tight')

  plt.figure()
  for solver_name, value in data.items():
    x = [value[i][1] for i in range(len(value))]
    y = [value[i][2] for i in range(len(value))]

    ks = [value[i][3] for i in range(len(value))]

    plt.plot(ks, x, label=solver_name)

  if save:
    plt.savefig(f'./img/blog/{system_name}_compute_times.png', format='png', dpi=300, bbox_inches = 'tight')

def make_masspoint_cost_comp_for_blog(num_runs=1, randomize_start_state=False):
  T_sim = 3
  dt_sim = 0.02

  sys = JerkMasspointND(1)
  x0 = np.zeros(3)
  x0[0] = 4
  x0[1] = 0
  x0[2] = 0

  Q = np.diag([10, 0.01, 0.01])
  ref = np.zeros((3, 1))
  R = np.diag([.001])

  state_scaling = 1 / np.array([1, 1, 1])
  input_scaling = 1 / np.array([20])

  p = Problem(T_sim, dt_sim, sys, x0, QuadraticCost(Q, R, Q), ref, state_scaling, input_scaling)

  solvers = [
    "NMPC",
    "NU_MPC_linear",
    "NU_MPC_exp"
  ]

  all_data = []
  for i in range(num_runs):
    print(i)

    if randomize_start_state:
      np.random.seed(i)
      p.x0 = x0 + np.random.rand(3) * 0.1

    # data = make_cost_computation_curve(p, ks = [5, 7, 10, 20, 30, 40], T_pred= 40 * dt_sim, noise=True)
    data = make_cost_computation_curve(p, solvers, ks = [5, 10, 20, 30], T_pred= 40 * dt_sim, noise=False)
    
    normalizing_cost = data["NMPC"][-1][2]
    normalized_data = data
    
    for solver in solvers:
      num_pts = len(normalized_data[solver])
      for i in range(num_pts):
        normalized_data[solver][i].append(normalized_data[solver][i][2] / normalizing_cost)
    
    all_data.append(normalized_data)

  plot_data_from_cost_comp(all_data, "masspoint")

def make_cartpole_cost_comp_for_blog(num_runs=1, randomize_start_state = False):
  T_sim = 4
  sys = Cartpole()
  x0 = np.zeros(4)
  x0[2] = 0
  dt = 0.05

  Q = np.diag([5, 0.1, 50, 0.1])
  ref = np.zeros((4, 1))
  ref[2, 0] = np.pi

  R = np.diag([.001])

  state_scaling = 1 / np.array([1, 1, 1, 1])
  input_scaling = 1 / np.array([1])

  p = Problem(T_sim, dt, sys, x0, QuadraticCost(Q, R, Q), ref, state_scaling, input_scaling)

  solvers = [
    "NMPC",
    "NU_MPC_linear",
    "NU_MPC_exp"
  ]

  all_data = []
  for i in range(num_runs):
    print(i)

    if randomize_start_state:
      np.random.seed(i)
      p.x0 = x0 + np.random.rand(sys.state_dim) * 0.5
    
    data = make_cost_computation_curve(p, solvers, ks = [5, 7, 10, 12, 15, 20], T_pred=1)
    # data = make_cost_computation_curve(p, ks = [5, 7], T_pred=1)
    # data = make_cost_computation_curve(p, ks = [3, 5, 7, 10, 12, 15, 20], T_pred=1)

    normalizing_cost = data["NMPC"][-1][2]
    normalized_data = data
    
    for solver in solvers:
      num_pts = len(normalized_data[solver])
      for i in range(num_pts):
        normalized_data[solver][i].append(normalized_data[solver][i][2] / normalizing_cost)

        final_state = data[solver][i][4].states[-1]
        if not (-np.cos(final_state[2]) > 0.8 and \
                       abs(final_state[3]) < 0.5 and \
                       abs(final_state[0]) < 1):
          normalized_data[solver][i][-1] = 10
          normalized_data[solver][i][2] = 10

    all_data.append(normalized_data)

  plot_data_from_cost_comp(all_data, "cartpole")

def make_double_cartpole_cost_comp_for_blog():
  T_sim = 4

  sys = DoubleCartpole()
  x0 = np.zeros(6)
  x0[1] = np.pi
  x0[2] = 0
  dt = 0.025

  Q = np.diag([1, 2, 2, 0.1, 0.1, 0.1])
  R = np.diag([.001])

  # Q = np.diag([1, 5, 5, 0.01, 0.01, 0.01])
  # R = np.diag([.001])

  # Q = np.diag([1, 4, 4, 0.01, 0.01, 0.01])
  # R = np.diag([.05])

  ref = np.zeros((6, 1))

  state_scaling = 1 / (np.array([2, 5, 2, 10, 10, 10]))
  input_scaling = 1 / (np.array([50]))
  
  p = Problem(T_sim, dt, sys, x0, QuadraticCost(Q, R, Q), ref, state_scaling, input_scaling)
  data = make_cost_computation_curve(p, solvers, ks = [10, 20, 40], T_pred=1)
  # make_cost_computation_curve(p, ks = [5, 7, 10, 12, 15, 20], T_pred=1)

  for solver_name, value in data.items():
    x = [value[i][1] for i in range(len(value))]
    y = [value[i][2] for i in range(len(value))]

    ks = [value[i][3] for i in range(len(value))]

    x_sorted = np.array([x for _, x in sorted(zip(ks, x))])
    y_sorted = np.array([x for _, x in sorted(zip(ks, y))])

    mask = x_sorted < 1e5

    plt.plot(x_sorted[mask], y_sorted[mask], label=solver_name)

  plt.xlabel("Solver time [ms]")
  plt.ylabel("Cost")
  plt.legend()

  plt.savefig(f'./img/blog/double_cartpole_cost_comp.png', format='png', dpi=300, bbox_inches = 'tight')

  plt.figure()
  for solver_name, value in data.items():
    x = [value[i][1] for i in range(len(value))]
    y = [value[i][2] for i in range(len(value))]

    ks = [value[i][3] for i in range(len(value))]

    x_sorted = [x for _, x in sorted(zip(ks, x))]
    y_sorted = [x for _, x in sorted(zip(ks, y))]

    plt.plot(ks, x, label=solver_name)

  plt.savefig(f'./img/blog/double_cartpole_compute_times.png', format='png', dpi=300, bbox_inches = 'tight')

def make_quadcopter_cost_comp_for_blog(num_runs=1, randomize_start_state=False):
  T_sim = 3

  sys = Quadcopter2D()

  # Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01])
  # ref = np.zeros((6, 1))
  # R = np.diag([.0001, .0001])
  
  Q = np.diag([10, 10, 50, 0.01, 0.01, 0.01])
  ref = np.zeros((6, 1))
  R = np.diag([0.0001, 0.0001])

  # x0 = np.zeros(6)
  # x0[0] = 1
  # x0[1] = 1
  # x0[2] = 0
  # x0[3] = 2
  # x0[4] = 4

  x0 = np.zeros(6)
  x0[0] = 2
  x0[1] = 1
  x0[2] = np.pi
  x0[3] = -2
  x0[4] = 4

  dt = 0.05

  state_scaling = 1 / np.array([1., 1, 1, 10, 10, 10])
  input_scaling = 1 / np.array([2, 2])
  
  p = Problem(T_sim, dt, sys, x0, QuadraticCost(Q, R, Q), ref, state_scaling, input_scaling)

  solvers = [
    "NMPC",
    "NU_MPC_linear",
    "NU_MPC_exp"
  ]

  all_data = []
  for i in range(num_runs):
    print(i)
    if randomize_start_state:
      np.random.seed(i)
      p.x0 = x0 + np.random.rand(sys.state_dim) * 0.5

    # data = make_cost_computation_curve(p, ks = [5, 10, 20], T_pred=1)
    data = make_cost_computation_curve(p, solvers, ks = [5, 7, 10, 12, 15, 20], T_pred=1)
    all_data.append(data)

    normalizing_cost = data["NMPC"][-1][2]
    normalized_data = data
    
    for solver in solvers:
      num_pts = len(normalized_data[solver])
      for i in range(num_pts):
        normalized_data[solver][i].append(normalized_data[solver][i][2] / normalizing_cost)
    
    all_data.append(normalized_data)

  plot_data_from_cost_comp(all_data, "quadcopter")

def make_racecar_analysis(num_runs=1, track='fig8'):
  if track=='fig8':
    T_sim = 5
    Ns = [10, 15, 20, 30, 40]
  else:
    T_sim = 9
    Ns = [15, 20, 30, 40]

  dt_sim = 0.025

  T_pred = 1.
  dt_disc = 0.025

  # ref = lambda t: np.array([figure_eight(t)[0], figure_eight(t)[1]]).reshape(-1, 1)
  # ref = lambda t: np.array([square(t)[0], square(t)[1]]).reshape(-1, 1)
  if track=='fig8':
    ref = lambda t: np.array([figure_eight(t, 1.2)[0], figure_eight(t, 1.2)[1]]).reshape(-1, 1)
  else:
    ref = lambda t: np.array([racetrack(t)[0], racetrack(t)[1]]).reshape(-1, 1)

  mapping = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
  ])

  x0 = (mapping.T @ ref(0)).flatten()
  if track=='fig8':
    x0[2] = 0.5 * np.pi # fig 8
  else:
    x0[2] = -np.pi/4 # racetrack

  if track=='fig8':
    goal_progress = 2*np.pi
  else:
    goal_progress = 4

  for name in ['NMPC', 'NU_MPC_linear']:
  # for name in ['NMPC']:
    all_violations_all_runs = []
    all_laptimes_all_runs = []
    all_solvetimes_all_runs = []
    
    for _ in range(num_runs):
      all_violations = []
      all_laptimes = []
      all_solvetimes = []

      for N in Ns:
        if name == 'NMPC':
          dts = [T_pred / N] * N
        else:
          dts = get_linear_spacing(dt_disc, T_pred, N)

        sys = AMZRacecar()

        state_scaling = 1 / (np.array([1, 1, 2*np.pi, 2, 2, 5, 1, 0.35]))
        input_scaling = 1 / (np.array([5, 3]))

        # nmpcc = NMPCC(sys, dts, mapping, ref)
        nmpcc = Parameterized_NMPCC(sys, dts, mapping, ref)
        nmpcc.state_scaling = state_scaling
        nmpcc.input_scaling = input_scaling

        nmpcc.contouring_weight = 0.5
        nmpcc.cont_weight = 0.25
        nmpcc.progress_weight = 4

        nmpcc.diff_from_center = 0.15

        nmpcc.input_reg = 1e-3

        nmpcc_sol = eval(sys, nmpcc, x0, T_sim, dt_sim, mpcc=True)
        for res in [nmpcc_sol]:
          times = res.times
          states = res.states
          inputs = res.inputs
          computation_times = res.computation_time
          solve_times = res.solver_time

          x = [states[i][0] for i in range(len(times))]
          y = [states[i][1] for i in range(len(times))]

          progress = res.progress

          finish_idx = 0
          for i in range(len(progress)):
            if progress[i] / nmpcc.progress_scaling[0] > goal_progress:
              finish_idx = i
              break

          laptime = finish_idx * dt_disc
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

          all_laptimes.append(laptime)
          all_violations.append(violation)

          all_solvetimes.append(np.sum(solve_times))

      all_laptimes = np.array(all_laptimes)
      mask = all_laptimes < 2
      all_laptimes[mask] = np.nan

      all_laptimes_all_runs.append(all_laptimes)
      all_violations_all_runs.append(all_violations)
      all_solvetimes_all_runs.append(all_solvetimes)

    median_solve_times = np.median(all_solvetimes_all_runs, axis=0)
    median_laptimes = np.median(all_laptimes_all_runs, axis=0)
    median_violations = np.median(all_violations_all_runs, axis=0)

    plt.figure(f'laptimes {track}')
    plt.errorbar(median_solve_times, median_laptimes,
                 xerr=[median_solve_times - np.min(all_solvetimes_all_runs, axis=0), np.max(all_solvetimes_all_runs, axis=0) - median_solve_times], 
                 label=name, color=solver_to_color_map[name],
                 marker='o')
    # plt.plot(median_solve_times, all_laptimes, label=name)

    plt.figure(f'violations {track}')
    plt.errorbar(median_solve_times, median_violations,
                 xerr=[median_solve_times - np.min(all_solvetimes_all_runs, axis=0), np.max(all_solvetimes_all_runs, axis=0) - median_solve_times], 
                 label=name, color=solver_to_color_map[name],
                 marker='o')
    # plt.plot(median_solve_times, all_violations, label=name)

  plt.figure(f'laptimes {track}')
  plt.xlabel('Solve time [s]')
  plt.ylabel('Laptime [s]')
  plt.legend()

  plt.savefig(f'./img/blog/amzracecar_laptimes_comp_{track}.png', format='png', dpi=300, bbox_inches = 'tight')
  
  plt.figure(f'violations {track}')
  plt.xlabel('Solve time [s]')
  plt.ylabel('Track violation')
  plt.legend()

  plt.savefig(f'./img/blog/amzracecar_violations_comp_{track}.png', format='png', dpi=300, bbox_inches = 'tight')

if __name__ == "__main__":
  # motivation_horizon()
  # motivation_discretization()

  # make_cart_pole_animation()
  # make_quadcopter_animation()
  # make_double_cart_pole_animation()

  # mpcc_amzracecar_test(track='race')
  # mpcc_amzracecar_test(track='fig8')

  # make_masspoint_cost_comp_for_blog(num_runs=1, randomize_start_state=True)
  # make_quadcopter_cost_comp_for_blog(num_runs=100, randomize_start_state=True)
  # make_cartpole_cost_comp_for_blog(num_runs=100, randomize_start_state=True)
  
  # not used in the blog
  # make_double_cartpole_cost_comp_for_blog()

  make_racecar_analysis(num_runs=10, track='fig8')
  make_racecar_analysis(num_runs=10, track='race')

  plt.show()
