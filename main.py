import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacobian

import time

import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class System:
  def __init__(self):
    self.sys_jac = jax.jit(jax.jacfwd(self.f, argnums=0))
    self.input_jac = jax.jit(jax.jacfwd(self.f, argnums=1))

  def f(self, state, u):
      raise NotImplementedError("Implement me pls")

  def step(self, state, u, dt):
    x_dot = self.f(state, u)
    new_state = state + dt * x_dot

    return jnp.asarray(new_state)

  def A(self, x, u, dt):
    A_disc = self.sys_jac(x, u)
    return A_disc * dt

  def B(self, x, u, dt):
    B_disc = self.input_jac(x, u)
    return B_disc * dt

  def K(self, x, u, dt):
      return self.A(x, u, dt) @ x[:, None] + self.B(x, u, dt) @ u[:, None] - dt * self.f(x, u)[:, None]

  def linearization(self, x, u, dt):
    A_disc = self.sys_jac(x, u)
    B_disc = self.input_jac(x, u)
    K = dt * (A_disc @ x[:, None] + B_disc @ u[:, None] - self.f(x, u)[:, None])

    return A_disc * dt, B_disc * dt, K

class Masspoint2D(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 2

    self.state_limits = np.array([[-10, 10], [-10, 10], [-10, 10], [-10, 10]])
    self.input_limits = np.array([[-5, 5], [-5, 5]])

    super().__init__()

  def f(self, state, u):
    return jnp.asarray([state[1], u[0], state[3], u[1]])

class LaplacianDynamics(System):
  def __init__(self):
    self.state_dim = 3
    self.input_dim = 3

    self.state_limits = np.array([[-10, 10], [-10, 10], [-10, 10]])
    self.input_limits = np.array([[-5, 0], [-5, 0], [-5, 0]])

    super().__init__()

  def f(self, state, u):
    A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    return (A @ state[:, None] + u[:, None]).flatten()

class Cartpole(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 1

    self.state_limits = np.array([[-20, 20], [-40, 40], [-20, 20], [-100, 100]])
    self.input_limits = np.array([[-100, 100]])

    super().__init__()
  
  def f(self, state, u):
    x = state
    q2 = x[2]
    q2_d = x[3]

    l = 0.5
    m1 = 1.
    m2 = 0.3
    g = 9.81

    U = u[0]

    x_dot = x[1]
    x_acc = ((l*m2*jnp.sin(q2)*q2_d**2)+U+(m2*g*jnp.cos(q2)*jnp.sin(q2)))/(m1 + m2*(1-(jnp.cos(q2))**2))
    theta_dot = x[3]
    theta_acc = -((l*m2*jnp.cos(q2)*jnp.sin(q2)*q2_d**2)+(U*jnp.cos(q2)) + ((m1+m2)*g*jnp.sin(q2)))/(l*m1 + l*m2*(1-(jnp.cos(q2))**2))

    return jnp.asarray([x_dot, x_acc, theta_dot, theta_acc])

class DoubleCartpole(System):
  def __init__(self):
    self.state_dim = 6
    self.input_dim = 1

    self.state_limits = np.array([[-20, 20], [-20, 20], [-20, 20], [-80, 80], [-80, 80], [-80, 80]])
    self.input_limits = np.array([[-80, 80]])

    super().__init__()

  def f(self, state, u):
    x = state
    m = 0.6
    m1 = 0.2
    m2 = 0.2
    l1 = 0.25
    l2 = 0.25
    g = 9.81
    L1 = 2 * l1
    L2 = 2 * l2
    J1 = m1 * L1 ** 2 / 12
    J2 = m2 * L2 ** 2 / 12

    # Helper variables
    h1 = m + m1 + m2
    h2 = m1 * l1 + m2 * L1
    h3 = m2 * l2
    h4 = m1 * l1 ** 2 + m2 * L1 ** 1 + J1
    h5 = m2 * l2 * L1
    h6 = m2 * l2 ** 2 + J2
    h7 = m1 * l1 * g + m2 * L1 * g
    h8 = m2 * l2 * g

    #print(x)
    #print(u)

    # inertia matrix
    M = jnp.asarray([[h1, h2 * jnp.cos(x[1]), h3 * jnp.cos(x[2])],
                     [h2 * jnp.cos(x[1]), h4, h5 * jnp.cos(x[1] - x[2])],
                     [h3 * jnp.cos(x[2]), h5 * jnp.cos(x[1] - x[2]), h6] ])
    G = jnp.asarray([0., -h7 * jnp.sin(x[1]), -h8 * jnp.sin(x[2])])

    # Coriolis and centrifugal vector
    C = jnp.asarray([[0., -h2 * jnp.sin(x[1]) * x[4], -h3 * x[5] * jnp.sin(x[2])], 
                  [0., 0., h5 * x[5] * jnp.sin(x[1] - x[2])], 
                  [0., -h5 * x[4] * jnp.sin(x[1] - x[2]), 0.]])

    u = u[0]
    Q = jnp.asarray([u, 0., 0.])

    # Create state space
    M_invers = jnp.linalg.inv(M)
    q_dot = C @ x[3:6]
    q_dotdot = M_invers @ (Q.T - q_dot - G.T)

    # Create function
    x_vel = x[3]
    a1_vel = x[4]
    a2_vel = x[5]
    x_acc = q_dotdot[0]
    a1_acc = q_dotdot[1]
    a2_acc = q_dotdot[2]

    return jnp.asarray([x_vel, a1_vel, a2_vel, x_acc, a1_acc, a2_acc])

class Quadcopter2D(System):
  def __init__(self):
    self.state_dim = 6
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20], [-5, 20], [-20, 20], [-80, 80], [-80, 80], [-80, 80]])
    self.input_limits = np.array([[0, 2], [0, 2]])

    super().__init__()

  def f(self, state, u):
    x = state

    # First derivative, xdot = [vy, vz, phidot, ay, az, phidotdot]
    g     = 9.81    # Gravitational acceleration (m/s^2)
    m     = 0.18    # Mass (kg)
    Ixx   = 0.00025 # Mass moment of inertia (kg*m^2)
    L     = 0.086   # Arm length (m)

    F = u[0] + u[1]
    M = (u[1] - u[0]) * L

    return jnp.asarray([x[3],
            x[4],
            x[5],
            -F * jnp.sin(x[2]) / m,
            F * jnp.cos(x[2]) / m - g,
            M / Ixx])

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
    self.sqp_iters = 3

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
        result = prob.solve(solver='OSQP', eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
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

  nmpc = NMPC(sys, 10, dt)

  dts = get_linear_spacing(dt, 20 * dt, 5)
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

  nmpc = NMPC(sys, 5, dt*4)

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

def main():
    pass

if __name__ == "__main__":
  test_quadcopter()
  #test_laplacian_dynamics()
  #test_masspoint()

  #cartpole_test()
  #double_cartpole_test()
