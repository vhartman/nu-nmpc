import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacobian

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

class ChainOfMasses(System):
  def __init__(self, num_masses=3):
    self.num_masses = num_masses
    self.dim = 2
    self.x0 = np.array([0, 0])

    self.state_dim = num_masses * 2 * 2 + 2
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20] * (num_masses * 2 * 2 + 2)])
    self.input_limits = np.array([[-1, 1], [-1, 1]])

    super().__init__()

  def f(self, state, u):
    x = state

    x_dot = jnp.zeros(len(state))

    L = 0.033
    m = 0.03
    D = 0.1

    for i in range(0, self.num_masses):
      prev_idx = (i-1) * self.dim
      idx = i * self.dim
      next_idx = (i + 1) * self.dim
      
      prev = self.x0
      if (i > 0):
        prev = x[prev_idx:prev_idx+self.dim]
      curr = x[idx:idx+self.dim]
      next = x[next_idx:next_idx+self.dim]

      def F(x1, x2):
        # delta = x[idx:idx+self.dim] - x[next_idx:next_idx+self.dim]
        delta = x2 - x1
        F = D * (1 - L / jnp.linalg.norm(delta)) * delta
        return F

      offset = (self.num_masses + 1) * self.dim
      x_dot = x_dot.at[offset + idx:offset+idx+2].set(1. / m * (F(curr, next) - F(prev, curr)) + np.array([0., -9.81]))
      x_dot = x_dot.at[idx:idx+2].set(x[offset + idx:offset+idx+2])

    x_dot = x_dot.at[self.num_masses*self.dim:(self.num_masses+1)*self.dim].set(u)

    return x_dot

class Racecar(System):
  def __init__(self):
    # self.state_dim = 6
    # self.input_dim = 2

    # self.state_limits = np.array([[-20, 20], [-5, 20], [-20, 20], [-80, 80], [-80, 80], [-80, 80]])
    # self.input_limits = np.array([[0, 2], [0, 2]])

    super().__init__()

  def f(self, state, u):
    pass
    # x = state

    # # First derivative, xdot = [vy, vz, phidot, ay, az, phidotdot]
    # g     = 9.81    # Gravitational acceleration (m/s^2)
    # m     = 0.18    # Mass (kg)
    # Ixx   = 0.00025 # Mass moment of inertia (kg*m^2)
    # L     = 0.086   # Arm length (m)

    # F = u[0] + u[1]
    # M = (u[1] - u[0]) * L

    # return jnp.asarray([x[3],
    #         x[4],
    #         x[5],
    #         -F * jnp.sin(x[2]) / m,
    #         F * jnp.cos(x[2]) / m - g,
    #         M / Ixx])
