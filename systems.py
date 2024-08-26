import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacobian

jax.config.update("jax_enable_x64", True)

class System:
  def __init__(self):
    self.sys_jac = jax.jit(jax.jacfwd(self.f, argnums=0))
    self.input_jac = jax.jit(jax.jacfwd(self.f, argnums=1))

    self.tmp = jax.jit(jax.jacfwd(self.step_rk4, argnums=0))
    self.tmp_inp = jax.jit(jax.jacfwd(self.step_rk4, argnums=1))

    self.state_names = []
    self.input_names = []

    self.jit_lin = jax.jit(self.linearization)
    self.vmap_linearization = jax.jit(jax.vmap(self.linearization, in_axes=(1, 1, 0)))

  def f(self, state, u):
    raise NotImplementedError("Implement me pls")

  def step(self, state, u, dt, method="euler"):
    if method == "euler":
      return self.step_euler(state, u, dt)
    elif method == "rk4":
      return self.step_rk4(state, u, dt)
    else:
      return self.step_heun(state, u, dt)

  def step_euler(self, state, u, dt):
    x_dot = self.f(state, u)
    new_state = state + dt * x_dot

    return jnp.asarray(new_state)
  
  # heuns method
  def step_heun(self, state, u, dt):
    x_dot = self.f(state, u)
    tmp = state + dt * x_dot
    new_state = state + dt/2. * (x_dot + self.f(tmp, u))
    return new_state
  
  def step_rk4(self, state, u, dt_full, steps=1):
    x = state
    dt = dt_full / steps
    for _ in range(steps):
      k1 = self.f ( x, u )
      k2 = self.f ( x + dt * k1 / 2.0, u)
      k3 = self.f ( x + dt * k2 / 2.0, u)
      k4 = self.f ( x + dt * k3, u)

      x = x + dt * ( k1 + 2.0 * k2 + 2.0 * k3 + k4 ) / 6.0

    return x

  def A(self, x, u, dt):
    A_disc = self.sys_jac(x, u)
    return A_disc * dt

  def B(self, x, u, dt):
    B_disc = self.input_jac(x, u)
    return B_disc * dt

  def K(self, x, u, dt):
    return self.A(x, u, dt) @ x[:, None] + self.B(x, u, dt) @ u[:, None] - dt * self.f(x, u)[:, None]

  def jitted_linearization(self, x, u, dt):
    return self.jit_lin(x, u, dt)

  # f = f() + A(x-xref) + B(u-uref)
  # xn = x + dt * (f() + A(x-xref) + B(u-uref))
  def linearization(self, x, u, dt):
    # A_disc = self.sys_jac(x, u)
    # B_disc = self.input_jac(x, u)
    # K = dt * (A_disc @ x[:, None] + B_disc @ u[:, None] - self.f(x, u)[:, None])

    # return A_disc*dt, B_disc * dt, K

    # A_disc = self.tmp(x, u, dt) - np.eye(len(x))
    # B_disc = self.tmp_inp(x, u, dt)
    # K = (A_disc @ x[:, None] + B_disc @ u[:, None] - dt * self.f(x, u)[:, None])

    # return A_disc, B_disc, K

    A_disc = self.tmp(x, u, dt)
    B_disc = self.tmp_inp(x, u, dt)
    K = (A_disc @ x[:, None] + B_disc @ u[:, None] - self.step_rk4(x, u, dt)[:, None])

    return A_disc, B_disc, K

class MasspointND(System):
  def __init__(self, N):
    self.N = N

    self.state_dim = 2*N
    self.input_dim = N

    self.state_limits = np.array([[-10, 10], [-2, 2]]*N)
    self.input_limits = np.array([[-10, 10]]*N)

    super().__init__()

    self.state_names = ['p', 'v'] * N
    self.input_names = ['a'] * N

  def f(self, state, u):
    A_cont = np.zeros((self.state_dim, self.state_dim))
    for i in range(self.input_dim):
      A_cont[i*2, i*2+1] = 1

    B_cont = np.zeros((self.state_dim, self.input_dim))
    for i in range(self.input_dim):
      B_cont[i*2+1, i] = 1

    return jnp.asarray(A_cont @ state + B_cont @ u)

class Masspoint2D(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 2

    self.state_limits = np.array([[-10, 10], [-5, 5], [-10, 10], [-5, 5]])
    self.input_limits = np.array([[-5, 5], [-5, 5]])

    super().__init__()

    self.state_names = ['x', 'v_x', 'y', 'v_y']
    self.input_names = ['a_x', 'a_y']

  def f(self, state, u):
    return jnp.asarray([state[1], u[0], state[3], u[1]])

class JerkMasspointND(System):
  def __init__(self, N):
    self.N = N

    self.state_dim = 3*N
    self.input_dim = N

    self.state_limits = np.array([[-10, 10], [-3, 3], [-10, 10]]*N)
    self.input_limits = np.array([[-20, 20]]*N)

    super().__init__()

    self.state_names = ['p', 'v', 'a'] * N
    self.input_names = ['j'] * N

  def f(self, state, u):
    A_cont = np.zeros((self.state_dim, self.state_dim))
    for i in range(self.input_dim):
      A_cont[i*3, i*3+1] = 1
      A_cont[i*3+1, i*3+2] = 1

    B_cont = np.zeros((self.state_dim, self.input_dim))
    for i in range(self.input_dim):
      B_cont[i*3+2, i] = 1

    return jnp.asarray(A_cont @ state + B_cont @ u)

class JerkMasspoint2D(System):
  def __init__(self):
    self.state_dim = 6
    self.input_dim = 2

    self.state_limits = np.array([[-10, 10], # pos
                                  [-1, 1], # vel
                                  [-3, 3], # acc
                                  [-10, 10], 
                                  [-1, 1], 
                                  [-3, 3]])
    self.input_limits = np.array([[-50, 50], 
                                  [-50, 50]])

    super().__init__()

    self.state_names = ['x', 'v_x', 'a_x', 'y', 'v_y', 'a_y']
    self.input_names = ['j_x', 'j_y']

  def f(self, state, u):
    return jnp.asarray([state[1], state[2], u[0], state[4], state[5], u[1]])

class LaplacianDynamics(System):
  def __init__(self):
    self.state_dim = 3
    self.input_dim = 3

    self.state_limits = np.array([[-10, 10], [-10, 10], [-10, 10]])
    self.input_limits = np.array([[-5, 0], [-5, 0], [-5, 0]])

    super().__init__()

    self.state_names = ['1', '2', '3']
    self.input_names = ['u_1', 'u_2', 'u_3']

  def f(self, state, u):
    A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    return (A @ state[:, None] + u[:, None]).flatten()

class Cartpole(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 1

    self.state_limits = np.array([[-1, 1], [-40, 40], [-20, 20], [-30, 30]])
    self.input_limits = np.array([[-20, 20]])

    super().__init__()
  
    self.state_names = ['x', 'alpha', 'v', 'omega']
    self.input_names = ['f']

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

    self.state_limits = np.array([[-5, 5], [-20, 20], [-20, 20], [-10, 10], [-30, 30], [-30, 30]])
    self.input_limits = np.array([[-20, 20]])

    super().__init__()

    self.state_names = ['x', 'theta_1', 'theta_2', 'v', 'w_1', 'w_2']
    self.input_names = ['f']

  def f(self, state, u):
    x = state
    m = 1.
    m1 = 0.2
    m2 = 0.2
    l1 = 0.25
    l2 = 0.25
    g = 9.80665
    L1 = 2 * l1
    L2 = 2 * l2
    J1 = m1 * L1 ** 2 / 12
    J2 = m2 * L2 ** 2 / 12

    # Helper variables
    h1 = m + m1 + m2
    h2 = m1 * l1 + m2 * L1
    h3 = m2 * l2
    h4 = m1 * l1 ** 2 + m2 * L1 ** 2 + J1
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
    
    G = jnp.asarray([[0.], 
                     [h7 * jnp.sin(x[1])], 
                     [h8 * jnp.sin(x[2])]])

    # Coriolis and centrifugal vector
    C = jnp.asarray([[0., h2 * jnp.sin(x[1]) * x[4], h3 * jnp.sin(x[2]) * x[5]], 
                  [0., 0., -h5 * jnp.sin(x[1] - x[2]) * x[5]], 
                  [0., h5 * jnp.sin(x[1] - x[2]) * x[4], 0.]])

    u = u[0]
    Q = jnp.asarray([[u], [0.], [0.]])

    # jax.debug.print("C: \n{c}", c=C)
    # jax.debug.print("x: \n{x}", x=x[3:6])
    # jax.debug.print("G: \n{G}", G=G.T)

    # Create state space
    M_invers = jnp.linalg.inv(M)
    q_dot = C @ x[3:6][:, None]
    # jax.debug.print("qd: \n{qd}", qd=q_dot)

    q_dotdot = M_invers @ (Q + q_dot + G)

    # Create function
    x_vel = x[3]
    a1_vel = x[4]
    a2_vel = x[5]
    x_acc = q_dotdot[0, 0]
    a1_acc = q_dotdot[1, 0]
    a2_acc = q_dotdot[2, 0]

    return jnp.asarray([x_vel, a1_vel, a2_vel, x_acc, a1_acc, a2_acc])

class Quadcopter2D(System):
  def __init__(self):
    self.state_dim = 6
    self.input_dim = 2

    self.state_limits = np.array([[-0.1, 20], [-5, 5], [-20, 20], [-10, 10], [-10, 10], [-80, 80]])
    self.input_limits = np.array([[0, 2], [0, 2]])

    super().__init__()

    self.state_names = ['x', 'y', 'alpha', 'v_x', 'v_y', 'omega']
    self.input_names = ['f1', 'f2']

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

    self.state_dim = (num_masses + 1) * 2 + num_masses * 2
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20] * self.state_dim]).reshape(-1, 2)
    self.input_limits = np.array([[-1, 1], [-1, 1]])

    super().__init__()

    self.state_names = ['tmp']
    self.input_names = ['tmp']

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
  
class CSTR(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 2

    self.state_limits = np.array([[0.1, 2], [0.1, 2], [50, 150], [50, 150]])
    self.input_limits = np.array([[5, 100], [-8500, 0.0]])

    super().__init__()

    self.state_names = ['tmp']
    self.input_names = ['tmp']

  # taken from https://www.do-mpc.com/en/latest/example_gallery/CSTR.html
  def f(self, state, u):
    C_a, C_b, T_R, T_K = state
    F, Q_dot = u

    # Certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 #0.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    alpha = 1.
    beta = 1.

    K_1 = beta * K0_ab * jnp.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * jnp.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * jnp.exp((-alpha*E_A_ad)/((T_R+273.15)))

    T_dif = T_R - T_K

    C_a_dot = F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2)
    C_b_dot = -F*C_b + K_1*C_a - K_2*C_b
    T_R_dot = ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R))
    T_K_dot = (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k)

    return jnp.asarray([C_a_dot, C_b_dot, T_R_dot, T_K_dot], dtype=np.float64)
  
# taken from https://www.do-mpc.com/en/latest/example_gallery/batch_reactor.html
class BatchBioreactor(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 1

    self.state_limits = np.array([[0, 3.7], [-0.01, 200], [0.0, 3.0], [0.0, 200]])
    self.input_limits = np.array([[0, 0.2]])

    super().__init__()

    self.state_names = ['tmp']
    self.input_names = ['tmp']

  def f(self, state, u):
    X_s, S_s, P_s, V_s = state
    inp = u[0]

    mu_m  = 0.02
    K_m   = 0.05
    K_i   = 5.0
    v_par = 0.004
    Y_p   = 1.2

    # Uncertain parameters:
    Y_x  = 0.4
    S_in = 200.

    # Auxiliary term
    mu_S = mu_m*S_s/(K_m+S_s+(S_s**2/K_i))

    # Differential equations
    X_s_dot = mu_S*X_s - inp/V_s*X_s
    S_s_dot = -mu_S*X_s/Y_x - v_par*X_s/Y_p + inp/V_s*(S_in-S_s)
    P_s_dot = v_par*X_s - inp/V_s*P_s
    V_s_dot = inp

    return jnp.asarray([X_s_dot, S_s_dot, P_s_dot, V_s_dot])

# bicycle model from liniger, ttps://arxiv.org/pdf/1711.07300
class Racecar(System):
  def __init__(self):
    self.state_dim = 6
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20], [-20, 20], [-20, 20], [-0.1, 4], [-2, 2], [-7, 7]])
    self.input_limits = np.array([[-0.1, 1], [-0.35, 0.35]])

    # self.state_limits = np.array([[-20, 20], [-20, 20], [-10, 10], [0.3, 3.5], [-2, 2], [-7, 7]])
    # self.input_limits = np.array([[-0.1, 1], [-0.35, 0.35]])

    super().__init__()

    self.state_names = ['x', 'y', 'phi', 'v_x', 'v_y', 'omega']
    self.input_names = ['acc', 'steer']

  def f(self, state, u):
    x, y, phi, v_x, v_y, omega = state
    d, delta = u
    
    # v_x = jnp.max(jnp.asarray([v_x, 0.3]))

    m = 0.041
    I_z = 27.8e-6
    l_f = 0.029
    l_r = 0.033

    C_m_1 = 0.287
    C_m_2 = 0.0545
    Cr0 = 0.0518
    Cr2 = 0.00035

    B_r = 3.3852
    C_r = 1.2691
    D_r = 0.1737

    B_f = 2.579
    C_f = 1.2
    D_f = 0.192

    # make everything play nicely if v_x < 0.1
    v_x = jnp.maximum(v_x, 0.3)
    v_y = jax.lax.cond(v_x > 0.3, lambda: v_y, lambda: 0.)

    alpha_f = -jnp.arctan2((omega * l_f + v_y), v_x) + delta
    F_f_y = D_f * jnp.sin(C_f * jnp.arctan(B_f*alpha_f))

    alpha_r = jnp.arctan2((omega*l_r - v_y), v_x)
    F_r_y = D_r * jnp.sin(C_r * jnp.arctan(B_r*alpha_r))

    F_r_x = (C_m_1 - C_m_2 * v_x)*d - Cr0 - Cr2*v_x**2

    x_dot = v_x * jnp.cos(phi) - v_y * jnp.sin(phi)
    y_dot = v_x * jnp.sin(phi) + v_y * jnp.cos(phi)
    phi_dot = omega
    v_x_dot = 1/m * (F_r_x - F_f_y * jnp.sin(delta) + m*v_y*omega)
    # print("vxd")
    # print(v_x_dot)
    # print(F_r_x)
    v_y_dot = 1/m * (F_r_y + F_f_y * jnp.cos(delta) - m*v_x*omega)
    omega_dot = 1/I_z * (F_f_y * l_f*jnp.cos(delta) - F_r_y * l_r)

    # print("ff")
    # print(F_f_y)
    # print(-F_r_y)
    # print(omega_dot)

    return jnp.asarray([x_dot, y_dot, phi_dot, v_x_dot, v_y_dot, omega_dot])

class AMZRacecar(System):
  def __init__(self):
    self.state_dim = 8
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20], [-20, 20], [-20, 20], [-0.1, 3.5], [-2, 2], [-7, 7], [-0.1, 1], [-0.35, 0.35]])
    self.input_limits = np.array([[-5, 5], [-5, 5]])

    # self.state_limits = np.array([[-20, 20], [-20, 20], [-10, 10], [0.3, 3.5], [-2, 2], [-7, 7]])
    # self.input_limits = np.array([[-0.1, 1], [-0.35, 0.35]])

    super().__init__()

    self.state_names = ['x', 'y', 'phi', 'v_x', 'v_y', 'omega', 'acc', 'steer']
    self.input_names = ['acc_d', 'steer_d']

  def f(self, state, u):
    m = 0.041
    I_z = 27.8e-6
    l_f = 0.029
    l_r = 0.033

    C_m_1 = 0.287
    C_m_2 = 0.0545
    Cr0 = 0.0518
    Cr2 = 0.00035

    B_r = 3.3852
    C_r = 1.2691
    D_r = 0.1737

    B_f = 2.579
    C_f = 1.2
    D_f = 0.192

    def f_dyn(state, u):
      x, y, phi, v_x, v_y, omega, d, delta = state
      d_dot, delta_dot = u

      v_x = jnp.maximum(v_x, 0.1)
      alpha_f = -jnp.arctan2((omega * l_f + v_y), v_x) + delta
      # print(alpha_f)
      F_f_y = D_f * jnp.sin(C_f * jnp.arctan(B_f*alpha_f))
      # print(F_f_y)

      alpha_r = jnp.arctan2((omega*l_r - v_y), v_x)
      F_r_y = D_r * jnp.sin(C_r * jnp.arctan(B_r*alpha_r))

      F_r_x = (C_m_1 - C_m_2 * v_x)*d - Cr0 - Cr2*v_x**2

      x_dot = v_x * jnp.cos(phi) - v_y * jnp.sin(phi)
      y_dot = v_x * jnp.sin(phi) + v_y * jnp.cos(phi)
      phi_dot = omega

      v_x_dot = 1/m * (F_r_x - F_f_y * jnp.sin(delta) + m*v_y*omega)
      v_y_dot = 1/m * (F_r_y + F_f_y * jnp.cos(delta) - m*v_x*omega)
      omega_dot = 1/I_z * (F_f_y * l_f*jnp.cos(delta) - F_r_y * l_r)

      return jnp.asarray([x_dot, y_dot, phi_dot, v_x_dot, v_y_dot, omega_dot, d_dot, delta_dot])

    # kinematic model
    def f_kin(state, u):
      x, y, phi, v_x, v_y, omega, d, delta = state
      d_dot, delta_dot = u

      F_r_x = C_m_1*d - Cr0 - Cr2*v_x**2
  
      x_dot = v_x * jnp.cos(phi) - v_y * jnp.sin(phi)
      y_dot = v_x * jnp.sin(phi) + v_y * jnp.cos(phi)
      phi_dot = omega

      v_x_dot = F_r_x/m
      v_y_dot = (delta_dot * v_x + delta * v_x_dot) * l_r / (l_r + l_f)
      omega_dot = (delta_dot * v_x + delta * v_x_dot) * 1. / (l_r + l_f)
      
      return jnp.asarray([x_dot, y_dot, phi_dot, v_x_dot, v_y_dot, omega_dot, d_dot, delta_dot])

    x, y, phi, v_x, v_y, omega, d, delta = state

    v_x_blend_min = 0.5
    v_x_blend_max = 1.5

    v_x_stop_grad = jax.lax.stop_gradient(v_x)

    blend = jnp.minimum(jnp.maximum((v_x_stop_grad - v_x_blend_min) / (v_x_blend_max - v_x_blend_min), 0), 1)
    res = blend * f_dyn(state, u) + (1-blend) * f_kin(state, u)

    return res

class Unicycle(System):
  def __init__(self):
    self.state_dim = 3
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20], [-20, 20], [-50, 50]])
    self.input_limits = np.array([[-0.1, 3], [-10, 10]])

    # self.state_limits = np.array([[-20, 20], [-20, 20], [-10, 10], [0.3, 3.5], [-2, 2], [-7, 7]])
    # self.input_limits = np.array([[-0.1, 1], [-0.35, 0.35]])

    super().__init__()
    
    self.state_names = ['x', 'y', 'phi']
    self.input_names = ['v', 'w']

  def f(self, state, u):
    _, _, theta = state
    us, uw = u

    x_dot = us * jnp.cos(theta)
    y_dot = us * jnp.sin(theta)
    theta_dot = uw

    return jnp.asarray([x_dot, y_dot, theta_dot])
  
class Unicycle2ndOrder(System):
  def __init__(self):
    self.state_dim = 5
    self.input_dim = 2

    # x, y, theta, v, omega
    self.state_limits = np.array([[-20, 20], [-20, 20], [-50, 50], [-0.1, 20], [-30, 30]])
    self.input_limits = np.array([[-100, 100], [-100, 100]])

    super().__init__()
    
    self.state_names = ['x', 'y', 'phi', 'v', 'omega']
    self.input_names = ['a', 'wd']

  def f(self, state, u):
    x, y, theta, v, omega = state
    us, uw = u

    x_dot = v * jnp.cos(theta)
    y_dot = v * jnp.sin(theta)
    theta_dot = omega

    v_dot = us
    omega_dot = uw

    return jnp.asarray([x_dot, y_dot, theta_dot, v_dot, omega_dot])
  
class Acrobot(System):
  def __init__(self):
    self.state_dim = 4
    self.input_dim = 1

    # x, y, theta, v, omega
    self.state_limits = np.array([[-20, 20], [-20, 20], [-20, 20], [-20, 20]])
    self.input_limits = np.array([[5, 5]])

    super().__init__()

    self.state_names = ['theta_1', 'theta_2', 't1_d', 't2d']
    self.input_names = ['tau']

  # http://incompleteideas.net/book/ebook/node110.html
  def f(self, state, u):
    theta_1, theta_2, theta_1_dot, theta_2_dot = state
    tau = u[0]

    m_1 = m_2 = 1.
    l_1 = l_2 = 1.
    l_c1 = l_c2 = 0.5
    I_1 = I_2 = 1.

    g=9.81
    
    d_1 = m_1 * l_c1**2 + m_2 * (l_1**2 + l_c2**2 + 2 * l_1*l_c2*jnp.cos(theta_2)) + I_1 + I_2
    d_2 = m_2 * (l_c2**2 + l_1*l_c2 * jnp.cos(theta_2)) + I_2

    phi_2 = m_2 * l_c2 * g * jnp.cos(theta_1 + theta_2 - np.pi/2)
    phi_1 = -m_2*l_1*l_c2*theta_2_dot**2*jnp.sin(theta_2) - 2*m_2*l_1*l_c2*theta_2_dot*theta_1_dot*jnp.sin(theta_2) + (m_1*l_c1 + m_2*l_1)*g*jnp.cos(theta_1 - np.pi/2) + phi_2

    theta_2_ddot = (tau + d_2/d_1 * phi_1 - m_2*l_1*l_c2*theta_1_dot**2*jnp.sin(theta_2) - phi_2) / (m_2*l_c2**2 + I_2 - d_2**2/d_1)
    theta_1_ddot = -1/d_1 * (d_2*theta_2_ddot + phi_1)

    return jnp.asarray([theta_1_dot, theta_2_dot, theta_1_ddot, theta_2_ddot])

class CarWithTrailer(System):
  def __init__(self):
    # x, y, theta, beta
    self.state_dim = 4
    # v, alpha
    self.input_dim = 2

    self.state_limits = np.array([[-20, 20], [-20, 20], [-20, 20], [-20, 20]])
    self.input_limits = np.array([[-0.1, 5], [-3, 3]])

    super().__init__()
        
    self.state_names = ['x', 'y', 'theta', 'beta']
    self.input_names = ['v', 'alpha']

  # https://ch.mathworks.com/help/mpc/ug/truck-and-trailer-automatic-parking-using-multistage-mpc.html
  def f(self, state, u):
    _, _, theta, beta = state
    v, alpha = u

    M_1 = 1
    L_1 = 6
    L_2 = 10

    x_dot = v * jnp.cos(beta) * (1 + M_1/L_1 * jnp.tan(beta) * jnp.tan(alpha))*jnp.cos(theta)
    y_dot = v * jnp.cos(beta) * (1 + M_1/L_1 * jnp.tan(beta) * jnp.tan(alpha))*jnp.sin(theta)
    theta_dot = v * (jnp.sin(beta) / L_2 - M_1 / (L_1 * L_2) * jnp.cos(beta)*jnp.tan(alpha))
    beta_dot = v * (jnp.tan(alpha)/L_1 - jnp.sin(beta)/L_2 + M_1/(L_1*L_2)*jnp.cos(beta) * jnp.tan(alpha))

    return jnp.asarray([x_dot, y_dot, theta_dot, beta_dot])

class PlanarQuadrotorPole(System):
  pass

class RobotArmPole(System):
  pass