import numpy as np

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