import numpy as np
import jax

def squircle(t):
  w = t * 0.8
  x = abs(jnp.cos(w)) ** (2/12) * jnp.sign(jnp.cos(w))
  y = abs(jnp.sin(w)) ** (2/12) * jnp.sign(jnp.sin(w))
  # return jnp.asarray([x, 0, y, 0]).reshape((-1, 1))
  return jnp.asarray([x, y])

def square(t, side_length=1):
  t = (t * 1.0) % 4
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
  
  return jax.numpy.asarray([x, y])

track_center = np.loadtxt('./track/center.txt')
track_inner = np.loadtxt('./track/inner.txt')
track_outer = np.loadtxt('./track/outer.txt')
def racetrack(t, track=track_center):
  num_pts = len(track[0])
  idx = int(num_pts * t / 4.) % num_pts

  x = track[0][idx]
  y = track[1][idx]

  return np.array([x, y])

ts_square_track = np.linspace(0, 4, 200)
square_path = [(square(t,3)[0], square(t,3)[1]) for t in ts_square_track]

def square_track(t):
  idx = int(200 * t/4.) % 200

  x = square_path[idx][0]
  y = square_path[idx][1]

  return np.array([x,y])

def figure_eight(t, r=1):
  x = jax.numpy.sin(t) * r
  y = jax.numpy.sin(t) * jax.numpy.cos(t) * r
  return jax.numpy.asarray([x, y])

from scipy.interpolate import interp1d, make_interp_spline
def figure_eight_bounds(r=1, offset=0.15):
  figure_eight_knots = np.linspace(0, 2*np.pi, 200)
  figure_eight_pts = [figure_eight(t, r) for t in figure_eight_knots]
  figure_eight_spline = make_interp_spline(figure_eight_knots, figure_eight_pts)
  figure_eight_tangent = figure_eight_spline.derivative(1)

  inners = []
  outers = []

  for i, t in enumerate(figure_eight_knots):
    tangent = figure_eight_tangent(t)

    normal = np.array([-tangent[1], tangent[0]])
    normal = normal / np.linalg.norm(normal)
    
    inners.append(figure_eight_pts[i] + offset*normal)
    outers.append(figure_eight_pts[i] - offset*normal)

  return inners, outers