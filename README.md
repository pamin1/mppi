# Model Predictive Path Integral (MPPI) Controller
MPPI implementation for racing (i.e. F1/10th)

## Controller
MPPI is a sampling-based controller which rolls out N vehicle trajectories with a distribution of steering and acceleration control inputs. Typically, the sample size is large to get good coverage; I used 10,000 samples when testing, but it could work with as few as 1,000.

After rolling out the trajectories, the cost to go is evaluated. In this implementation, I opted for a quadratic cost function penalizing deviation from a known trajectory. The trajectory provided is processed into a vector of 6-state points:

$$[x, y, \theta, v_x, v_y, \dot{\theta}]$$

The state of the vehicle is known by the simulator, so there is always perfect odometry input, which provides the initial pose to roll out from. Then for each sample, the controller creates a set of control inputs sampled from a normal distribution around the previous sequence of control inputs up to the horizon length. The trajectory rolls out with each of the control inputs in the horizon, updating the sample's pose and accumulating the trajectory cost. The pose is only stored locally, but the trajectory cost is stored in a vector.

At this point there is a vector of control input sequences and another vector of corresponding trajectory costs. The costs are exponentially weighted and normalized with respect to the least cost. This creates a relation to weight the control input sequences, with better controls receiving a higher weighting.

- `temperature`: controls the exponential weighting — smaller temperature $\rightarrow$ heavier weighting towards minimum cost trajectories, and vice versa.

Finally, the normalized weights are used to create a single aggregated control sequence. Each sample's acceleration and steering commands are weighted by the normalized weight and blended with the previous/nominal control input.

- `alpha`: controls how much the new control inputs are "trusted" by blending with the nominal control.

The first control input from the sequence is then published to the vehicle and the rest are shifted forward by one timestep (receding horizon + warm-starting the control sequence).

[![MPPI Controller + F1/10th Gym Simulation](/figures/mppi_thumbnail.png)](https://youtu.be/wxVFO-ZpAfo)

## Path Planner
Using OpenCV and SciPy, an optimal path is generated for the MPPI controller. Initially, OpenCV finds the contours and establishes a CSV of points describing the centerline and the distance to the left and right walls. Next in the pipeline, the raceline optimizer uses SciPy to find a minimum-curvature line.

Once the raceline geometry is established, a forward-backward pass computes the speed profile, constrained by friction, gravity, and acceleration/deceleration limits. The result is a CSV of x, y, vx, yaw, and kappa.

![Fig 1: Raceline (15 m/s top speed)](/figures/optimized_speed_map_fast.png)

## Implementation Details
- **CUDA/C++:** The sample computations are parallelized using CUDA kernels. During the MPPI rollout, each thread handles a single sample: generating control perturbations, rolling out the trajectory, and computing the cost.
- **Thrust:** The Thrust library is used to perform reductions during the exponential weighting and aggregation steps. Custom kernels could have been implemented for this, but Thrust is far more efficient and keeps the scope manageable.
- **cuRAND:** Each sample has a uniquely seeded `curandState` to generate Gaussian noise for the control sequences. The persistent internal state of cuRAND produces a different sequence of noise without re-seeding per iteration, providing better exploration and distribution.

# Acknowledgement
- [Aggressive Driving with Model Predictive Path Integral Control](https://ieeexplore.ieee.org/abstract/document/7487277)
- [F1/10th (RoboRacer)](https://github.com/f1tenth)