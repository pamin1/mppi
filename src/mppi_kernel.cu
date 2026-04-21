#include <mppi/kernel_launch.hpp>
#include <mppi/mppi_kernel.cuh>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

__global__ void setupRNG(curandState *states, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void mppiKernel(ControlInput *controlSamples, MPPIConfig *config, double *costs, const ControlInput *nominalControlSequence, const VehicleState *refTrajectory, const VehicleState *currState, const CostWeights *weights, const VehicleParams *params, curandState *states)
{
    // each thread will handle 1 of the MPPI samples
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k >= config->samples)
    {
        return;
    }

    // get cuRand state
    curandState localState = states[k];

    // first we need the control sequence for this horizon
    ControlInput controls[30];
    for (int t = 0; t < config->horizon; ++t)
    {
        float accelNoise = curand_normal(&localState) * config->sigmaAcceleration;
        float steerNoise = curand_normal(&localState) * config->sigmaSteering;

        controls[t].acceleration = nominalControlSequence[t].acceleration + accelNoise;
        controls[t].steering = nominalControlSequence[t].steering + steerNoise;

        controls[t].acceleration = clamp(controls[t].acceleration, params->minAcceleration, params->maxAcceleration);
        controls[t].steering = clamp(controls[t].steering, params->minSteeringAngle, params->maxSteeringAngle);

        // store in global memory for next iteration
        controlSamples[k * config->horizon + t] = controls[t];
    }

    VehicleState rollingState = *currState;
    VehicleParams p = *params;
    CostWeights w = *weights;
    double cost = 0;
    for (int i = 0; i < config->horizon; i++)
    {
        rollingState = stepDynamics(rollingState, p, controls[i], config->dt);
        cost += computeCost(rollingState, refTrajectory[i], controls[i], w);

        if (i > 0)
        {
            double dsteer = controls[i].steering - controls[i - 1].steering;
            cost += w.rSteeringRate * dsteer * dsteer;
        }
    }

    // store costs and cuRAND
    costs[k] = cost;
    states[k] = localState;
}

__global__ void aggregateControls(ControlInput *optimalControls, const ControlInput *sampleControls, const ControlInput *nominalControls, const double *weightedCosts, float alpha, int samples, int horizon)
{
    // time step parallel for now -- could improve to block parallel

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= horizon)
        return;

    double accel_sum = 0.0;
    double steer_sum = 0.0;

    // compute the timestep sum across all samples
    for (int i = 0; i < samples; ++i) // 30 threads doing sample number of FMA ops -- inefficient use of cores
    {
        ControlInput ctrl = sampleControls[i * horizon + t];
        accel_sum += weightedCosts[i] * ctrl.acceleration;
        steer_sum += weightedCosts[i] * ctrl.steering;
    }

    optimalControls[t].acceleration = accel_sum * alpha + (1 - alpha) * nominalControls[t].acceleration;
    optimalControls[t].steering = steer_sum * alpha + (1 - alpha) * nominalControls[t].steering;
}

void launchSetupRNG(curandState *d_states, unsigned long seed, int grid, int block)
{
    setupRNG<<<grid, block>>>(d_states, seed);
    cudaDeviceSynchronize();
}

void launchMPPIKernel(ControlInput *d_controlSamples, MPPIConfig *config, double *d_costs, const ControlInput *d_nominalSequence, const VehicleState *d_refTraj, const VehicleState *d_currState, const CostWeights *d_weights, const VehicleParams *d_params, curandState *d_rngStates, int grid, int block)
{
    mppiKernel<<<grid, block>>>(d_controlSamples, config, d_costs, d_nominalSequence, d_refTraj, d_currState, d_weights, d_params, d_rngStates);
    cudaDeviceSynchronize();
}

void launchThrustWeighting(double *d_costs, int samples, float temperature)
{
    // use thrust to do parallel reductions and cost weighting
    // reduce for min cost
    double minCost = thrust::reduce(thrust::device_pointer_cast(d_costs), thrust::device_pointer_cast(d_costs + samples), INFINITY, thrust::minimum<double>());

    // exponential weighting functor to transform the cost array
    thrust::transform(thrust::device_pointer_cast(d_costs), thrust::device_pointer_cast(d_costs + samples), thrust::device_pointer_cast(d_costs), weightFunctor(minCost, temperature));

    // sum to get the sum of the weights
    double weightSum = thrust::reduce(thrust::device_pointer_cast(d_costs), thrust::device_pointer_cast(d_costs + samples));

    // normalize the weighted costs
    thrust::transform(thrust::device_pointer_cast(d_costs), thrust::device_pointer_cast(d_costs + samples), thrust::device_pointer_cast(d_costs), thrust::placeholders::_1 / weightSum);
}

void launchAggregateControls(ControlInput *d_optimalControls, const ControlInput *d_sampleControls, const ControlInput *d_nominalControls, const double *d_weightedCosts, float alpha, int samples, int horizon, int grid, int block)
{
    aggregateControls<<<grid, block>>>(d_optimalControls, d_sampleControls, d_nominalControls, d_weightedCosts, alpha, samples, horizon);
    cudaDeviceSynchronize();
}