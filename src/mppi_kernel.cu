#include <mppi/mppi_kernel.cuh>

__global__ void mppiKernel(ControlInput *controlSamples, double *costs, const ControlInput *nominalControlSequence, const VehicleState *refTrajectory, VehicleState *currState, const CostWeights *weights, const VehicleParams *params, curandState *states, int samples, int horizon, float dt, float sigmaAccel, float sigmaSteering, float minAccel, float maxAccel, float minSteer, float maxSteer)
{
    // each thread will handle 1 of the MPPI samples
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k >= samples)
    {
        return;
    }

    // get cuRand state
    curandState localState = states[k];

    // first we need the control sequence for this horizon
    ControlInput controls[30];
    for (int t = 0; t < horizon; ++t)
    {
        float accelNoise = curand_normal(&localState) * sigmaAccel;
        float steerNoise = curand_normal(&localState) * sigmaSteering;

        controls[t].acceleration = nominalControlSequence[t].acceleration + accelNoise;
        controls[t].steering = nominalControlSequence[t].steering + steerNoise;

        controls[t].acceleration = clamp(controls[t].acceleration, minAccel, maxAccel);
        controls[t].steering = clamp(controls[t].steering, minSteer, maxSteer);

        // store in global memory for next iteration
        controlSamples[k * horizon + t] = controls[t];
    }

    VehicleState rollingState = *currState;
    VehicleParams p = *params;
    CostWeights w = *weights;
    double cost = 0;
    for (int i = 0; i < horizon; i++)
    {
        rollingState = stepDynamics(rollingState, p, controls[i], dt);
        cost += computeCost(rollingState, refTrajectory[i], controls[i], w);
    }

    // store costs and cuRAND
    costs[k] = cost;
    states[k] = localState;
}

__global__ void softMax()
{
    // step 1: find the minmum cost -- cpu

    // step 2: for each sample and cost we get an exponential weight -- gpu

    // step 3: normalize the weights -- gpu
}