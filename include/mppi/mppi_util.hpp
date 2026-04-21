#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

// parameterize vehicle state
struct VehicleParams
{
    float mu;       // friction coefficient
    float csf, csr; // front/rear cornering stiffness
    float lf, lr;   // front/rear axle to CG distance
    float h;        // CG height
    float mass;     // vehicle mass
    float iz;       // yaw moment of inertia
    float g;        // gravity
    float alphaSat; // slip angle saturation
    float fxSplit;  // longitudinal force split
    float minVelocity, maxVelocity;
    float minAcceleration, maxAcceleration;
    float minSteeringAngle, maxSteeringAngle;

    // Default constructor with F1TENTH values
    VehicleParams()
        : mu(1.0489), csf(4.718), csr(5.456), lf(0.15875), lr(0.17145), h(0.074), mass(3.74), iz(0.04712), g(9.81), alphaSat(0.19), fxSplit(0.45), minVelocity(-5.0), maxVelocity(10.0), minAcceleration(-10.0), maxAcceleration(20.0), minSteeringAngle(-0.5236), maxSteeringAngle(0.5236) // ~52.7 degrees
    {
    }
};

struct ControlInput
{
    double acceleration, steering; // integrate during publishing to command velocity and steering
};

// 3 DoF planar model: [x,y,heading]
// Dynamic Bicycle Model: [x,y,heading,vx,vy,yawRate]
struct VehicleState
{
    double x, y;
    double heading;
    double vx, vy;
    double yawRate;
};

struct CostWeights
{
    double qX, qY;
    double qHeading;
    double qVx, qVy;
    double qYawRate;
    double rAccel, rSteering, rSteeringRate;
};

struct Gaussian
{
    float mean, std_dev;
};

struct CostmapInfo
{
    const int8_t *data;
    int width;
    int height;
    float resolution;
    float origin_offset;
    int8_t lethal_cost;
};

struct MPPIConfig
{
    int samples;
    int horizon;
    float dt;
    float sigmaAcceleration;
    float sigmaSteering;
};

__host__ __device__ inline VehicleState stepDynamics(VehicleState &state, const VehicleParams &params, const ControlInput &control, float dt)
{
    VehicleState nextState = state;

    double L = params.lf + params.lr;
    double beta = atan(params.lr / L * tan(control.steering));

    nextState.x = state.x + state.vx * cos(state.heading + beta) * dt;
    nextState.y = state.y + state.vx * sin(state.heading + beta) * dt;
    nextState.heading = state.heading + (state.vx / L) * sin(beta) * dt;
    nextState.vx = fmax(state.vx + control.acceleration * dt, 0.0);
    nextState.vy = 0.0;
    nextState.yawRate = (state.vx / L) * sin(beta);

    return nextState;
}