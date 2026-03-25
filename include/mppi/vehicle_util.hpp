#pragma once

#include <cmath>
#include <cuda_runtime.h>

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
        : mu(1.0489), csf(4.718), csr(5.456), lf(0.15875), lr(0.17145), h(0.074), mass(3.74), iz(0.04712), g(9.81), alphaSat(0.19), fxSplit(0.45), minVelocity(-5.0), maxVelocity(7.5), minAcceleration(-20.0), maxAcceleration(20.0), minSteeringAngle(-0.5236), maxSteeringAngle(0.5236) // ~52.7 degrees
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
    double rAccel, rSteering;
};

__host__ __device__ inline VehicleState stepDynamics(VehicleState &state, const VehicleParams &params, const ControlInput &control, float dt)
{
    VehicleState nextState = state;

    double vx_safe = fmax(0.5, state.vx);

    // compute derivatives
    // pose derivatives
    double x_dot = vx_safe * cos(state.heading) - state.vy * sin(state.heading);
    double y_dot = vx_safe * sin(state.heading) + state.vy * cos(state.heading);
    double heading_dot = state.yawRate;

    // speed derivatives
    double vx_dot = control.acceleration + state.vy * state.yawRate;

    double vy_dot = -(params.csf + params.csr) / (params.mass * vx_safe) * state.vy;
    vy_dot += (params.csr * params.lr - params.csf * params.lf) / (params.mass * vx_safe) * state.yawRate;
    vy_dot -= vx_safe * state.yawRate;
    vy_dot += params.csf / params.mass * control.steering;

    double yawRate_dot = state.vy * (params.csr * params.lr - params.csf * params.lf) / (params.iz * vx_safe);
    yawRate_dot += state.yawRate * (params.csr * params.lr * params.lr - params.csf * params.lf * params.lf) / (params.iz * vx_safe);
    yawRate_dot += control.steering * params.lf * params.csf / params.iz;

    // compute integration - euler for now
    nextState.x += x_dot * dt;
    nextState.y += y_dot * dt;
    nextState.heading += heading_dot * dt;
    nextState.vx = fmax(nextState.vx + vx_dot * dt, 0.0);
    nextState.vy += vy_dot * dt;
    nextState.yawRate += yawRate_dot * dt;

    return nextState;
}