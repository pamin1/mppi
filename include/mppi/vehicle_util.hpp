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
        : mu(1.0489), csf(4.718), csr(5.456), lf(0.15875), lr(0.17145), h(0.074), mass(3.74), iz(0.04712), g(9.81), alphaSat(0.19), fxSplit(0.45), minVelocity(-5.0), maxVelocity(20.0), minAcceleration(-10.0), maxAcceleration(9.51), minSteeringAngle(-0.9189), maxSteeringAngle(0.9189) // ~52.7 degrees
    {
    }
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