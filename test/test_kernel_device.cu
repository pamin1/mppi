#include <cuda_runtime.h>
#include <mppi/mppi_kernel.cuh>

// device test kernels
__global__ void testCostKernel(double *d_result)
{
    VehicleState pred{1.0, 1.0, 0.1, 5.0, 0.0, 0.0};
    VehicleState ref{0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
    ControlInput ctrl{0.0, 0.0};
    CostWeights weights{1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1};

    *d_result = computeCost(pred, ref, ctrl, weights);
}

__global__ void testCostZeroKernel(double *d_result)
{
    VehicleState state{1.0, 1.0, 0.1, 5.0, 0.0, 0.0};
    ControlInput ctrl{0.0, 0.0};
    CostWeights weights{1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1};

    // perfect tracking - same state for predicted and reference
    *d_result = computeCost(state, state, ctrl, weights);
}

// host wrapper functions
extern "C"
{

    void runCostPositiveTest(double *h_result)
    {
        double *d_result;
        cudaMalloc(&d_result, sizeof(double));

        testCostKernel<<<1, 1>>>(d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }

    void runCostZeroTest(double *h_result)
    {
        double *d_result;
        cudaMalloc(&d_result, sizeof(double));

        testCostZeroKernel<<<1, 1>>>(d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }

    cudaError_t runMPPIKernelSmokeTest()
    {
        int samples = 100;
        int horizon = 10;

        ControlInput *d_controls;
        double *d_costs;
        ControlInput *d_nominal;
        VehicleState *d_refTraj;
        VehicleState *d_currState;
        CostWeights *d_weights;
        VehicleParams *d_params;
        curandState *d_rngStates;

        cudaMalloc(&d_controls, samples * horizon * sizeof(ControlInput));
        cudaMalloc(&d_costs, samples * sizeof(double));
        cudaMalloc(&d_nominal, horizon * sizeof(ControlInput));
        cudaMalloc(&d_refTraj, horizon * sizeof(VehicleState));
        cudaMalloc(&d_currState, sizeof(VehicleState));
        cudaMalloc(&d_weights, sizeof(CostWeights));
        cudaMalloc(&d_params, sizeof(VehicleParams));
        cudaMalloc(&d_rngStates, samples * sizeof(curandState));

        VehicleState h_state{0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
        VehicleParams h_params;
        CostWeights h_weights{1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1};

        ControlInput h_nominal[10];
        for (int i = 0; i < 10; ++i)
        {
            h_nominal[i].acceleration = 0.0;
            h_nominal[i].steering = 0.0;
        }

        VehicleState h_refTraj[10];
        for (int i = 0; i < 10; ++i)
        {
            h_refTraj[i] = VehicleState{i * 0.5, 0.0, 0.0, 5.0, 0.0, 0.0};
        }

        cudaMemcpy(d_currState, &h_state, sizeof(VehicleState), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params, &h_params, sizeof(VehicleParams), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, &h_weights, sizeof(CostWeights), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nominal, h_nominal, horizon * sizeof(ControlInput), cudaMemcpyHostToDevice);
        cudaMemcpy(d_refTraj, h_refTraj, horizon * sizeof(VehicleState), cudaMemcpyHostToDevice);

        setupRNG<<<1, samples>>>(d_rngStates, 12345);
        cudaDeviceSynchronize();

        mppiKernel<<<1, samples>>>(
            d_controls, d_costs, d_nominal, d_refTraj, d_currState,
            d_weights, d_params, d_rngStates,
            samples, horizon, 0.02f, 2.0f, 0.3f);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        cudaFree(d_controls);
        cudaFree(d_costs);
        cudaFree(d_nominal);
        cudaFree(d_refTraj);
        cudaFree(d_currState);
        cudaFree(d_weights);
        cudaFree(d_params);
        cudaFree(d_rngStates);

        return err;
    }

    int runMPPIKernelValidityTest(double *h_costs, int samples_to_check)
    {
        int samples = 100;
        int horizon = 10;

        ControlInput *d_controls;
        double *d_costs;
        ControlInput *d_nominal;
        VehicleState *d_refTraj;
        VehicleState *d_currState;
        CostWeights *d_weights;
        VehicleParams *d_params;
        curandState *d_rngStates;

        cudaMalloc(&d_controls, samples * horizon * sizeof(ControlInput));
        cudaMalloc(&d_costs, samples * sizeof(double));
        cudaMalloc(&d_nominal, horizon * sizeof(ControlInput));
        cudaMalloc(&d_refTraj, horizon * sizeof(VehicleState));
        cudaMalloc(&d_currState, sizeof(VehicleState));
        cudaMalloc(&d_weights, sizeof(CostWeights));
        cudaMalloc(&d_params, sizeof(VehicleParams));
        cudaMalloc(&d_rngStates, samples * sizeof(curandState));

        VehicleState h_state{0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
        VehicleParams h_params;
        CostWeights h_weights{1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1};

        ControlInput h_nominal[10];
        VehicleState h_refTraj[10];
        for (int i = 0; i < 10; ++i)
        {
            h_nominal[i] = {0.0, 0.0};
            h_refTraj[i] = {i * 0.5, 0.0, 0.0, 5.0, 0.0, 0.0};
        }

        cudaMemcpy(d_currState, &h_state, sizeof(VehicleState), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params, &h_params, sizeof(VehicleParams), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, &h_weights, sizeof(CostWeights), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nominal, h_nominal, horizon * sizeof(ControlInput), cudaMemcpyHostToDevice);
        cudaMemcpy(d_refTraj, h_refTraj, horizon * sizeof(VehicleState), cudaMemcpyHostToDevice);

        setupRNG<<<1, samples>>>(d_rngStates, 54321);
        cudaDeviceSynchronize();

        mppiKernel<<<1, samples>>>(
            d_controls, d_costs, d_nominal, d_refTraj, d_currState,
            d_weights, d_params, d_rngStates,
            samples, horizon, 0.02f, 2.0f, 0.3f);

        cudaDeviceSynchronize();

        cudaMemcpy(h_costs, d_costs, samples_to_check * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_controls);
        cudaFree(d_costs);
        cudaFree(d_nominal);
        cudaFree(d_refTraj);
        cudaFree(d_currState);
        cudaFree(d_weights);
        cudaFree(d_params);
        cudaFree(d_rngStates);

        return 0;
    }

}