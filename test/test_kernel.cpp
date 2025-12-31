#include <cmath>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

// declarations of wrapper functions
extern "C"
{
    void runCostPositiveTest(double *h_result);
    void runCostZeroTest(double *h_result);
    cudaError_t runMPPIKernelSmokeTest();
    int runMPPIKernelValidityTest(double *h_costs, int samples);
}

class KernelTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        cudaDeviceSynchronize();
    }

    void TearDown() override
    {
        cudaDeviceSynchronize();
    }
};

TEST_F(KernelTest, CostFunctionPositive)
{
    double result;
    runCostPositiveTest(&result);

    EXPECT_GT(result, 0.0) << "Cost should be positive for non-zero error";
    EXPECT_FALSE(std::isnan(result)) << "Cost should not be NaN";
    EXPECT_FALSE(std::isinf(result)) << "Cost should not be infinite";
}

TEST_F(KernelTest, CostFunctionZeroForPerfectTracking)
{
    double result;
    runCostZeroTest(&result);

    EXPECT_NEAR(result, 0.0, 1e-6) << "Cost should be zero for perfect tracking";
    EXPECT_FALSE(std::isnan(result)) << "Cost should not be NaN";
}

TEST_F(KernelTest, MPPIKernelLaunches)
{
    cudaError_t err = runMPPIKernelSmokeTest();

    EXPECT_EQ(err, cudaSuccess) << "MPPI kernel failed: " << cudaGetErrorString(err);
}

TEST_F(KernelTest, MPPIKernelProducesValidCosts)
{
    const int samples = 100;
    double costs[samples];

    int result = runMPPIKernelValidityTest(costs, samples);
    EXPECT_EQ(result, 0) << "Kernel validity test setup failed";

    // Check all costs are valid
    for (int i = 0; i < samples; ++i)
    {
        EXPECT_FALSE(std::isnan(costs[i])) << "Cost " << i << " is NaN";
        EXPECT_FALSE(std::isinf(costs[i])) << "Cost " << i << " is infinite";
        EXPECT_GE(costs[i], 0.0) << "Cost " << i << " is negative";
    }

    // Check costs vary (not all identical)
    double min_cost = costs[0];
    double max_cost = costs[0];
    for (int i = 1; i < samples; ++i)
    {
        if (costs[i] < min_cost)
            min_cost = costs[i];
        if (costs[i] > max_cost)
            max_cost = costs[i];
    }

    EXPECT_GT(max_cost - min_cost, 0.01)
        << "Costs should vary across samples (sampling is working)";
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}