#include <cmath>
#include <gtest/gtest.h>
#include <mppi/vehicle_util.hpp>

class DynamicsTest : public ::testing::Test
{
  protected:
    VehicleParams params;

    void SetUp() override
    {
        // default params
        params = VehicleParams();
    }
};

TEST_F(DynamicsTest, StraightLineMotion)
{
    VehicleState state{0.0, 0.0, 0.0, 5.0, 0.0, 0.0}; // high speed
    ControlInput control{0.0, 0.0}; // no accel, no steering

    VehicleState next = stepDynamics(state, params, control, 0.02);

    // small x change, no y change, and velocity should be maintained
    EXPECT_NEAR(next.x, 0.1, 0.01);
    EXPECT_NEAR(next.y, 0.0, 0.01);
    EXPECT_NEAR(next.vx, 5.0, 0.1);
}

TEST_F(DynamicsTest, Acceleration)
{
    VehicleState state{0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
    ControlInput control{2.0, 0.0}; // acceleration, no steering

    VehicleState next = stepDynamics(state, params, control, 0.02);

    // change in vx
    EXPECT_GT(next.vx, 5.0);
    EXPECT_NEAR(next.vx, 5.04, 0.01);
}

TEST_F(DynamicsTest, Turning)
{
    VehicleState state{0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
    ControlInput control{0.0, 0.3}; // no acceleration, steering

    // only yaw rate should change first iter
    VehicleState next1 = stepDynamics(state, params, control, 0.02);
    EXPECT_NEAR(fabs(next1.yawRate), 0.1, 0.01);

    // steering change should propogate in second iteration
    VehicleState next2 = stepDynamics(next1, params, control, 0.02);
    EXPECT_NEAR(fabs(next2.heading), 0.002, 0.0002);
}

TEST_F(DynamicsTest, VelocitySafeguard)
{
    VehicleState state{0.0, 0.0, 0.0, 0.05, 0.0, 0.0}; // low speed
    ControlInput control{0.0, 0.1}; // no acceleration, small steering

    // stable at small values
    EXPECT_NO_THROW({
        VehicleState next = stepDynamics(state, params, control, 0.02);
    });
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}