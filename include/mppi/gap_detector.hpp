#include <sensor_msgs/msg/laser_scan.hpp>
#include <vector>
#include <cmath>

struct Gap {
    int start, end;
};

/**
 * filter and average the msg data -- going to skip filtering, assume theres no nans for now
 */
std::vector<float> preprocessScan(const sensor_msgs::msg::LaserScan::SharedPtr msg, int windowSize)
{
    // Crop to +-90 degrees
    float fov_limit = M_PI / 2.0f;
    int start_idx = static_cast<int>((-fov_limit - msg->angle_min) / msg->angle_increment);
    int end_idx = static_cast<int>((fov_limit - msg->angle_min) / msg->angle_increment);
    start_idx = std::max(start_idx, 0);
    end_idx = std::min(end_idx, static_cast<int>(msg->ranges.size()) - 1);

    std::vector<float> data(msg->ranges.begin() + start_idx, msg->ranges.begin() + end_idx + 1);
    int n = data.size();
    int half = windowSize / 2;

    std::vector<float> padded(n + 2 * half);
    for (int i = 0; i < half; i++)
        padded[i] = data[0];
    for (int i = 0; i < n; i++)
        padded[i + half] = data[i];
    for (int i = 0; i < half; i++)
        padded[n + half + i] = data[n - 1];

    std::vector<float> res;
    res.reserve(n);

    double currSum = 0;
    for (int i = 0; i < windowSize; i++)
        currSum += padded[i];
    res.push_back(currSum / windowSize);

    for (int i = 1; i < n; i++)
    {
        currSum += padded[i + windowSize - 1] - padded[i - 1];
        res.push_back(currSum / windowSize);
    }

    return res;
}

/**
 * pseudocode:
 * gaps = []
in_gap = false
for i in 0..N:
    if ranges[i] > r_min:
        if not in_gap:
            gap_start = i
            in_gap = true
    else:
        if in_gap:
            gaps.append((gap_start, i - 1))
            in_gap = false
 */
std::vector<Gap> findGaps(const sensor_msgs::msg::LaserScan::SharedPtr msg, float r_min)
{
    float fov_limit = M_PI / 2.0f;
    int crop_offset = std::max(0, static_cast<int>((-fov_limit - msg->angle_min) / msg->angle_increment));

    std::vector<float> data_avg = preprocessScan(msg, 5);
    std::vector<Gap> gaps;
    int gap_start = 0;
    bool in_gap = false;

    for (int i = 0; i < static_cast<int>(data_avg.size()); i++)
    {
        if (data_avg[i] > r_min)
        {
            if (!in_gap)
            {
                gap_start = i;
                in_gap = true;
            }
        }
        else if (in_gap)
        {
            Gap gap;
            gap.start = gap_start + crop_offset;
            gap.end = i + crop_offset;
            gaps.push_back(gap);
            in_gap = false;
        }
    }

    if (in_gap)
    {
        Gap gap;
        gap.start = gap_start + crop_offset;
        gap.end = static_cast<int>(data_avg.size()) - 1 + crop_offset;
        gaps.push_back(gap);
    }

    return gaps;
}
