#include <sensor_msgs/msg/laser_scan.hpp>
#include <vector>

struct Gap {
    int start, end;
};

/**
 * filter and average the msg data -- going to skip filtering, assume theres no nans for now
 */
std::vector<float> preprocessScan(const sensor_msgs::msg::LaserScan::SharedPtr msg, int windowSize)
{
    std::vector<float> data = msg->ranges;
    int n = data.size();
    int half = windowSize / 2;

    // Pad edges by clamping to boundary values
    std::vector<float> padded(n + 2 * half);
    for (int i = 0; i < half; i++)
        padded[i] = data[0];
    for (int i = 0; i < n; i++)
        padded[i + half] = data[i];
    for (int i = 0; i < half; i++)
        padded[n + half + i] = data[n - 1];

    // Sliding window over padded array
    std::vector<float> res;
    res.reserve(n);

    double currSum = 0;
    for (int i = 0; i < windowSize; i++)
        currSum += padded[i];
    res.push_back(currSum / windowSize);

    for (int i = windowSize; i < padded.size(); i++)
    {
        currSum += padded[i] - padded[i - windowSize];
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
    std::vector<float> data_avg = preprocessScan(msg, 5);
    std::vector<Gap> gaps;
    int gap_start = 0;
    bool in_gap = false;

    for (int i = 0; i < data_avg.size(); i++)
    {
        if (data_avg[i] > r_min) {
            if (!in_gap) {
                gap_start = i;
                in_gap = true;
            }
        }
        else if (in_gap) {
            Gap gap;
            gap.start = gap_start;
            gap.end = i;
            gaps.push_back(gap);
            in_gap = false;
        }
    }

    if (in_gap)
    {
        Gap gap;
        gap.start = gap_start;
        gap.end = static_cast<int>(data_avg.size()) - 1;
        gaps.push_back(gap);
    }

    return gaps;
}
