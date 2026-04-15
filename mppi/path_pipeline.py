import os
import sys
import pathlib
from baseline_generation import generate_baseline, read_yaml
from raceline_optimizer import optimize_baseline

cwd = os.getcwd()    
input_path = f"{cwd}/src/f1tenth_gym_ros/maps"
output_path = f"{cwd}/src/mppi/resources"

def main():
    fn = sys.argv[1]
    src = f"{input_path}/{fn}"
    dst = f"{output_path}/{fn}.csv"

    generate_baseline(src, dst, True)

    src = dst
    dst = f"{output_path}/{fn}_optimized.csv"
    optimize_baseline(src, dst)

if __name__ == '__main__':
    main()