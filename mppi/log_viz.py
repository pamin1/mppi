#!/usr/bin/env python3

import sys
import csv
import matplotlib.pyplot as plt

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append({
                'lap': int(row['lap']),
                'time': float(row['time']),
                'collisions': int(row['collisions']),
                'avg_speed': float(row['avg_speed']),
                'distance': float(row['distance']),
            })
    return data

def plot(data):
    laps = [d['lap'] for d in data]
    times = [d['time'] for d in data]
    speeds = [d['avg_speed'] for d in data]
    collisions = [d['collisions'] for d in data]
    distances = [d['distance'] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('F1TENTH Lap Results', fontsize=14, fontweight='bold')

    # Lap time
    ax = axes[0, 0]
    ax.plot(laps, times, 'o-', color='#3b82f6', linewidth=2, markersize=6)
    ax.axhline(min(times), color='#22c55e', linestyle='--', linewidth=1, label=f'Best: {min(times):.2f}s')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Time (s)')
    ax.set_title('Lap time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average speed
    ax = axes[0, 1]
    ax.plot(laps, speeds, 'o-', color='#22c55e', linewidth=2, markersize=6)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Average speed')
    ax.grid(True, alpha=0.3)

    # Collisions
    ax = axes[1, 0]
    colors = ['#ef4444' if c > 0 else '#22c55e' for c in collisions]
    ax.bar(laps, collisions, color=colors, alpha=0.8)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Collisions')
    ax.set_title('Collisions per lap')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis='y')

    # Distance
    ax = axes[1, 1]
    ax.bar(laps, distances, color='#f59e0b', alpha=0.8)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance per lap')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lap_results.png', dpi=150)
    print('Saved to lap_results.png')
    plt.show()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'lap_results.csv'
    data = load_csv(path)
    plot(data)