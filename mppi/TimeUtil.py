#!/usr/bin/env python3

from time import time

class TimeUtil:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.starts = {}  # Start times for each section
        self.current_times = {}  # Most recent timing for each section
        self.total_times = {}  # Accumulated total time for summary
        self.counts = {}  # Count of calls for summary

    def s(self, section='total'):
        """Start timing a section"""
        if not self.enabled:
            return
        self.starts[section] = time()
        if section not in self.counts:
            self.counts[section] = 0
            self.total_times[section] = 0.0

    def e(self, section='total'):
        """End timing a section"""
        if not self.enabled or section not in self.starts:
            return
        elapsed = time() - self.starts[section]
        self.current_times[section] = elapsed * 1000  # Store in milliseconds
        self.total_times[section] += elapsed
        self.counts[section] += 1
        del self.starts[section]

    def get_time(self, section):
        """Get the most recent timing for a section in milliseconds"""
        return self.current_times.get(section, 0.0)

    def summary(self):
        """Print timing summary"""
        if not self.enabled:
            return
        print("\nTiming Summary:")
        for section in sorted(self.total_times.keys()):
            avg_time = self.total_times[section] / max(1, self.counts[section]) * 1000
            total_time = self.total_times[section] * 1000
            print(f"{section:>20}: {avg_time:>8.1f}ms avg, {total_time:>8.1f}ms total, {self.counts[section]:>6} calls")

# sample usage
if __name__ == '__main__':
    from time import sleep
    #Create an instance of exe_timer for all procedures you want to monitor
    # Initialize with True to enable all functions
    # When you're done analyzing, simply change the argument to False or 
    # initialize without an argument, that will cause all methods to return instantly
    t = TimeUtil(True)

    # A typical scenario is to find average execution
    # time during several iterations, average exe time will be 
    # updated during each iteration
    for i in range(1,3):
        # Start global timer in the very beginning of the procedure
        t.s()

        # To track an operation, enclose it with INSTANCE.s('identifier')
        # and INSTANCE.e('identifier')
        # s and e are shorthand for start and end
        # each start() must be matched with an end() with identical identifier
        t.s('sleep2')
        sleep(0.2)
        t.e('sleep2')

        for j in range(1,3):
            t.s('sleep1')
            sleep(0.1)
            t.e('sleep1')

        # not all operations in your procedure will be timed, those not timed are called
        # unaccounted time
        sleep(0.1)

        # it is also possible to track average value of a variable, this is how you do it.
        t.track('var', 5)

        # at the end of the operation, end the global timer with a matching e()
        t.e()
    # this function prints a summary of everything.
    t.summary()
