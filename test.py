from algorithm import sample
from gen import event_generater
import numpy as np

cis = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999]
error_targets = [0.5, 1, 10, 100, 1000]
events, length, a, b = event_generater()

print("The true value is %d"%length)

for error_target in error_targets:
    for ci in cis:
        estimates = []
        for i in range(100):
            estimate, _, _ = sample(events, a, b, ci = ci, method = 'hoeffding', error_target=error_target)
            estimates.append(estimate)
        avg = np.mean(np.abs(np.array(estimates) - length)/length)
        print("With the designated error target %.1f and the designated confidence interval %.3f, hoeffding generates average sampling error rates of %.4f."%(error_target, ci, avg))

for ci in cis:
    estimates = []
    left_bounds = []
    right_bounds = []
    for i in range(100):
        estimate, left_bound, right_bound = sample(events, a, b, ci = ci, method = 'bootstrap')
        estimates.append(estimate)
        left_bounds.append(left_bound)
        right_bounds.append(right_bound)
    avg = np.mean(estimates)
    left_bounds = np.mean(left_bounds)
    right_bounds = np.mean(right_bounds)
    errors = np.mean(np.abs(np.array(estimates) - length)/length)
    # left_bounds = np.abs(np.array(left_bounds) - length)/length
    # right_bounds = np.abs(np.array(right_bounds) - length)/length
    print("With the designated confidence interval %.3f, bootstrap generates an estimation of %d with CI in (%d, %d), the average sampling error rates of %.4f, "%(ci, avg, left_bounds, right_bounds, errors))