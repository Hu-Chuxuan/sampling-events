from math import sqrt, log, ceil, floor
import random
import numpy as np
from gen import event_generater

# Average estimated by confidence interval using hoeffding inequality
def sample_hoeffding(events, ci, a, b, num_samples):
    t = sqrt(-1/(2*num_samples)*(b-a)**2*log((1-ci)/2))
    samples = random.sample(events, num_samples)
    res = np.mean(samples)
    return res, res-t, res+t

if __name__ == "__main__":
    # Generate synthetic data (event sizes)
    data, _, a, b = event_generater(a=0, b=100)

    # Run the MCMC algorithm
    sample_sizes = [1000, 3000, 5000, 10000]
    confidence_levels = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

    print(f"Actual Mean from Data: {np.mean(data):.2f}")

    for num_samples in sample_sizes:
        for confidence_level in confidence_levels:
            sample_mean, lower_ci, upper_ci = sample_hoeffding(data, confidence_level, a, b, num_samples)
            print(f"Estimated Mean from Sample: {sample_mean:.2f} with {num_samples:d} samples")
            print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")