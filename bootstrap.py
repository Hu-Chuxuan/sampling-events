import random
import numpy as np
from gen import event_generater

# Average estimated by confidence interval using bootstrap sampling
def sample_bootstrap(events, ci, resample = 1000, sample_size = 10):
    # doing bootstrap
    sample_mean = []
    for _ in range(resample):
        samples = random.sample(events, sample_size)
        sample_mean.append(np.mean(samples))
    
    # finding average
    sample_mean.sort()
    ranges = (1-ci)/2
    
    avg = np.mean(sample_mean)
    
    return avg, sample_mean[int(resample*ranges)], sample_mean[int(resample*(1-ranges))]

if __name__ == "__main__":
    # Generate synthetic data (event sizes)
    data, _, a, b = event_generater(a=0, b=100)

    confidence_levels = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

    print(f"Actual Mean from Data: {np.mean(data):.2f}")

    for confidence_level in confidence_levels:
        sample_mean, lower_ci, upper_ci = sample_bootstrap(data, confidence_level)

        print(f"Estimated Mean from Sample: {sample_mean:.2f}.")
        print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")