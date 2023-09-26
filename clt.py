import random
import numpy as np
import scipy.stats as stats
from gen import event_generater

# Function to estimate average and calculate confidence intervals from a sample
def sample_clt(data, sample_size, confidence_level):
    n = len(data)

    sample = random.sample(data, sample_size)
    sample_mean = np.mean(sample)
    std_dev_of_sample_means = np.std(sample)

    z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z_critical * (std_dev_of_sample_means / np.sqrt(sample_size))

    lower_ci = sample_mean - margin_of_error
    upper_ci = sample_mean + margin_of_error

    return sample_mean, lower_ci, upper_ci
if __name__ == "__main__":
    # Generate event data
    event_data, _, a, b = event_generater(a=0, b=100)

    # Sample size and number of samples for estimation
    sample_sizes = [1000, 3000, 5000, 10000]
    confidence_levels = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
    print(f"Actual Mean from Data: {np.mean(event_data):.2f}")
    for sample_size in sample_sizes:
        for confidence_level in confidence_levels:
            # Estimate average and confidence intervals from a sample
            sample_mean, lower_ci, upper_ci = sample_clt(event_data, sample_size, confidence_level)
            # Print the resultss
            print(f"Estimated Mean from Sample: {sample_mean:.2f} with {sample_size:d} samples")
            print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")
