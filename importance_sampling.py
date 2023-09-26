import random
import numpy as np
from gen import event_generater
import scipy.stats as stats

def target_distribution(x, a, b):
    return 1.0 / (b - a)

def proposal_distribution(x, a, b):
    return 1.0 / (2*(b-a))

# Function to estimate average using importance sampling
def importance_sampling(data, num_samples, a, b, confidence_level):
    
    # Sample from the current proposal distribution
    sample = random.sample(data, num_samples)

    importance_weights = [target_distribution(x, a, b) / proposal_distribution(x, a, b) for x in sample]
    
    estimated_average = np.mean([x * w / np.mean(importance_weights) for x, w in zip(sample, importance_weights)])

    estimated_variance = np.std([x * w / np.mean(importance_weights) for x, w in zip(sample, importance_weights)])

    t_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    lower_bound = estimated_average - t_critical * (estimated_variance / np.sqrt(num_samples))
    upper_bound = estimated_average + t_critical * (estimated_variance / np.sqrt(num_samples))

    return estimated_average, lower_bound, upper_bound

def adaptive_importance_sampling(data, num_samples, a, b, confidence_level):
    samples = []
    importance_weights = []
    sample = random.sample(data, num_samples)

    for s in sample:
        samples.append(s)
        if len(samples) > 1:
            sample_mean = np.mean(samples)
            sample_stddev = np.std(samples)
            # Use these statistics to adapt the proposal distribution
            proposal_dis = stats.norm(sample_mean, sample_stddev).pdf(s)
        else:
            proposal_dis = proposal_distribution(s,a,b)
        
        importance_weights.append(target_distribution(s,a,b) / proposal_dis)

    estimated_average = np.mean([x * w / np.mean(importance_weights) for x, w in zip(sample, importance_weights)])

    estimated_variance = np.std([x * w / np.mean(importance_weights) for x, w in zip(sample, importance_weights)])

    t_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    lower_bound = estimated_average - t_critical * (estimated_variance / np.sqrt(num_samples))
    upper_bound = estimated_average + t_critical * (estimated_variance / np.sqrt(num_samples))

    return estimated_average, lower_bound, upper_bound

if __name__ == "__main__":
    # Generate event data
    event_data, _, a, b = event_generater(0, 100)
    print(f"Actual Mean from Data: {np.mean(event_data):.2f}")

    sample_sizes = [1000, 3000, 5000, 10000]
    confidence_levels = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

    for confidence_level in confidence_levels:
        for num_samples in sample_sizes:
            # Estimate the average using importance sampling
            estimated_average, lower_bound, upper_bound = importance_sampling(event_data, num_samples, a, b, confidence_level)

            # Print the confidence interval
            print(f"Importance Sampling Estimated Expectation: {estimated_average:.2f} with {num_samples:d} samples")
            print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

    for confidence_level in confidence_levels:
        for num_samples in sample_sizes:
            estimated_average, lower_bound, upper_bound = adaptive_importance_sampling(event_data, num_samples, a, b, confidence_level)
            print(f"Adaptive Importance Sampling Estimated Expectation: {estimated_average:.2f} with {num_samples:d} samples")
            print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
