import random
import numpy as np
import math
from gen import event_generater

# Metropolis-Hastings MCMC algorithm
def sample_mcmc(data, a, b, num_samples, confidence_level, proposal_stddev=0.1):
    samples = random.sample(data, num_samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    # Initialize the chain
    chain = []
    current_sample = random.uniform(min(samples), max(samples))  # Initial sample

    # Number of MCMC iterations
    num_iterations = 1000

    # Perform Metropolis-Hastings sampling
    for _ in range(num_iterations):
        # Propose a new sample from the proposal distribution
        proposed_sample = random.normalvariate(current_sample, proposal_stddev)
        
        # Calculate the acceptance ratio
        # this likelihood could be modified
        target_likelihood_current = sum(-(x - current_sample) ** 2 / (2 * sample_var) for x in samples)
        target_likelihood_proposed = sum(-(x - proposed_sample) ** 2 / (2 * sample_var) for x in samples)
        
        acceptance_ratio = math.exp(target_likelihood_proposed - target_likelihood_current)
        
        # Accept or reject the proposed sample
        if random.uniform(0, 1) < acceptance_ratio:
            current_sample = proposed_sample
        
        # Add the current sample to the chain
        chain.append(current_sample)

    # Calculate the estimated average of the subset
    estimated_average = np.mean(chain)
    chain.sort()

    ranges = (1-confidence_level)/2
    lower_ci = chain[int(len(chain)*ranges)]
    upper_ci = chain[int(len(chain)*(1-ranges))]

    return estimated_average, lower_ci, upper_ci

if __name__ == "__main__":
    # Generate synthetic data (event sizes)
    data, _, a, b = event_generater(a=0, b=100)

    # Run the MCMC algorithm
    sample_sizes = [1000, 3000, 5000, 10000]
    confidence_levels = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

    print(f"Actual Mean from Data: {np.mean(data):.2f}")

    for num_samples in sample_sizes:
        for confidence_level in confidence_levels:
            sample_mean, lower_ci, upper_ci = sample_mcmc(data, a, b, num_samples, confidence_level)

            print(f"Estimated Mean from Sample: {sample_mean:.2f} with {num_samples:d} samples")
            print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")
