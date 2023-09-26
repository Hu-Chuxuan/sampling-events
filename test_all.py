import numpy as np
from gen import event_generater
from bootstrap import sample_bootstrap
from hoeffding import sample_hoeffding
from clt import sample_clt
from mcmc import sample_mcmc
from importance_sampling import importance_sampling, adaptive_importance_sampling
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--confidence_level", default=0.95, help="confidence interval")
argParser.add_argument("-n", "--num_samples", default=10000, help="confidence interval")
args = argParser.parse_args()

confidence_level = float(args.confidence_level)
num_samples = int(args.num_samples)

event_data, _, a, b = event_generater(a=0, b=100)
print(f"Actual Mean from Data: {np.mean(event_data):.2f}")

sample_mean, lower_ci, upper_ci = sample_clt(event_data, num_samples, confidence_level)
# Print the resultss
print(f"Estimated Mean from CLT: {sample_mean:.2f}")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")

sample_mean, lower_ci, upper_ci = sample_mcmc(event_data, a, b, num_samples, confidence_level)
print(f"Estimated Mean from MCMC: {sample_mean:.2f}.")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")

sample_mean, lower_ci, upper_ci = sample_hoeffding(event_data, confidence_level, a, b, num_samples)
print(f"Estimated Mean from Hoeffding: {sample_mean:.2f}.")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")

sample_mean, lower_ci, upper_ci = sample_bootstrap(event_data, confidence_level)
print(f"Estimated Mean from Bootstrap: {sample_mean:.2f}.")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_ci:.2f}, {upper_ci:.2f})")

estimated_average, lower_bound, upper_bound = importance_sampling(event_data, num_samples, a, b, confidence_level)
print(f"Estimated Mean from Importance Sampling: {estimated_average:.2f}")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

estimated_average, lower_bound, upper_bound = adaptive_importance_sampling(event_data, num_samples, a, b, confidence_level)
print(f"Estimated Mean from Adaptive Importance Sampling: {estimated_average:.2f}")
print(f"{int(100*confidence_level)}% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

