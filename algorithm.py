# the sample algorithm
from math import sqrt, log, ceil, floor
import random
import numpy as np

# This function generates an estimate for the number of samples
# Input: a - the lowest number of events happening at each interval
# b - the highest number of events happening at each interval
# ci - the confidence interval set by user
# method - the sampling method, choose from 'hoeffding' or 'bootstrap'
# error_target - the error target set by user, only used for hoeffding
# Return: the estimated total number of events occurred, left-bound and right-bound of CI (only applicable for bootstrap)
def sample(events, a, b, ci = 0.9, method = 'bootstrap', error_target = 1):
    left_bound, right_bound = -1, -1
    if method == 'bootstrap':
        avg, left_bound, right_bound = sample_bootstrap(events, ci)
    elif method == 'hoeffding':
        avg = sample_hoeffding(events, ci, a, b, error_target)
    return int(avg*1e6), left_bound*1e6, right_bound*1e6

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


# Average estimated by confidence interval using hoeffding inequality
def sample_hoeffding(events, ci, a, b, error_target):
    n = ceil(-(b-a)**2*log((1-ci)/2)/(2*error_target**2))
    if n >= 1e6:
        print("Can't be achieved through hoeffding sampling.")
        return -1
    samples = random.sample(events, n)
    res = np.mean(samples)
    return res
    
