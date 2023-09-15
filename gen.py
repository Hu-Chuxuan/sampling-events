import numpy as np
import random

# This function generates event lists in the form of how many events happens at each interval
# Input: a - the lowest number of events happening at each interval
# b - the highest number of events happening at each interval
# Return: res - the event list
# length - the number of total events
# a - the lowest number of events happening at each interval
# b - the highest number of events happening at each interval

def event_generater(a = 0, b = 100):
    '''To be more realistic these should be a list of floating point numbers,
       but it works the same by just placing the number of events at each interval
    '''
    res = []
    for _ in range(int(1e6)):
        size = random.randint(a, b)
        res.append(size)
    length = sum(res)
    return res, length, a, b