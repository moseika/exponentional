import time
import numpy as np
from numba import jit
import math
from all_tests import t_stats


def generate_data(distribution_name, sample_size, ):
    if distribution_name == "Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.exponential(scale=1.0, size=sample_size)
    elif distribution_name == "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.lognormal(mean=0.0, sigma=1.0, size=sample_size)
    elif distribution_name == "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000":
        shape = 0.8  # shape parameter (a)
        scale = 1.0  # scale parameter (this is just a multiplier in numpy's weibull)
        shift = 0.0  # shift parameter (if necessary)
        sample = np.random.weibull(a=shape, size=sample_size) * scale + shift
    elif distribution_name == "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.weibull(a=1.2, size=sample_size)
    else:
        print("Неизвестное распределение")
        return []
    return sample

    
def modeling_criterion(criterion_name, data_name, number_of_s, size_of):
    start_time = time.time()
    #total_numbers = 16600
    #sample_size = 300
    #name = "Крит.показ-ти Аткинсона ПолуНорм(-0.25)"
        
    
    result = []
    for i in range(number_of_s):
        result.append(t_stats(criterion_name, generate_data(data_name, size_of)))
    end_time = time.time()
    return result

#print(modeling_criterion("Крит.показ-ти Андерсона-Дарлинга", "Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", 10, 100))
