import numpy as np

#ВЫЧИСЛЕНИЕ КРИТИЧЕСКИХ ЗНАЧЕНИЙ
#Для правостороннего
def calculate_critical_value_right_tail(h0, alpha):
    sorted_h0 = sorted(h0)
    index = int((1 - alpha) * len(sorted_h0))
    return sorted_h0[index]

#Для левостороннего
def calculate_critical_value_left_tail(h0, alpha):
    sorted_h0 = sorted(h0)
    index = int(alpha * len(sorted_h0))
    return sorted_h0[index]

#Для двустороннего
def calculate_critical_values(h0, alpha):
    sorted_h0 = sorted(h0)
    lower_index = int((alpha / 2) * len(sorted_h0))
    upper_index = int((1 - alpha / 2) * len(sorted_h0))
    return sorted_h0[lower_index], sorted_h0[upper_index]

#ВЫЧИСЛЕНИЕ МОЩНОСТИ КРИТЕРИЯ
#Для правостороннего
def calculate_power_right_tail(h1, critical_value):
    count = sum(1 for x in h1 if x > critical_value)
    return count / len(h1)

#Для левостороннего
def calculate_power_left_tail(h1, critical_value):
    count = sum(1 for x in h1 if x < critical_value)
    return count / len(h1)

#Для двустороннего
def calculate_power_two_tailed(h1, lower_critical_value, upper_critical_value):
    count = sum(1 for x in h1 if x < lower_critical_value or x > upper_critical_value)
    return count / len(h1)

def find_power_of(name_h0, name_h1, tail, flag):
    if(flag == 0):
        h0 = np.loadtxt(name_h0, skiprows=2)
        h1 = np.loadtxt(name_h1, skiprows=2)
    else:
        h0 = name_h0
        h1 = name_h1
    power_arr = []
    if(tail == "Правосторонний"):
        critical_value = calculate_critical_value_right_tail(h0, 0.1)
        power = calculate_power_right_tail(h1, critical_value)
        power_arr.append(power)
    elif(tail == "Левосторонний"):
        critical_value = calculate_critical_value_left_tail(h0, 0.1)
        power = calculate_power_left_tail(h1, critical_value)
        power_arr.append(power)

    else:   
        lower_critical_value, upper_critical_value = calculate_critical_values(h0, 0.1)
        power = calculate_power_two_tailed(h1, lower_critical_value, upper_critical_value)

    return(power_arr)

def find_power_of_no_tail(h0, h1, tail):
    power_arr = []
    if(tail == "Правосторонний"):

        critical_value = calculate_critical_value_right_tail(h0, 0.1)
        power = calculate_power_right_tail(h1, critical_value)

    elif(tail == "Левосторонний"):

        critical_value = calculate_critical_value_left_tail(h0, 0.1)
        power = calculate_power_left_tail(h1, critical_value)
        power_arr.append(power)
    else:   
        lower_critical_value, upper_critical_value = calculate_critical_values(h0, 0.1)
        power = calculate_power_two_tailed(h1, lower_critical_value, upper_critical_value)
        power_arr.append(power)

    return(power_arr)
