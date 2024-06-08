import numpy as np
from scipy.special import exp1
from scipy.special import gamma
from scipy import stats
import random
import numpy as np
from scipy.stats import anderson
import numpy as np
from scipy.stats import kstest, expon, norm
from math import factorial
import numpy as np
import time
from scipy.stats import norm
import numpy as np
from scipy.stats import expon, norm

#Константа для вычисление p
N = 1000

def phi_func(a, b):
    return 1 if a > b else 0

# Вспомогательные функции для вычисления p_value
def my_rvs(size):
    return np.random.exponential(size=size)

def p_value(sample, statistic, alternative):
    n = 1000
    return stats.monte_carlo_test(sample, rvs=my_rvs, statistic=statistic, vectorized=None, n_resamples=n, batch=None, alternative=alternative, axis=0)

def hypothesis(p):
    alpha = 0.1
    if (p > alpha):
        return "НЕ ОТКЛОНЯЕТСЯ"
    return "ОТКЛОНЯЕТСЯ"

# РЕАЛИЗАЦИЯ ТЕСТОВ
def shapiro_wilk(x):
    u = 0
    x = x - u
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    x_1 = x_sorted[0]
    W_e = n * (x_bar - x_1) * (x_bar - x_1) / (n - 1) / np.sum((x - x_bar) ** 2)
    return W_e

def shapiro_wilk_We0(x):
    u = 0
    x = x - u
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    x_1 = x_sorted[0]
    W_e0 = np.sum((x - x_bar) ** 2) / (np.sum(x) * np.sum(x))
    return W_e0

def frocini(x):
    n = len(x)
    x_bar = np.mean(x)
    B_n = 0
    x_sorted = np.sort(x)
    for i in range(1, n + 1):  # начинаем с 1
        B_n += np.abs(1 - np.exp(-x_sorted[i - 1] / x_bar) - (i - 0.5) / n)  # используем i - 1 для индексации x
    return B_n / np.sqrt(n)

def correlation_criterion_of_exp(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    eta_hat = n * (x_bar - x_sorted[0]) / (n - 1)
    lambda_hat = x_sorted[0] - eta_hat / n

    z = (x_sorted - lambda_hat) / eta_hat
    z_bar = np.mean(z)
    m = []
    for i in range(1, n + 1):
        a = 0
        for j in range(1, i + 1):
            a += 1 / ( n - j + 1)
        m.append(a)
    m_bar = np.mean(m)

    r = np.sum((z - z_bar) * (m - m_bar))  / (np.sqrt(np.sum((z - z_bar) ** 2) * np.sum((m -m_bar) ** 2)))
    return n * (1 - r * r)

def correlation_criterion_of_exp_ap(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    eta_hat = n * (x_bar - x_sorted[0]) / (n - 1)
    lambda_hat = x_sorted[0] - eta_hat / n

    z = (x_sorted - lambda_hat) / eta_hat
    z_bar = np.mean(z)
    m = []
    for i in range(1, n + 1):
        m.append(-np.log(1 - i / (n + 1)))
    m_bar = np.mean(m)

    r = np.sum((z - z_bar) * (m - m_bar))  / (np.sqrt(np.sum((z - z_bar) ** 2) * np.sum((m -m_bar) ** 2)))
    return n * (1 - r * r)

def kimber_michael(x):
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    d = []
    for i in range(n):
        s = 2 / np.pi * np.arcsin(np.sqrt(1 - np.exp(-x_sorted [i] / x_bar)))
        r = 2 / np.pi * np.arcsin(np.sqrt((i + 1 - 0.5) / n))
        d.append(np.abs(s - r))
    return max(d)

def fisher(x):
    n = len(x)
    return sum(x) / ((n - 1) * x[0])

def gnedenko(x, R):
    n = len(x)
    R = R * n
    R = int(R)
    x_sorted = np.sort(x)
    x_sorted = np.insert(x_sorted, 0, 0)
    D = []
    
    for i in range(1, n + 1):
        D.append((n - i + 1) * (x_sorted[i] - x_sorted[i-1]))
    return np.sum(D[:R]) / R / ( np.sum(D[R:]) / (n - R)) 

def harris(x, R):
    n = len(x)
    R = R * n
    R = int(R)
    x_sorted = np.sort(x)
    x_sorted = np.insert(x_sorted, 0, 0)
    D = []
    for i in range(1, n + 1):
        D.append((n - i + 1) * (x_sorted[i] - x_sorted[i-1]))
    return (np.sum(D[:R]) + np.sum(D[n - R:])) / (2 * R) / (np.sum(D[R:n-R]) / (n - 2 * R))

    R = 0.4
    n = len(x)
    R = R * n
    R = int(R)
    x_sorted = np.sort(x)
    x_sorted = np.insert(x_sorted, 0, 0)
    D = []
    for i in range(1, n + 1):
        D.append((n - i + 1) * (x_sorted[i] - x_sorted[i-1]))
    return (np.sum(D[:R]) + np.sum(D[n - R:])) / (2 * R) / (np.sum(D[R:n-R]) / (n - 2 * R))

# ЭТОТ КРИТЕРИЙ НЕ РЕАЛИЗОВАН В ISW 2.8
def bartlett(x):
    n = len(x)
    x_bar = np.mean(x)
    sum_ln_xi = np.sum(np.log(x))
    numerator = 2 * n * (np.log(x_bar) + sum_ln_xi / n)
    denominator = 1 + (n + 1) / (6 * n)
    return numerator / denominator

# ЭТОТ КРИТЕРИЙ НЕ РЕАЛИЗОВАН В ISW 2.9
def pietra(x):
    '''
    n = len(x)
    x_bar = np.mean(x)
    wn = (1 / (2 * n)) * sum(abs(xi - x_bar) for xi in x) / x_bar
    ch = wn - 0.3679 * (1 - 1/(2*n - 2)) 
    z = 0.2431 * np.sqrt(1/(n - 1)) * (1 - 0.605/(n - 1))
    V = ch / z
    return V - 0.0955* np.sqrt(1/(n - 1)) * (V * V - 1)
    '''
    n = len(x)
    xm = np.mean(x)
    t = np.sum(np.abs(x - xm)) / (2 * n * xm)
    return t

def epps_pulley(x):
    n = len(x)
    x_bar = np.mean(x)
    return np.sqrt(48 * n) * (np.sum(np.exp(-x / x_bar) / n) - 0.5)

def phi_func(a, b):
    return 1 if a > b else 0

def hollander_proshan(x):
    # тройный цикл супер долго
    n = len(x)
    T = 0
    x = np.sort(x)
    for i in range(n):
        for j in range(i):
            for k in range(j):
                T += phi_func(x[i], x[j] + x[k])
    ET = n * (n - 1) * (n - 2) / 8
    #модуль или не модуль
    DT = 3 * n * (n - 1) * (n - 2) / 2 * ( 5 * (n - 3) * (n - 4) / 2592 + 7 * (n - 3) / 432 + 1 / 48)
    return (T - ET) / np.sqrt(DT) 

def krit_naib_inter(x):
    n = len(x)
    x_bar = np.mean(x)
    x = np.insert(x, 0, 0)
    diff = np.diff(x)
    nm = np.max(diff) / np.sum(x)
    return nm
  
def kochar(X):
    n = len(X)
    sorted_X = np.sort(X)
    J_values = 2 * (1 - np.arange(1, n + 1) / (n + 1)) * (1 - np.log(1 - np.arange(1, n + 1) / (n + 1))) - 1
    numerator = np.sum(J_values * sorted_X)
    denominator = np.sum(X)
    T = np.sqrt((108 * n) / 17) * (numerator / denominator)
    return T
    
def klimko(x):
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    x_min = x_sorted[0]
    v = np.sqrt((1 / (n - 1)) * np.sum((x - x_bar) ** 2)) / (1 / n * np.sum(x - (x_min - 1 / (n * (n - 1)) * np.sum(x - x_min))))
    return np.sqrt(n) * ((v ** -1.075) - 1)

def greenwood(x):
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    return n * (np.sum(x * x)) / (np.sum(x) ** 2)

def epstein(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_sorted = np.insert(x_sorted, 0, 0)
    D = []
    for i in range(1, n + 1):
        D.append((n - i + 1) * (x_sorted[i] - x_sorted[i - 1]))
    num = 2 * n * (np.log(np.sum(D) / n) - np.sum(np.log(D)) / n)
    d = 1 + (n + 1) / (6 * n)
    return num / d

def phi_func(a, b):
    return 1 if a > b else 0

def deshpande(x, b):
    
    n = len(x)
    a = 0
    for i in range(n):
        for j in range(n):
            if(i != j):
                a += phi_func(x[i], b * x[j])
    J = a / (n * (n - 1))
    E = 1 / (b + 1)
    D = (1 + b/(b+2) + 1/(2*b+1) + 2*(1-b)/(b+1) - 2*b/(b**2+b+1) - 4/(b+1)**2) / n
    return (J - E) / np.sqrt(D) 
    
def ob_moran(x):
    x_bar = np.mean(x)
    s_x = np.prod(x)**(1/len(x))
    return s_x / x_bar

def moran(x):
    y = 0.577215
    n = len(x)
    x = np.sort(x)
    x_bar = np.mean(x)
    Tn = y +  np.sum(np.log(x / x_bar)) / n
    E = 0
    D = ((np.pi * np.pi / 6 - 1) ** 2 ) / n
    return (Tn - E) / np.sqrt(D)

def wong_and_wong(x):
    return max(x) / min(x)

def hegazy_green_T1(x):
    T1 = 0
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    for i in range(1, n + 1):
        T1 += np.abs(Y[i - 1] + np.log(1 - i / (n + 1)))
    T1 = T1 / n
    return T1

def hegazy_green_T2(x):
    T2 = 0
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    for i in range(1, n + 1):
        T2 += (Y[i - 1] + np.log(1 - i / (n + 1))) ** 2
    T2 = T2 / n
    return T2

def madukaife(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    M = 0
    for i in range(1, n + 1):
        M += (x_sorted[i - 1] / x_bar + np.log((n - i + 0.5) / n)) ** 2 
    return M

def lorenz(x, p):
    n = len(x)
    x_sorted = np.sort(x)
    return np.sum(x_sorted[:int(n * p)]) / np.sum(x_sorted)

def cox_oakes(x):
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    Y = x_sorted / x_bar
    CON = n + np.sum((1 - Y) * (np.log(Y)))
    return np.sqrt(6 / n) * (CON / np.pi)

def jackson(x):
    n = len(x)
    x_bar = np.mean(x)
    x_sorted = np.sort(x)
    Y = x_sorted / x_bar

    def t(j):
        a = 0
        for i in range(1, j + 1):
            a += 1 / (n - i + 1)
        return a
    a = 0
    for i in range(n):
        a += t(i + 1) * Y[i]
    return a / n

def gini(x):
    n = len(x)
    x_bar = np.mean(x)
    Y = x / x_bar
    abs_diff = np.abs(Y[:, np.newaxis] - Y)  #
    Gn = np.sum(abs_diff) / (2 * n * (n - 1))
    return ((12 * (n - 1)) ** 0.5) * (Gn - 0.5)


def atkinson(x, p):
    y = 0.577215
    n = len(x)
    x_bar = np.mean(x)
    
    if(p != 0 and p > -0.5 and p < 1):
        x_powered = np.power(x, p)
        atn = np.sqrt(n) * np.abs( ((np.sum(x_powered) / n) ** (1 / p)) / x_bar - gamma(1 + p) ** (1 / p)) 
        quantil = np.power(gamma(1 + p), 2 / p) * (gamma(1 + 2 * p) / (p * p * gamma(1 + p) * gamma(1 + p)) - 1 - 1 / ( p * p))
        return atn / np.sqrt(quantil)
    elif(p == 0):
        quantil = (np.pi * np.pi / 6 - 1) * np.exp(-2 * y)
        x_powered = np.power(x, 1 / n)
        atn = np.sqrt(n) * np.abs( np.prod(x_powered) / x_bar - np.exp(-y))
        return atn / np.sqrt(quantil)
    else:
        x_powered = np.power(x, p)
        return np.sqrt(n) * np.abs( ((np.sum(x_powered) / n) ** (1 / p)) / x_bar - gamma(1 + p) ** (1 / p))

def sadeghpour(x, r):
    
    x = np.sort(x)
    #x = np.insert(x, 0, 0)
    x_bar = np.mean(x)
    a = []
    n = len(x)
    for i in range(0, n- 1):
        a1 = np.power((n - i) / n , r)
        a2 = np.exp((r - 1) * x[i + 1] / x_bar) - np.exp((r - 1) * x[i] / x_bar)
        a.append(a1 * abs(a2))
    return np.sum(a)

def tiko(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_sorted = np.insert(x_sorted, 0, 0)
    D = []
    for i in range(1, n + 1):
        D.append((n - i + 1) * (x_sorted[i] - x_sorted[i-1]))

    tk = 0
    for i in range(1, n - 1):
        tk += (n - i + 1) * D[i] / ((n - 2) * D[i])   
    

    tk = (tk - 10 / (n * n) - 4 / n - 1) / (np.sqrt(0.333 / n + 0.667 / (n * n)))
    return tk

def klar(x , a):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    a1 = 2 * (3 * a + 2) * n / ((2 + a) * (1 + a) * (1 + a))
    a2 = 0
    a3 = 0
    for i in range(n):
        a2 += np.exp(-(1 + a) * Y[i]) / ((a + 1) * (a + 1))
        a3 += np.exp(-a * Y[i])
    a2 = -2 * a **3 * a2
    a3 = -2 / n * a3
    a4 = 0
    for i in range(n):
        for k in range(i + 1, n):
            a4 += (a * (Y[k] - Y[i]) - 2) * np.exp(-a * Y[i])
    a4 = 2 / n * a4
    return a1 + a2 + a3 + a4

def henze(x, a):
    '''
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar

    a1 = 0
    for j in range(n):
        for k in range(n):
            a1 += 1 / (Y[j] + Y[k] + a)
    b = 0
    for i in range(n):
        b += np.exp(Y[i] + a) * exp1(Y[i] + a)
    return a1 / n - b + n * (1 - a * np.exp(a) * exp1(a))'''
    
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    
    Y_exp_a = np.exp(Y + a)
    
    a1 = np.sum(1 / (np.outer(Y, np.ones(n)) + Y + a))
    b = np.sum(Y_exp_a * exp1(Y + a))
    
    return a1 / n - b + n * (1 - a * np.exp(a) * exp1(a))

def baringhaus_henze(x, a):
    x_bar = np.mean(x)
    Y = x / x_bar

    epsilon = 1e-10
    denom = Y[:, np.newaxis] + Y + a + epsilon  # Создание матрицы denom
    Bn = np.sum((1 - Y[:, np.newaxis]) * (1 - Y) / denom +
                (2 * Y[:, np.newaxis] * Y - Y[:, np.newaxis] - Y) / (denom**2) +
                2 * Y[:, np.newaxis] * Y / (denom**3))
    return Bn / len(x)

def henze_meintanis_L(x, a):
    n = len(x)
    x_bar = np.mean(x)
    Y = x / x_bar
    term1_sum = 0
    for j in range(n):
        for k in range(n):
            term1_sum += (1 + (Y[j] + Y[k] + a + 1) ** 2) / ((Y[j] + Y[k] + a) ** 3)

    term2_sum = 0
    for j in range(n):
        term2_sum += (1 + Y[j] + a) / (Y[j] + a) ** 2
    Ln_a = (1 / n) * term1_sum - 2 * term2_sum + n / a
    return Ln_a

def henze_meintanis_W1(x, a):

    n = len(x)
    x_bar = np.mean(x)
    Y = x / x_bar

    diffs = Y[:, None] - Y  # Efficiently calculate all pairwise differences
    Y_sum = Y[:, None] + Y

    term1 = 1 / (a**2 + diffs**2)
    term2 = 1 / (a**2 + Y_sum**2)
    term3 = Y_sum / ((a**2 + Y_sum**2)**2)
    term4 = (2 * a**2 - 6 * diffs**2) / ((a**2 + diffs**2)**3)
    term5 = (2 * a**2 - 6 * Y_sum**2) / ((a**2 + Y_sum**2)**3)


    W_1 = a / (2 * n) * (term1.sum() - term2.sum() - 4 * term3.sum() + term4.sum() + term5.sum())
    return W_1

def henze_meintanis_W2(x, a):
    n = len(x)
    x_bar = np.mean(x)
    Y = x / x_bar

    diffs = Y[:, None] - Y  # Efficiently calculate all pairwise differences
    Y_sum = Y[:, None] + Y

    term6 = np.exp(-diffs**2 / (4 * a)) * (1 + (2 * a - diffs**2) / (4 * a**2))
    term7 = np.exp(-Y_sum**2 / (4 * a)) * ((2 * a - Y_sum**2) / (4 * a**2) - Y_sum / a - 1)

    W_2 = (np.pi**0.5) / (4 * n * (a**0.5)) * (term6.sum() + term7.sum())
    return W_2

def fortiana_grane(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    l = []

    for i in range(1, n + 1):
        if( (n - i) == 0 or (n - i + 1) == 0):
            if ((n - i) == 0):
                a = 0
                a1 = 0
            else:
                if((n - i + 1) == 0):
                    b = 0
                    b1 = 0
        else:
            a = np.log(n - i)
            b = np.log(n - i + 1)
            a1 = n - i
            b1 = n - i + 1
        l.append(a1 * a - b1 * b + np.log(n))

    Q = 0
    for i in range(n):
        Q += l[i] * Y[i]
    return Q / n 
    
def montazeri_torabi(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    COV = 0
    for i in range(1, n + 1):
        COV += (2 * i - n - 1) * (1 - np.exp( -Y[i - 1]))
    return COV / (2 * n * n)

def torabi_h1(x):
    def h1(x):
        if(x >=0 and x < 1):
            return np.exp(x - 1) - x
        elif(x==1):
            return 0
        else:
            return (np.abs(x*x*x -1) ** (1 / 3))


    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    H1 = 0
    for i in range(1, n + 1):
        H1 = h1((2 - np.exp( -Y[i - 1])) / (1 + i / n))
    return H1 / n

def torabi_h2(x):
    def h2(x):
        if(x >=0 and x < 1):
            return np.exp(x - 1) - x
        elif(x==1):
            return 0
        else:
            return (x-1)**2 / (x+1)**2

    
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar
    H2 = 0
    for i in range(1, n + 1):
        H2 = h2((2 - np.exp( -Y[i - 1])) / (1 + i / n))
    return H2 / n

def integral_LL(x):
    n = len(x)
    x_sorted = np.sort(x)
    x_bar = np.mean(x)
    Y = x_sorted / x_bar 
    a = 0
    for i in range(n):
        for j in range(n):
            a += np.abs(Y[i] - Y[j]) - 1

    return 2 * (a / (n * n)) ** 2

def cramer_mises(x):
    n = len(x)
    y = x / np.mean(x)
    z = np.sort(1 - np.exp(-y))
    c = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    z = (z - c)**2
    t = 1 / (12 * n) + np.sum(z)
    return t

def cramer_mises_mrl(x):
    n = len(x)
    y = x / np.mean(x)
    z = np.sort(1 - np.exp(-y))
    c = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    z = (z - c)**2
    mrl = np.sqrt(np.mean(z))
    return mrl

def kolmogorov(x):
    n = len(x)
    lambda_hat = 1 / np.mean(x)
    F_n = np.arange(1, n + 1) / n
    F = expon.cdf(np.sort(x), scale=1/lambda_hat)
    D_n_plus = np.max(F_n - F)
    D_n_minus = np.max(F - (np.arange(0, n) / n))
    D_n = max(D_n_plus, D_n_minus)
    S_K = np.sqrt(n) * D_n + 1 / (6 * np.sqrt(n))
    return S_K

def kuper(x):
    n = len(x)
    lambda_hat = 1 / np.mean(x)
    F_n = np.arange(1, n + 1) / n
    F = expon.cdf(np.sort(x), scale=1/lambda_hat)
    D_n_plus = np.max(F_n - F)
    D_n_minus = np.max(F - (np.arange(0, n) / n))
    V_n = D_n_plus + D_n_minus
    return V_n

def ahsanullah(data):
    n = len(data)
    h = 0
    g = 0
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if abs(data[i] - data[j]) < data[k]:
                    h += 1
                if 2 * min(data[i], data[j]) < data[k]:
                    g += 1
                    
    r = (h - g) / (n**3)
    return r

def ahsanullah_p(x, simulate_p_value=True, nrepl=N):
    n = len(x)
    h = 0
    g = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if abs(x[i] - x[j]) < x[k]:
                    h += 1
                if 2 * min(x[i], x[j]) < x[k]:
                    g += 1
    r = (h - g) / (n ** 3)
    v = np.sqrt(n) * abs(r) / np.sqrt(647 / 4725)
    
    if simulate_p_value:
        l = 0
        for m in range(nrepl):
            z = np.random.exponential(size=n)
            h = 0
            g = 0
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if abs(z[i] - z[j]) < z[k]:
                            h += 1
                        if 2 * min(z[i], z[j]) < z[k]:
                            g += 1
            R = (h - g) / (n ** 3)
            if abs(R) > abs(r):
                l += 1
        p_value = l / nrepl
    else:
        p_value = 2 * (1 - norm.cdf(v))
    return p_value

def rossberg(data):
    n = len(data)
    s = 0
    sh = 0
    sg = 0

    for m in range(n):
        h = 0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if ((data[i] + data[j] + data[k] - 2 * min(data[i], data[j], data[k]) - max(data[i], data[j], data[k])) < data[m]):
                        h += 1
        h = ((6 * factorial(n - 3)) / factorial(n)) * h
        sh += h

    for m in range(n):
        g = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if min(data[i], data[j]) < data[m]:
                    g += 1
        g = ((2 * factorial(n - 2)) / factorial(n)) * g
        sg += g

    s = sh - sg
    s = s / n 
    return s

def rossberg_p(x):
    n = len(x)
    s = 0
    sh = 0
    sg = 0
    for m in range(n):
        h = 0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if (x[i] + x[j] + x[k] - 2 * min(x[i], x[j], x[k]) - max(x[i], x[j], x[k])) < x[m]:
                        h += 1
        h = ((6 * factorial(n - 3)) / factorial(n)) * h
        sh += h
    
    for m in range(n):
        g = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if min(x[i], x[j]) < x[m]:
                    g += 1
        g = ((2 * factorial(n - 2)) / factorial(n)) * g
        sg += g
    
    s = sh - sg
    s /= n
    v = np.sqrt(n) * np.abs(s) / np.sqrt(52 / 1125)
    p_value = 2 * (1 - norm.cdf(v))
    
    return p_value

def criterion_is(name):
    # двусторонний if( abs(a) >= abs(t_stat)):
    # правосторонний if( a >= t_stat):
    # левосторонний if( a <= t_stat):
    if name == "Крит.показ-ти Андерсона-Дарлинга":
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.25)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.75)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.99)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.25)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.75)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.99)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Садепура r 2"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(0.1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(1.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(10)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(2.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Барингхауса-Хензе(5)"):
        return "Правосторонний"
    if(name == "Корреляционный крит.показ-ти"):
        return "Правосторонний"
    if(name == "Корреляционный крит.показ-ти аппроксимация"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Кокса-Оукса"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Крамера-Мизеса"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Крамера-Мизеса-Смирнова MRL"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Дешпанде(0.1)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.2)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.3)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.4)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.44)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.6)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.7)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.8)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Дешпанде(0.9)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Ибрагими"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Эппса-Палли"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Эпштейна"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Фишера"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Фортиана и Гране"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Фроцини"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Джини"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Гнеденко(0.1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.2)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.3)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.4)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.6)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.7)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.8)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гнеденко(0.9)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Гринвуда"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Харриса(0.1)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Харриса(0.2)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Харриса(0.25)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Харриса(0.3)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Харриса(0.4)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хегази-Грина T1"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хегази-Грина T2"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(0.025)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе(0.1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(1.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(2.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе(5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.75)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (1.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (2.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.75)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (2.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.75)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (2.5)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Холландера-Прошана"):
        return "Двусторонний"
    if(name == "Крит.показ-ти L2"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Джексона"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Климко-Антла"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Кимбера-Мичела"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Клара(1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Клара(10)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Кочара"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Колмогорова-Смирнова"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Купера"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Мадукайфе"):
        return "Правосторонний"
    if(name == "Крит.показ-ти наибольшего интервала"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Монтазери и Тораби"):
        return "Левосторонний"
    if(name == "Крит.показ-ти Морана(норм)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Лоуренса(0.1)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Лоуренса(0.25)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Лоуренса(0.5)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Лоуренса(0.75)"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Лоуренса(0.9)"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Шапиро-Уилка"):
        return "Левосторонний"
    if(name == "Крит.показ-ти Шапиро-Уилка We0"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Шермана/Пиэтра"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Тико"):
        return "Двусторонний"
    if(name == "Крит.показ-ти Тораби1"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Тораби2"):
        return "Правосторонний"
    if(name == "Крит.показ-ти Вонга-Вонга"):
        return "Двусторонний"
    if name == "Крит.показ-ти Ахсануллаха":
        return "Двусторонний"
    if name == "Крит.показ-ти Россберга":
        return "Двусторонний"

def t_stats(criterion_name, data):
    if criterion_name == "Крит.показ-ти Андерсона-Дарлинга":
        return anderson(data, dist='expon').statistic
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(-0.25)":
        return atkinson(data, -0.25)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(-0.5)":
        return atkinson(data, -0.5)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(-0.75)":
        return atkinson(data, -0.75)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(-0.99)":
        return atkinson(data, -0.99)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(0)":
        return atkinson(data, 0)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(0.25)":
        return atkinson(data, 0.25)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(0.5)":
        return atkinson(data, 0.5)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(0.75)":
        return atkinson(data, 0.75)
    elif criterion_name == "Крит.показ-ти Аткинсона ПолуНорм(0.99)":
        return atkinson(data, 0.99)
    elif criterion_name == "Крит.показ-ти Садепура r = 2":
        return sadeghpour(data, 2)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(0.1)":
        return baringhaus_henze(data, 0.1)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(0.5)":
        return baringhaus_henze(data, 0.5)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(1)":
        return baringhaus_henze(data, 1)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(1.5)":
        return baringhaus_henze(data, 1.5)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(10)":
        return baringhaus_henze(data, 10)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(2.5)":
        return baringhaus_henze(data, 2.5)
    elif criterion_name == "Крит.показ-ти Барингхауса-Хензе(5)":
        return baringhaus_henze(data, 5)
    elif criterion_name == "Корреляционный крит.показ-ти":
        return correlation_criterion_of_exp(data)
    elif criterion_name == "Корреляционный крит.показ-ти аппроксимация":
        return correlation_criterion_of_exp_ap(data)
    elif criterion_name == "Крит.показ-ти Кокса-Оукса":
        return cox_oakes(data)
    elif criterion_name == "Крит.показ-ти Крамера-Мизеса":
        return cramer_mises(data)
    elif criterion_name == "Крит.показ-ти Крамера-Мизеса-Смирнова MRL":
        return cramer_mises_mrl(data)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.1)":
        return deshpande(data, 0.1)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.2)":
        return deshpande(data, 0.2)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.3)":
        return deshpande(data, 0.3)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.4)":
        return deshpande(data, 0.4)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.44)":
        return deshpande(data, 0.44)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.5)":
        return deshpande(data, 0.5)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.6)":
        return deshpande(data, 0.6)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.7)":
        return deshpande(data, 0.7)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.8)":
        return deshpande(data, 0.8)
    elif criterion_name == "Крит.показ-ти Дешпанде(0.9)":
        return deshpande(data, 0.9)
    elif criterion_name == "Крит.показ-ти Эппса-Палли":
        return epps_pulley(data)
    elif criterion_name == "Крит.показ-ти Эпштейна":
        return epstein(data)
    elif criterion_name == "Крит.показ-ти Фишера":
        return fisher(data)
    elif criterion_name == "Крит.показ-ти Фортиана и Гране":
        return fortiana_grane(data)
    elif criterion_name == "Крит.показ-ти Фроцини":
        return frocini(data)
    elif criterion_name == "Крит.показ-ти Джини":
        return gini(data)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.1)":
        return gnedenko(data, 0.1)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.2)":
        return gnedenko(data, 0.2)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.3)":
        return gnedenko(data, 0.3)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.4)":
        return gnedenko(data, 0.4)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.5)":
        return gnedenko(data, 0.5)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.6)":
        return gnedenko(data, 0.6)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.7)":
        return gnedenko(data, 0.7)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.8)":
        return gnedenko(data, 0.8)
    elif criterion_name == "Крит.показ-ти Гнеденко(0.9)":
        return gnedenko(data, 0.9)
    elif criterion_name == "Крит.показ-ти Гринвуда":
        return greenwood(data)
    elif criterion_name == "Крит.показ-ти Харриса(0.1)":
        return harris(data, 0.1)
    elif criterion_name == "Крит.показ-ти Харриса(0.2)":
        return harris(data, 0.2)
    elif criterion_name == "Крит.показ-ти Харриса(0.25)":
        return harris(data, 0.25)
    elif criterion_name == "Крит.показ-ти Харриса(0.3)":
        return harris(data, 0.3)
    elif criterion_name == "Крит.показ-ти Харриса(0.4)":
        return harris(data, 0.4)
    elif criterion_name == "Крит.показ-ти Хегази-Грина T1":
        return hegazy_green_T1(data)
    elif criterion_name == "Крит.показ-ти Хегази-Грина T2":
        return hegazy_green_T2(data)
    elif criterion_name == "Крит.показ-ти Хензе(0.025)":
        return henze(data, 0.025)
    elif criterion_name == "Крит.показ-ти Хензе(0.1)":
        return -henze(data, 0.1)
    elif criterion_name == "Крит.показ-ти Хензе(0.5)":
        return -henze(data, 0.5)
    elif criterion_name == "Крит.показ-ти Хензе(1)":
        return -henze(data, 1)
    elif criterion_name == "Крит.показ-ти Хензе(1.5)":
        return henze(data, 1.5)
    elif criterion_name == "Крит.показ-ти Хензе(2.5)":
        return henze(data, 2.5)
    elif criterion_name == "Крит.показ-ти Хензе(5)":
        return henze(data, 5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (0.1)":
        return henze_meintanis_L(data, 0.1)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (0.5)":
        return henze_meintanis_L(data, 0.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (0.75)":
        return henze_meintanis_L(data, 0.75)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (1)":
        return henze_meintanis_L(data, 1)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (1.5)":
        return henze_meintanis_L(data, 1.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (2.5)":
        return henze_meintanis_L(data, 2.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса L (5)":
        return -henze_meintanis_L(data, 5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.5)":
        return henze_meintanis_W1(data, 0.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.75)":
        return henze_meintanis_W1(data, 0.75)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1)":
        return henze_meintanis_W1(data, 1)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1.5)":
        return henze_meintanis_W1(data, 1.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W1 (2.5)":
        return henze_meintanis_W1(data, 2.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.5)":
        return henze_meintanis_W2(data, 0.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.75)":
        return henze_meintanis_W2(data, 0.75)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1)":
        return henze_meintanis_W2(data, 1)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1.5)":
        return henze_meintanis_W2(data, 1.5)
    elif criterion_name == "Крит.показ-ти Хензе-Мейнтаниса W2 (2.5)":
        return henze_meintanis_W2(data, 2.5)
    elif criterion_name == "Крит.показ-ти Холландера-Прошана":
        return hollander_proshan(data)
    elif criterion_name == "Крит.показ-ти L2":
        return integral_LL(data)
    elif criterion_name == "Крит.показ-ти Джексона":
        return jackson(data)
    elif criterion_name == "Крит.показ-ти Климко-Антла":
        return klimko(data)
    elif criterion_name == "Крит.показ-ти Кимбера-Мичела":
        return kimber_michael(data)
    elif criterion_name == "Крит.показ-ти Клара(1)":
        return klar(data, 1)
    elif criterion_name == "Крит.показ-ти Клара(10)":
        return klar(data, 10)
    elif criterion_name == "Крит.показ-ти Кочара":
        return kochar(data)
    elif criterion_name == "Крит.показ-ти Колмогорова-Смирнова":
        return kolmogorov(data)
    elif criterion_name == "Крит.показ-ти Купера":
        return kuper(data)
    elif criterion_name == "Крит.показ-ти Мадукайфе":
        return madukaife(data)
    elif criterion_name == "Крит.показ-ти наибольшего интервала":
        return krit_naib_inter(data)
    elif criterion_name == "Крит.показ-ти Монтазери и Тораби":
        return montazeri_torabi(data)
    elif criterion_name == "Крит.показ-ти Морана(норм)":
        return moran(data)
    elif criterion_name == "Крит.показ-ти Лоуренса(0.1)":
        return lorenz(data, 0.1)
    elif criterion_name == "Крит.показ-ти Лоуренса(0.25)":
        return lorenz(data, 0.25)
    elif criterion_name == "Крит.показ-ти Лоуренса(0.5)":
        return lorenz(data, 0.5)
    elif criterion_name == "Крит.показ-ти Лоуренса(0.75)":
        return lorenz(data, 0.75)
    elif criterion_name == "Крит.показ-ти Лоуренса(0.9)":
        return lorenz(data, 0.9)
    elif criterion_name == "Крит.показ-ти Шапиро-Уилка":
        return shapiro_wilk(data)
    elif criterion_name == "Крит.показ-ти Шапиро-Уилка We0":
        return shapiro_wilk_We0(data)
    elif criterion_name == "Крит.показ-ти Шермана/Пиэтра":
        return pietra(data)
    elif criterion_name == "Крит.показ-ти Вонга-Вонга":
        return wong_and_wong(data)
    elif criterion_name == "Крит.показ-ти Ахсануллаха":
        return ahsanullah(data)
    elif criterion_name == "Крит.показ-ти Россберга":
        return rossberg(data)

# size - размер выборок
def p_value(t_stat, name, size):
    # двухсторонний if( abs(a) >= abs(t_stat)):
    # правосторонний if( a >= t_stat):
    # левосторонний if( a <= t_stat):
    global N
    samples = np.random.exponential(scale=1.0, size=(N, size))
    
    if name == "Крит.показ-ти Андерсона-Дарлинга":
        count = 0
        for i in range(N):
            a = anderson(samples[i], dist='expon')
            critical_value = a.critical_values[4]
            t_stat = a.statistic

            if a.statistic >= critical_value:
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.25)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], -0.25)
            if( a >= t_stat):
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.5)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], -0.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.75)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], -0.75)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(-0.99)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], -0.99)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], 0)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.25)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], 0.25)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.5)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], 0.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.75)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], 0.75)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Аткинсона ПолуНорм(0.99)"):
        count = 0
        for i in range(N):
            a = atkinson(samples[i], 0.99)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Садепура r = 2"):
        count = 0
        for i in range(N):
            a = sadeghpour(samples[i], 2)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(0.1)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 0.1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(0.5)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 0.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(1)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(1.5)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 1.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(10)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 10)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(2.5)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 2.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Барингхауса-Хензе(5)"):
        count = 0
        for i in range(N):
            a = baringhaus_henze(samples[i], 5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Корреляционный крит.показ-ти"):
        count = 0
        for i in range(N):
            a = correlation_criterion_of_exp(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Корреляционный крит.показ-ти аппроксимация"):
        count = 0
        for i in range(N):
            a = correlation_criterion_of_exp_ap(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Кокса-Оукса"):
        count = 0
        for i in range(N):
            a = cox_oakes(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Крамера-Мизеса"):
        count = 0
        for i in range(N):
            a = cramer_mises(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Крамера-Мизеса-Смирнова MRL"):
        count = 0
        for i in range(N):
            a = cramer_mises_mrl(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N

    if(name == "Крит.показ-ти Заманзаде"):
        #count = 0
        #for i in range(N):
            #a = 0
           # if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти Дешпанде(0.1)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.1)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.2)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.2)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.3)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.3)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.4)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.4)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.44)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.44)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.5)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.6)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.6)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.7)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.7)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.8)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.8)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Дешпанде(0.9)"):
        count = 0
        for i in range(N):
            a = deshpande(samples[i], 0.9)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Ибрагими"):
        count = 0
        for i in range(N):
            a = 0
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N        
    if(name == "Крит.показ-ти Эппса-Палли"):
        count = 0
        for i in range(N):
            a = epps_pulley(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Эпштейна"):
        count = 0
        for i in range(N):
            a = epstein(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Фишера"):
        count = 0
        for i in range(N):
            a = fisher(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Фортиана и Гране"):
        count = 0
        for i in range(N):
            a = fortiana_grane(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Фроцини"):
        count = 0
        for i in range(N):
            a = frocini(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Джини"):
        count = 0
        for i in range(N):
            a = gini(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.1)"):
        count = 0
        # двухсторонний if( abs(a) >= abs(t_stat)):
        # правосторонний if( a >= t_stat):
        # левосторонний if( a <= t_stat):
        for i in range(N):
            a = gnedenko(samples[i], 0.1)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.2)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.2)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.3)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.3)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.4)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.4)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.5)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.5)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.6)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.6)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.7)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.7)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.8)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.8)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гнеденко(0.9)"):
        count = 0
        for i in range(N):
            a = gnedenko(samples[i], 0.9)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Гринвуда"):
        count = 0
        for i in range(N):
            a = greenwood(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Харриса(0.1)"):
        count = 0
        for i in range(N):
            a = harris(samples[i], 0.1)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Харриса(0.2)"):
        count = 0
        for i in range(N):
            a = harris(samples[i], 0.2)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Харриса(0.25)"):
        count = 0
        for i in range(N):
            a = harris(samples[i], 0.25)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Харриса(0.3)"):
        count = 0
        for i in range(N):
            a = harris(samples[i], 0.3)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Харриса(0.4)"):
        count = 0
        for i in range(N):
            a = harris(samples[i], 0.4)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хегази-Грина T1"):
        count = 0
        for i in range(N):
            a = hegazy_green_T1(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хегази-Грина T2"):
        count = 0
        for i in range(N):
            a = hegazy_green_T2(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(0.025)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 0.025)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(0.1)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 0.1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(0.5)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 0.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(1)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(1.5)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 1.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(2.5)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 2.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе(5)"):
        count = 0
        for i in range(N):
            a = henze(samples[i], 5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.1)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 0.1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 0.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (0.75)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 0.75)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (1)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (1.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 1.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (2.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 2.5)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса L (5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_L(samples[i], 5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Хензе-Мейнтаниса T1 (1.5)"):
        count = 0
        #for i in range(N):
           # a = 0
           # if( abs(a) >= abs(t_stat)):
            #    count = count + 1
        return -1
    if(name == "Крит.показ-ти Хензе-Мейнтаниса T1 (2.5)"):
        count = 0
        #for i in range(N):
           # a = 0
            #if( abs(a) >= abs(t_stat)):
               # count = count + 1
        return -1
    if(name == "Крит.показ-ти Хензе-Мейнтаниса T2 (1.5)"):
        count = 0
        #for i in range(N):
        #    a = 0
        #    if( abs(a) >= abs(t_stat)):
         #       count = count + 1
        return -1
    if(name == "Крит.показ-ти Хензе-Мейнтаниса T2 (2.5)"):
        count = 0
       # for i in range(N):
           # a = 0
            #if( abs(a) >= abs(t_stat)):
             #   count = count + 1
        return -1
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W1(samples[i], 0.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (0.75)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W1(samples[i], 0.75)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W1(samples[i], 1)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (1.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W1(samples[i], 1.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W1 (2.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W1(samples[i], 2.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W2(samples[i], 0.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (0.75)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W2(samples[i], 0.75)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W2(samples[i], 1)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (1.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W2(samples[i], 1.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Хензе-Мейнтаниса W2 (2.5)"):
        count = 0
        for i in range(N):
            a = henze_meintanis_W2(samples[i], 2.5)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Холландера-Прошана"):
        count = 0
        for i in range(N):
            a = hollander_proshan(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти L2"):
        count = 0
        for i in range(N):
            a = integral_LL(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    
    if(name == "Крит.показ-ти Джексона"):
        count = 0
        for i in range(N):
            a = jackson(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Климко-Антла"):
        count = 0
        for i in range(N):
            a = klimko(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти кернеел"):
        count = 0
        for i in range(N):
            a = 0
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Кимбера-Мичела"):
        count = 0
        for i in range(N):
            a = kimber_michael(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Клара(1)"):
        count = 0
        for i in range(N):
            a = klar(samples[i], 1)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Клара(10)"):
        count = 0
        for i in range(N):
            a = klar(samples[i], 10)
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Кочара"):
        count = 0
        for i in range(N):
            a = kochar(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N

    if(name == "Крит.показ-ти Колмогорова-Смирнова"):
        count = 0
        for i in range(N):
            a = kolmogorov(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Купера"):
        count = 0
        for i in range(N):
            a = kuper(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоулесса"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти Мадукайфе"):
        count = 0
        for i in range(N):
            a = madukaife(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти наибольшего интервала"):
        count = 0
        for i in range(N):
            a = krit_naib_inter(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Монтазери и Тораби"):
        count = 0
        for i in range(N):
            a = montazeri_torabi(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Морана(норм)"):
        count = 0
        for i in range(N):
            a = moran(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоуренса(0.1)"):
        count = 0
        for i in range(N):
            a = lorenz(samples[i], 0.1)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоуренса(0.25)"):
        count = 0
        for i in range(N):
            a = lorenz(samples[i], 0.25)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоуренса(0.5)"):
        count = 0
        for i in range(N):
            a = lorenz(samples[i], 0.5)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоуренса(0.75)"):
        count = 0
        for i in range(N):
            a = lorenz(samples[i], 0.75)
            if( a <= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Лоуренса(0.9)"):
        count = 0
        for i in range(N):
            a = lorenz(samples[i], 0.9)
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Шапиро-Уилка"):
        count = 0
        for i in range(N):
            a = shapiro_wilk(samples[i])
            if( a < t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Шапиро-Уилка We0"):
        count = 0
        for i in range(N):
            a = shapiro_wilk_We0(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Шермана/Пиэтра"):
        count = 0
        for i in range(N):
            a = pietra(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Sn(осн.на Gini)"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти Тико"):
        count = 0
        for i in range(N):
            a = tiko(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Тораби1"):
        count = 0
        for i in range(N):
            a = torabi_h1(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Тораби2"):
        count = 0
        for i in range(N):
            a = torabi_h2(samples[i])
            if( a >= t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти U1"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти U2"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти N2"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти Ватсона"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1
    if(name == "Крит.показ-ти Вонга-Вонга"):
        count = 0
        for i in range(N):
            a = wong_and_wong(samples[i])
            if( abs(a) >= abs(t_stat)):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Жанга Za"):
        count = 0
        #for i in range(N):
            #a = 0
            #if( abs(a) >= abs(t_stat)):
                #count = count + 1
        return -1


    if(name == "Крит.показ-ти Ахсануллаха"):
        count = 0
        for i in range(N):
            a = ahsanullah(samples[i])
            if( a < t_stat):
                count = count + 1
        return count / N
    if(name == "Крит.показ-ти Россберга"):
        count = 0
        for i in range(N):
            a = rossberg(samples[i])
            if( a < t_stat):
                count = count + 1
        return count / N

def test(name, data):
    global N_model
    t = t_stats(name, data)
    p = p_value(t, name, len(data))
    line = hypothesis(p)
    return t, p, line
    
names = ["Крит.показ-ти Андерсона-Дарлинга",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.25)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.5)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.75)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.99)",
         "Крит.показ-ти Аткинсона ПолуНорм(0)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.25)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.5)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.75)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.99)",
         "Крит.показ-ти Садепура r = 2",
         "Крит.показ-ти Барингхауса-Хензе(0.1)",
         "Крит.показ-ти Барингхауса-Хензе(0.5)",
         "Крит.показ-ти Барингхауса-Хензе(1)",
         "Крит.показ-ти Барингхауса-Хензе(1.5)",
         "Крит.показ-ти Барингхауса-Хензе(10)",
         "Крит.показ-ти Барингхауса-Хензе(2.5)",
         "Крит.показ-ти Барингхауса-Хензе(5)",
         "Корреляционный крит.показ-ти",
         "Корреляционный крит.показ-ти аппроксимация",
         "Крит.показ-ти Кокса-Оукса",
         "Крит.показ-ти Крамера-Мизеса",
         "Крит.показ-ти Крамера-Мизеса-Смирнова MRL",
         #"Крит.показ-ти Заманзаде",
         "Крит.показ-ти Дешпанде(0.1)",
         "Крит.показ-ти Дешпанде(0.2)",
         "Крит.показ-ти Дешпанде(0.3)",
         "Крит.показ-ти Дешпанде(0.4)",
         "Крит.показ-ти Дешпанде(0.44)",
         "Крит.показ-ти Дешпанде(0.5)",
         "Крит.показ-ти Дешпанде(0.6)",
         "Крит.показ-ти Дешпанде(0.7)",
         "Крит.показ-ти Дешпанде(0.8)",
         "Крит.показ-ти Дешпанде(0.9)",
         #"Крит.показ-ти Ибрагими",
         "Крит.показ-ти Эппса-Палли",
         "Крит.показ-ти Эпштейна",
         "Крит.показ-ти Фишера",
         "Крит.показ-ти Фортиана и Гране",
         "Крит.показ-ти Фроцини",
         "Крит.показ-ти Джини",
         "Крит.показ-ти Гнеденко(0.1)",
         "Крит.показ-ти Гнеденко(0.2)",
         "Крит.показ-ти Гнеденко(0.3)",
         "Крит.показ-ти Гнеденко(0.4)",
         "Крит.показ-ти Гнеденко(0.5)",
         "Крит.показ-ти Гнеденко(0.6)",
         "Крит.показ-ти Гнеденко(0.7)",
         "Крит.показ-ти Гнеденко(0.8)",
         "Крит.показ-ти Гнеденко(0.9)",
         "Крит.показ-ти Гринвуда",
         "Крит.показ-ти Харриса(0.1)",
         "Крит.показ-ти Харриса(0.2)",
         "Крит.показ-ти Харриса(0.25)",
         "Крит.показ-ти Харриса(0.3)",
         "Крит.показ-ти Харриса(0.4)",
         "Крит.показ-ти Хегази-Грина T1",
         "Крит.показ-ти Хегази-Грина T2",
         "Крит.показ-ти Хензе(0.025)",
         "Крит.показ-ти Хензе(0.1)",
         "Крит.показ-ти Хензе(0.5)",
         "Крит.показ-ти Хензе(1)",
         "Крит.показ-ти Хензе(1.5)",
         "Крит.показ-ти Хензе(2.5)",
         "Крит.показ-ти Хензе(5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.1)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса L (1)",
         "Крит.показ-ти Хензе-Мейнтаниса L (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (5)",
         "Крит.показ-ти Хензе-Мейнтаниса T1 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса T1 (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса T2 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса T2 (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (1)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (1)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (2.5)",
         "Крит.показ-ти Холландера-Прошана",
         "Крит.показ-ти L2",
         "Крит.показ-ти Джексона",
         "Крит.показ-ти Климко-Антла",
         #"Крит.показ-ти кернеел",
         "Крит.показ-ти Кимбера-Мичела",
         "Крит.показ-ти Клара(1)",
         "Крит.показ-ти Клара(10)",
         "Крит.показ-ти Кочара",
         #"Крит.показ-ти Колмогоровоа мрл",
         "Крит.показ-ти Колмогорова-Смирнова",
         "Крит.показ-ти Купера",
         #Крит.показ-ти Лоулесса",
         "Крит.показ-ти Мадукайфе",
         "Крит.показ-ти наибольшего интервала",
         "Крит.показ-ти Монтазери и Тораби",
         "Крит.показ-ти Морана(норм)",
         "Крит.показ-ти Лоуренса(0.1)",
         "Крит.показ-ти Лоуренса(0.25)",
         "Крит.показ-ти Лоуренса(0.5)",
         "Крит.показ-ти Лоуренса(0.75)",
         "Крит.показ-ти Лоуренса(0.9)",
         "Крит.показ-ти Шапиро-Уилка",
         "Крит.показ-ти Шапиро-Уилка We0",
         "Крит.показ-ти Шермана/Пиэтра",
         #"Крит.показ-ти Sn(осн.на Gini)",
         #"Крит.показ-ти Тико",
         #"Крит.показ-ти Тораби1",
         #"Крит.показ-ти Тораби2",
         #"Крит.показ-ти U1",
         #"Крит.показ-ти U2",
         #"Крит.показ-ти N2",
         #"Крит.показ-ти Ватсона",
         "Крит.показ-ти Вонга-Вонга",
         #"Крит.показ-ти Жанга Za",
         "Крит.показ-ти Ахсануллаха",
         "Крит.показ-ти Россберга"
         
]

sample = np.loadtxt("Вей (0.8000,1.0000,0.0000).dat")  # пример данных


'''
for i in range (len(names)): 
    start_time = time.time()
    a = test(names[i], sample)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(i, elapsed_time, names[i], a)
'''

def zhang_test(data, alpha=0.05):
    # Упорядочить данные по возрастанию
    data_sorted = np.sort(data)
    n = len(data)
    
    # Оценка параметра lambda экспоненциального распределения
    lambda_hat = 1 / np.mean(data)
    
    # Эмпирическая функция распределения (EDF)
    F_n = np.arange(1, n + 1) / n
    
    # Теоретическая функция распределения экспоненциального распределения
    F_exp = expon.cdf(data_sorted, scale=1/lambda_hat)
    
    # Рассчитать тестовую статистику Жанга
    zhang_statistic = np.max(np.abs(F_n - F_exp))
    
    # Критическое значение для уровня значимости alpha
    critical_value = np.sqrt(-np.log(alpha / 2) / (2 * n))
    
    # Проверка гипотезы
    reject_null = zhang_statistic > critical_value
    
    return zhang_statistic, critical_value, reject_null


