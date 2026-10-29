"""ZDT and DTLZ benchmark definitions as referenced in Section 4.1 of the paper."""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Benchmark:
    name: str
    function: object
    n_dim: int
    n_obj: int
    lower_bound: float
    upper_bound: float

def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    return np.array([f1, g * (1.0 - np.sqrt(f1 / g))])

def zdt2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    return np.array([f1, g * (1.0 - (f1 / g) ** 2)])

def zdt3(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    ratio = f1 / g
    return np.array([f1, g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))])

def zdt4(x):
    f1 = x[0]
    tail = x[1:]
    g = 1.0 + 10.0 * len(tail) + np.sum(tail**2 - 10.0 * np.cos(4.0 * np.pi * tail))
    return np.array([f1, g * (1.0 - np.sqrt(f1 / g))])

def zdt6(x):
    f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
    g = 1.0 + 9.0 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
    return np.array([f1, g * (1.0 - (f1 / g) ** 2)])

def _dtlz_g(x, m):
    return np.sum((x[m - 1:] - 0.5) ** 2)

def _dtlz1(x, m):
    k = len(x) - m + 1
    tail = x[m - 1:]
    g = 100.0 * (k + np.sum((tail - 0.5) ** 2 - np.cos(20.0 * np.pi * (tail - 0.5))))
    f = []
    for i in range(m):
        value = 0.5 * (1.0 + g)
        for j in range(m - i - 1):
            value *= x[j]
        if i:
            value *= 1.0 - x[m - i - 1]
        f.append(value)
    return np.asarray(f)

def _dtlz2(x, m, alpha=1.0):
    g = _dtlz_g(x, m)
    y = x[:m - 1] ** alpha
    f = []
    for i in range(m):
        value = 1.0 + g
        for j in range(m - i - 1):
            value *= np.cos(y[j] * np.pi / 2.0)
        if i:
            value *= np.sin(y[m - i - 1] * np.pi / 2.0)
        f.append(value)
    return np.asarray(f)

def _dtlz3(x, m):
    k = len(x) - m + 1
    tail = x[m - 1:]
    g = 100.0 * (k + np.sum((tail - 0.5) ** 2 - np.cos(20.0 * np.pi * (tail - 0.5))))
    y = x[:m - 1]
    f = []
    for i in range(m):
        value = 1.0 + g
        for j in range(m - i - 1):
            value *= np.cos(y[j] * np.pi / 2.0)
        if i:
            value *= np.sin(y[m - i - 1] * np.pi / 2.0)
        f.append(value)
    return np.asarray(f)

def _dtlz5_or_6(x, m, use_six=False):
    tail = x[m - 1:]
    if use_six:
        g = np.sum(tail**0.1)
    else:
        g = np.sum((tail - 0.5) ** 2)
    theta = np.empty(m - 1)
    theta[0] = x[0] * np.pi / 2.0
    if m > 2:
        theta[1:] = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[1:m - 1])
    f = []
    for i in range(m):
        value = 1.0 + g
        for j in range(m - i - 1):
            value *= np.cos(theta[j])
        if i:
            value *= np.sin(theta[m - i - 1])
        f.append(value)
    return np.asarray(f)

def _dtlz7(x, m):
    k = len(x) - m + 1
    g = 1.0 + 9.0 * np.mean(x[m - 1:])
    f = list(x[:m - 1])
    h = m - sum((fi / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * fi)) for fi in f)
    return np.asarray(f + [(1.0 + g) * h])

def make_benchmarks(n_obj=3):
    zdt = [
        Benchmark("ZDT1", zdt1, 30, 2, 0.0, 1.0),
        Benchmark("ZDT2", zdt2, 30, 2, 0.0, 1.0),
        Benchmark("ZDT3", zdt3, 30, 2, 0.0, 1.0),
        Benchmark("ZDT4", zdt4, 10, 2, np.array([0.0] + [-5.0] * 9), np.array([1.0] + [5.0] * 9)),
        Benchmark("ZDT6", zdt6, 10, 2, 0.0, 1.0),
    ]
    dtlz_m = n_obj
    dtlz_n = n_obj + 4  # k=5, standard Deb 2002 (n = m + k - 1 = m + 4)
    dtlz = [
        Benchmark("DTLZ1", lambda x: _dtlz1(x, dtlz_m), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ2", lambda x: _dtlz2(x, dtlz_m), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ3", lambda x: _dtlz3(x, dtlz_m), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ4", lambda x: _dtlz2(x, dtlz_m, alpha=100.0), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ5", lambda x: _dtlz5_or_6(x, dtlz_m), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ6", lambda x: _dtlz5_or_6(x, dtlz_m, use_six=True), dtlz_n, dtlz_m, 0.0, 1.0),
        Benchmark("DTLZ7", lambda x: _dtlz7(x, dtlz_m), dtlz_n, dtlz_m, 0.0, 1.0),
    ]
    return zdt + dtlz