import math
import numpy as np
import matplotlib.pyplot as plt

import constants as c


def runge_kutta_numerical_integration(dt, E_0, y_0, cycles, gen_graph=False):
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    num_of_time_intervals = cycles * math.ceil(T / dt)

    rz = np.zeros(num_of_time_intervals)
    ry = np.zeros(num_of_time_intervals)
    vz = np.zeros(num_of_time_intervals)
    vy = np.zeros(num_of_time_intervals)

    rz[0] = 0
    print(math.sqrt((2 * E_0)/c.m))
    ry[0] = y_0
    vz[0] = math.sqrt((2 * E_0)/c.m)
    vy[0] = 0

    def az(vy):
        return (c.q * vy * c.B) / c.m

    def ay(vz):
        return (c.q * c.E - c.q * c.B * vz) / c.m

    print(num_of_time_intervals)
    for i in range(1, num_of_time_intervals):
        rz[i] = rz[i - 1] + vz[i - 1] * dt
        ry[i] = ry[i - 1] + vy[i - 1] * dt
        vz[i] = vz[i - 1] + ((c.q * c.B * vy[i - 1]) / c.m) * dt
        vy[i] = vy[i - 1] + ((c.q * c.E - c.q * c.B * vz[i - 1]) / c.m) * dt


    if gen_graph:
        plt.rcParams['text.usetex'] = True

        #plt.ylim(-0.5, 1.5)
        #plt.xlim(-0.004, 0.004)

        plt.axvline(x=c.R, color='r', ymin= 0, ymax=1)
        plt.axvline(x=-c.R, color='r', ymin= 0, ymax=1)

        plt.plot(ry, rz)

        plt.ylabel("z")
        plt.xlabel("y")
        plt.title("pipe simulation")
        plt.savefig('runge_kutta_wien_filter.png')
        plt.grid(True)
        plt.show()

    return ry[num_of_time_intervals-1], rz[num_of_time_intervals-1]


if __name__ == "__main__":
    energy = np.linspace(c.E_0 - c.delta_E, c.E_0 + c.delta_E, num=100)
    radius = np.linspace(-c.R, c.R, num=100)
    runge_kutta_numerical_integration(10**(-13), energy[50], 0, 1, True)
