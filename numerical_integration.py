import math
import numpy as np
import matplotlib.pyplot as plt

import constants as c


def taylor_first_order(num_of_time_intervals, gen_graph=False):
    """
        r(t+dt) = r(t) + v(t)dt
        v(t+dt) = v(t) + a(t)dt
    """
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    dt = T / num_of_time_intervals

    rz = np.zeros(num_of_time_intervals+1)
    ry = np.zeros(num_of_time_intervals+1)
    vz = np.zeros(num_of_time_intervals+1)
    vy = np.zeros(num_of_time_intervals+1)

    rz[0] = 0
    ry[0] = 0
    vz[0] = (3 * c.E) / c.B
    vy[0] = 0

    for i in range(1, num_of_time_intervals+1):
        rz[i] = rz[i-1] + vz[i-1] * dt
        ry[i] = ry[i - 1] + vy[i - 1] * dt
        vz[i] = vz[i - 1] + ((c.q * c.B * vy[i-1]) / c.m) * dt
        vy[i] = vy[i - 1] + ((c.q * c.E - c.q * c.B * vz[i-1]) / c.m) * dt

    if gen_graph:
        plt.rcParams['text.usetex'] = True

        figure, axis = plt.subplots(2, 1, sharex=True)

        axis[0].plot(ry, rz)
        axis[0].set_title(r"$(y, z)$")

        axis[1].plot(vy, vz)
        axis[1].set_title(r"$(v_y, v_z)$")

        axis[0].grid(True)
        axis[1].grid(True)
        plt.savefig('taylor_first_order.png')
        plt.show()

    return ry[num_of_time_intervals], rz[num_of_time_intervals]


def midpoint(num_of_time_intervals, gen_graph=False):
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    dt = T / num_of_time_intervals

    rz = np.zeros(num_of_time_intervals+1)
    ry = np.zeros(num_of_time_intervals+1)
    vz = np.zeros(num_of_time_intervals+1)
    vy = np.zeros(num_of_time_intervals+1)

    rz[0] = 0
    ry[0] = 0
    vz[0] = (3 * c.E) / c.B
    vy[0] = 0

    def az(vy):
        return (c.q * vy * c.B) / c.m

    def ay(vz):
        return (c.q * c.E - c.q * c.B * vz) / c.m

    for i in range(1, num_of_time_intervals+1):
        k1vz = az(vy[i-1]) * dt
        k1vy = ay(vz[i - 1]) * dt
        k2vz = az(vy[i-1] + 0.5 * k1vy) * dt
        k2vy = ay(vz[i - 1] + 0.5 * k1vz) * dt

        # k1rz = vz[i - 1] * dt
        # k1ry = vy[i - 1] * dt
        k2rz = (vz[i-1] + 0.5 * k1vz) * dt
        k2ry = (vy[i-1] + 0.5 * k1vy) * dt

        rz[i] = rz[i-1] + k2rz
        ry[i] = ry[i - 1] + k2ry
        vz[i] = vz[i - 1] + k2vz
        vy[i] = vy[i - 1] + k2vy

    if gen_graph:
        plt.rcParams['text.usetex'] = True

        figure, axis = plt.subplots(2, 1, sharex=True)

        axis[0].plot(ry, rz)
        axis[0].set_title(r"$(y, z)$")

        axis[1].plot(vy, vz)
        axis[1].set_title(r"$(v_y, v_z)$")

        axis[0].grid(True)
        axis[1].grid(True)
        plt.savefig('midpoint.png')
        plt.show()

    return ry[num_of_time_intervals], rz[num_of_time_intervals]


def runge_kutta(num_of_time_intervals, gen_graph=False):
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    dt = T / num_of_time_intervals

    rz = np.zeros(num_of_time_intervals+1)
    ry = np.zeros(num_of_time_intervals+1)
    vz = np.zeros(num_of_time_intervals+1)
    vy = np.zeros(num_of_time_intervals+1)

    rz[0] = 0
    ry[0] = 0
    vz[0] = (3 * c.E) / c.B
    vy[0] = 0

    def az(vy):
        return (c.q * vy * c.B) / c.m

    def ay(vz):
        return (c.q * c.E - c.q * c.B * vz) / c.m


    for i in range(1, num_of_time_intervals+1):
        k1vz = az(vy[i-1]) * dt
        k1vy = ay(vz[i - 1]) * dt
        k2vz = az(vy[i-1] + 0.5 * k1vy) * dt
        k2vy = ay(vz[i - 1] + 0.5 * k1vz) * dt
        k3vz = az(vy[i - 1] + 0.5 * k2vy) * dt
        k3vy = ay(vz[i - 1] + 0.5 * k2vz) * dt
        k4vz = az(vy[i - 1] + k3vy) * dt
        k4vy = ay(vz[i - 1] + k3vz) * dt

        k1rz = vz[i - 1] * dt
        k1ry = vy[i - 1] * dt
        k2rz = (vz[i - 1] + 0.5 * k1vz) * dt
        k2ry = (vy[i - 1] + 0.5 * k1vy) * dt
        k3rz = (vz[i - 1] + 0.5 * k2vz) * dt
        k3ry = (vy[i - 1] + 0.5 * k2vy) * dt
        k4rz = (vz[i - 1] + k3vz) * dt
        k4ry = (vy[i - 1] + k3vy) * dt

        rz[i] = rz[i - 1] + (k1rz + 2 * k2rz + 2 * k3rz + k4rz) / 6
        ry[i] = ry[i - 1] + (k1ry + 2 * k2ry + 2 * k3ry + k4ry) / 6
        vz[i] = vz[i - 1] + (k1vz + 2 * k2vz + 2 * k3vz + k4vz) / 6
        vy[i] = vy[i - 1] + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6

    if gen_graph:
        plt.rcParams['text.usetex'] = True

        figure, axis = plt.subplots(2, 1, sharex=True)

        axis[0].plot(ry, rz)
        axis[0].set_title(r"$(y, z)$")

        axis[1].plot(vy, vz)
        axis[1].set_title(r"$(v_y, v_z)$")

        axis[0].grid(True)
        axis[1].grid(True)
        plt.savefig('runge_kutta.png')
        plt.show()

    return ry[num_of_time_intervals], rz[num_of_time_intervals]


def error(numeric, analytic):
    return (numeric[0] - analytic[0])**2 + (numeric[1] - analytic[1])**2


def plot_error_graph():
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    times = np.zeros(101) # N number of samples,
    taylor = np.zeros(101)
    mid = np.zeros(101)
    runge = np.zeros(101)
    i=0
    for n in np.linspace(100, 10000, 100):
        print(n)

        times[i] = T/n
        taylor[i] = error(taylor_first_order(int(n)), c.analytic)
        mid[i] = error(midpoint(int(n)), c.analytic)
        runge[i] = error(runge_kutta(int(n)), c.analytic)
        i += 1

    print(times)
    print(taylor)
    print(mid)
    print(runge)

    plt.rcParams['text.usetex'] = True
    plt.plot(times, taylor, label="taylor")
    plt.plot(times, mid, label="midpoint")
    plt.plot(times, runge, label="runge-kutta")

    plt.ylabel("error")
    plt.xlabel("dt")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("log log - error for dt")

    plt.grid(True)
    plt.legend()
    plt.savefig('error.png')

    plt.show()


if __name__ == "__main__":
    dt = 0.01
    plot_error_graph()
