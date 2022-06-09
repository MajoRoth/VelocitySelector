import math
import numpy as np
import matplotlib.pyplot as plt

import constants
import constants as c


def runge_kutta_passes_filter(dt, E_0, y_0, gen_graph=False):
    omega = (c.q * c.B) / c.m
    T = (2 * math.pi) / omega
    num_of_time_intervals = math.ceil(T / dt)

    rz = np.zeros(1)
    ry = np.zeros(1)
    vz = np.zeros(1)
    vy = np.zeros(1)

    rz[0] = 0
    ry[0] = y_0
    vz[0] = math.sqrt((2 * E_0)/c.m)
    vy[0] = 0

    def az(vy):
        return (c.q * vy * c.B) / c.m

    def ay(vz):
        return (c.q * c.E - c.q * c.B * vz) / c.m

    i = 0
    while rz[i] <= 1:
        i += 1

        k1vz = az(vy[i - 1]) * dt
        k1vy = ay(vz[i - 1]) * dt
        k2vz = az(vy[i - 1] + 0.5 * k1vy) * dt
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

        rz = np.append(rz, [rz[i - 1] + (k1rz + 2 * k2rz + 2 * k3rz + k4rz) / 6])
        ry = np.append(ry, ry[i - 1] + (k1ry + 2 * k2ry + 2 * k3ry + k4ry) / 6)
        vz = np.append(vz, vz[i - 1] + (k1vz + 2 * k2vz + 2 * k3vz + k4vz) / 6)
        vy = np.append(vy, vy[i - 1] + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6)


    if gen_graph:
        plt.rcParams['text.usetex'] = True

        plt.ylim(-0.05, 1.1)
        plt.xlim(-0.0035, 0.0035)

        plt.axvline(x=c.R, color='r', ymin= 0, ymax=1)
        plt.axvline(x=-c.R, color='r', ymin= 0, ymax=1)

        plt.plot(ry, rz)

        plt.ylabel("z")
        plt.xlabel("y")
        plt.title("particle in velocity selector")
        plt.grid(True)
        plt.savefig('runge_kutta_wien_filter.png')
        plt.show()

    print(rz[i-1], rz[i])
    return -c.R < ry[i] < c.R, vz[-1]



def error_plane():
    energy = np.linspace(c.E_0 - c.delta_E, c.E_0 + c.delta_E, num=100)
    radius = np.linspace(-c.R, c.R, num=100)

    output_velocity = []
    output_radius = []

    for e in energy:
        for r in radius:
            if runge_kutta_passes_filter(10**(-10), e, r,)[0]:
                output_velocity.append(math.sqrt(e/c.E_0)-1)
                output_radius.append(r/c.R)

    plt.rcParams['text.usetex'] = True
    plt.scatter(output_radius, output_velocity)
    plt.grid(True)
    plt.title(r"$(\frac{y_0}{R}, \frac{\delta v}{v_0})$")
    plt.grid(True)
    plt.savefig('error_plane.png')
    plt.show()

def velocity_distribution(num_of_particles):
    energy = np.linspace(c.E_0 - c.delta_E, c.E_0 + c.delta_E, num=math.ceil(math.sqrt(num_of_particles)))
    radius = np.linspace(-c.R, c.R, num=math.ceil(math.sqrt(num_of_particles)))

    velocities = []
    i=0
    for e in energy:
        for r in radius:
            i+=1
            passes, vel = runge_kutta_passes_filter(10 ** (-10), e, r)
            if passes:
                velocities.append(vel)

            print(i)

    plt.hist(velocities)
    plt.rcParams['text.usetex'] = True
    plt.title(r"velocity distribution")
    plt.xlabel("velocity")
    plt.grid(True)
    plt.savefig('velocity_distribution.png')
    plt.show()


if __name__ == "__main__":
    # error_plane()
    # velocity_distribution(10**5)
    energy = np.linspace(c.E_0 - c.delta_E, c.E_0 + c.delta_E, num=math.ceil(100))
    radius = np.linspace(-c.R, c.R, num=math.ceil(100))
    # print(runge_kutta_passes_filter(10**(-10), energy[5], radius[5], gen_graph=True))

    velocity_distribution(1000)