import math

# E_0 = 8.0109 * 10**(-13)
# delta_E = E_0/20
#
# m = 1.672621898 * 10**(-27)  # [Kg]
# q = 1.602176634 * 10**(-19)  # [C]
# B = 1
# E = math.sqrt((2*E_0)/m)*B  # [N/C]
# R = 0.003  # [m]




E_0 = 1
delta_E = 1

m = 1
q = 2 * math.pi
B = 1
E = 1
R = 1


analytic = (0, (2 * m * E * math.pi) / (q * B * B))

