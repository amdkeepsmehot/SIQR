import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Initial population
N = 100
S0 = 98
I0 = 1
Q0 = 1
R0 = 0

# System parameters
beta = 0.1
delta = 9
alfa = 0.5
sigma = 0.8
omega = 2

# A grid of timepoints
t = np.linspace(0, 10)

# The SIQR model differential equations with the IQ saturation term
def SIQR(y, t, beta, delta, alfa, sigma, omega):
    S, I, Q, R = y
    dSdt = -beta*S*I + sigma*R + omega*Q
    dIdt = beta*S*I - delta*Q*I
    dQdt = delta*Q*I - omega*Q - alfa*Q
    dRdt = alfa*Q - sigma*R
    return dSdt, dIdt, dQdt, dRdt

# Initial conditions vector
y0 = S0, I0, Q0, R0

# Integrate the ODE over the time grid, t
ret = odeint(SIQR, y0, t, args=(beta, delta, alfa, sigma, omega))
S, I , Q, R = ret.T

# Plot the data
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='w', axisbelow = True)
ax.set_xlabel('Time')
ax.set_xlim(0, 10)
ax.set_ylabel('Population')
ax.set_ylim(0, 100)
ax.plot(t, S, label='Susceptible')
ax.plot(t, I, '--', label='Infected')
ax.plot(t, Q, '.-', label='Quarentined')
ax.plot(t, R, '*', label='Removed')
legend = ax.legend(loc='best')
legend.get_frame().set_alpha(0.5)
plt.title("With non linear IQ term", loc='center')
plt.show()