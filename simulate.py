import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# params
alpha = 1.0     # prey birth rate
beta = 0.1      # predation rate    (how often a prey is eaten)
delta = 0.075   # predator growth from eating
gamma = 1.5     # predator death rate

def lotka_volterra(t, state):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Starting populations and time span
x0, y0 = 20, 10
t_span = (0, 100)
t_eval = np.linspace(*t_span, 2000)

solve = solve_ivp(lotka_volterra, t_span, [x0,y0], t_eval=t_eval, rtol=1e-8)
prey, predators = solve.y

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Populations over time
ax1.plot(solve.t, prey, label="Prey")
ax1.plot(solve.t, predators, label="Predators")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population")
ax1.set_title("Populations over time")
ax1.legend()

# Phase plot: predators against prey
ax2.plot(prey, predators)
ax2.set_xlabel("Prey")
ax2.set_ylabel("Predators")
ax2.set_title("Phase plot")

plt.tight_layout()
plt.show()