import numpy as np
import cupy as cp
import grid as g
import variables as var
import fluxes as fx
import stepper
import matplotlib.pyplot as plt
import scipy.optimize as opt

low, high = -np.pi, np.pi
res, order = 4, 20
nu = 0  # 1.0e-1

grid = g.Grid(low=low, high=high, elements=res, order=order)
scalar = var.Scalar(resolution=res, order=order)
scalar.initialize(grid=grid, nu=nu)

flux = fx.DGFlux(resolution=res, order=order, nu=nu)

steps = 15000
dt = 1.0e-4
final_time = steps * dt

plt.figure()
plt.plot(grid.arr.flatten(), scalar.arr.get().flatten(), 'o--')
plt.grid(True), plt.tight_layout()
plt.show()

initial_arr = scalar.arr

TimeStepper = stepper.Stepper(dt=dt, res=res, order=order, flux=flux)
TimeStepper.main_loop(scalar=scalar, grid=grid, steps=steps)

plt.figure()
plt.plot(TimeStepper.time_arr, TimeStepper.saved_arr, 'o--')
plt.xlabel(r'Time $t$'), plt.ylabel(r'Quantity $\frac{1}{2}\int |u|^2 dx$')
plt.grid(True), plt.tight_layout()
plt.savefig('figs/conservation_a_100e_5o.png')
plt.show()
# plt.show()

cons_to = np.abs((TimeStepper.saved_arr[-1] - TimeStepper.saved_arr[0]) / TimeStepper.saved_arr[0])

print('Energy conservation to {:0.3e}'.format(cons_to))

scalar.l2_error(time=final_time, grid=grid)

diff = scalar.arr - initial_arr


# exact_solution = nu * np.sin(grid.arr) * np.exp(-nu * final_time) / (1.0 + 0.5 * np.cos(grid.arr) * np.exp(-nu *
# final_time))

def exact_sol_functional(u, x, t):
    return u - 0.5 * np.sin(x - u * t)


def exact_sol_jacobian(u, x, t):
    return 1.0 + 0.5 * t * np.cos(x - u * t)


exact_sol = np.zeros_like(scalar.arr.flatten().get())
flat_arr = scalar.arr.flatten().get()
for idx, x in enumerate(grid.arr.flatten()):
    this_sol = opt.fsolve(exact_sol_functional, x0=flat_arr[idx], args=(x, final_time),
                          fprime=exact_sol_jacobian)
    exact_sol[idx] = this_sol

error = np.absolute(scalar.arr.get().flatten() - exact_sol)

plt.figure()
plt.plot(grid.arr.flatten(), initial_arr.get().flatten(), 'o--', label='Initial condition')
plt.plot(grid.arr.flatten(), scalar.arr.get().flatten(), 'o--', label='Numerical solution')
plt.plot(grid.arr.flatten(), exact_sol, 'o--', label='Exact solution')
plt.legend(loc='best')
plt.grid(True), plt.tight_layout()

plt.show()

# plt.figure()
# plt.plot(grid.arr.flatten(), diff.get().flatten(), 'o--')
# plt.title('Difference from IC')
# plt.grid(True), plt.tight_layout()

# plt.figure()
# plt.semilogy(grid.arr.flatten(), error.flatten(), 'o--')
# plt.title('Aliased Flux Divergence, dt={:0.1e}'.format(dt) + ' to t={:0.1f}'.format(steps*dt))
# plt.ylim([1e-12, 1e-4])
# plt.ylabel(r'Error $|u - u_a|$'), plt.xlabel(r'Position $x$')
# plt.grid(True), plt.tight_layout()
# plt.savefig('figs/error_aliased_basis_o20_3e.png')

# Compute L2 error


plt.show()
