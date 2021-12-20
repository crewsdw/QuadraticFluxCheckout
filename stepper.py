import numpy as np
# import scipy.integrate as spint
import time as timer
# import fluxes as fx
import variables as var
# import cupy as cp


nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, res, order, flux):
        self.res = res
        self.order = order
        self.dt = dt
        self.flux = flux

        self.time = 0
        self.time_arr = None
        self.saved_arr = None

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))


    def main_loop(self, scalar, grid, steps):
        print('Beginning main loop')
        t0 = timer.time()
        self.saved_arr = np.zeros(steps // 50)
        self.time_arr = np.zeros(steps // 50)
        for i in range(steps):
            self.ssprk3(scalar=scalar, grid=grid)
            self.time += self.dt
            if i % 50 == 0:
                print('\nTook x steps, time is {:0.3e}'.format(self.time))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
                # Compute conserved quantity
                self.time_arr[i//50] = self.time
                self.saved_arr[i//50] = scalar.conserved_quantity(grid=grid)

        print('\nAll done at time {:0.3e}'.format(self.time))
        print('Total steps were ' + str(steps))
        print('Time since start is {:0.3e}'.format((timer.time() - t0)))

    def ssprk3(self, scalar, grid):
        stage0 = var.Scalar(resolution=self.res, order=self.order)
        stage1 = var.Scalar(resolution=self.res, order=self.order)
        # zero stage
        self.flux.semi_discrete_rhs(scalar=scalar, grid=grid)
        stage0.arr = scalar.arr + self.dt * self.flux.output.arr
        # first stage
        self.flux.semi_discrete_rhs(scalar=stage0, grid=grid)
        stage1.arr = (
                self.rk_coefficients[0, 0] * scalar.arr +
                self.rk_coefficients[0, 1] * stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux.output.arr
        )
        # second stage
        self.flux.semi_discrete_rhs(scalar=stage1, grid=grid)
        scalar.arr = (
                self.rk_coefficients[1, 0] * scalar.arr +
                self.rk_coefficients[1, 1] * stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux.output.arr
        )
