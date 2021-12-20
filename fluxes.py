import numpy as np
import cupy as cp
import variables as var


def basis_product(flux, basis_arr, axis):
    return cp.tensordot(flux, basis_arr,
                        axes=([axis], [1]))


def quadratic_basis_product(flux, basis_arr, axes):
    return cp.tensordot(flux, basis_arr,
                        axes=(axes, [1, 2]))


class DGFlux:
    def __init__(self, resolution, order, nu):
        self.res = resolution
        self.order = order
        self.nu = nu

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(self.res), 0),
                                (slice(self.res), -1)]
        # self.boundary_slices_pad = [(slice(self.res + 2), 0),
        #                             (slice(self.res + 2), -1)]
        # self.flux_slice = [(slice(resolution), slice(order))]  # not necessary
        self.num_flux_size = (self.res, 2)

        # for array padding
        # self.pad_field, self.pad_spectrum = None, None

        # arrays
        self.flux = var.Scalar(resolution=self.res, order=order)
        self.output = var.Scalar(resolution=self.res, order=order)

    def semi_discrete_rhs(self, scalar, grid):
        """ Computes the semi-discrete equation for DG of Burger's equation """
        # Compute the gradient variable
        grad = grid.J * self.compute_grad(scalar, grid=grid)
        diffusion_term = grid.J * self.nu * self.compute_diffusion(grad=grad, grid=grid)
        momentum_flux_term = grid.J * self.compute_momentum_flux(scalar=scalar, grid=grid)
        self.output.arr = diffusion_term - momentum_flux_term

    def compute_grad(self, scalar, grid):
        """ Compute the gradient variable using one side of the alternating flux """
        return -1.0 * (basis_product(flux=scalar.arr, basis_arr=grid.local_basis.internal, axis=1) -
                       self.numerical_flux_grad(flux=scalar.arr, grid=grid))

    def compute_diffusion(self, grad, grid):
        return -1.0 * (basis_product(flux=grad, basis_arr=grid.local_basis.internal, axis=1) -
                       self.numerical_flux_diff(flux=grad, grid=grid))

    def compute_momentum_flux(self, scalar, grid):
        # outer_product = scalar.arr[:, :, None] * scalar.arr[]
        # Using aliased basis
        # return -1.0 * (basis_product(flux=0.5 * scalar.arr ** 2.0, basis_arr=grid.local_basis.internal, axis=1) -
        #                self.numerical_flux_hyperbolic(flux=0.5 * scalar.arr ** 2.0, speed=scalar.arr, grid=grid))

        # Using quadratic flux basis
        return -1.0 * (quadratic_basis_product(flux=0.5 * scalar.arr[:, :, None] * scalar.arr[:, None, :],
                                               basis_arr=grid.local_basis.quadratic_flux_matrix,
                                               axes=[1, 2]) -
                       self.numerical_flux_hyperbolic(flux=0.5 * scalar.arr ** 2.0, speed=scalar.arr, grid=grid))

    def numerical_flux_grad(self, flux, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size)

        # "Alternating flux" for gradient: always choose value to the left
        num_flux[self.boundary_slices[0]] = -1.0 * flux[self.boundary_slices[0]]
        num_flux[self.boundary_slices[1]] = cp.roll(flux[self.boundary_slices[0]], shift=-1, axis=0)

        return basis_product(flux=num_flux, basis_arr=grid.local_basis.numerical, axis=1)

    def numerical_flux_diff(self, flux, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size)

        # "Alternating flux" for basic variable: always choose value to the right
        num_flux[self.boundary_slices[0]] = -1.0 * cp.roll(flux[self.boundary_slices[1]], shift=+1, axis=0)
        num_flux[self.boundary_slices[1]] = flux[self.boundary_slices[1]]

        return basis_product(flux=num_flux, basis_arr=grid.local_basis.numerical, axis=1)

    def numerical_flux_hyperbolic(self, flux, speed, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size)

        # Measure upwind directions
        speed_neg = cp.where(condition=speed < 0, x=1, y=0)
        speed_pos = cp.where(condition=speed >= 0, x=1, y=0)

        # Upwind flux, left and right faces
        num_flux[self.boundary_slices[0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[1]], shift=1, axis=0),
                                                                speed_pos[self.boundary_slices[0]]) +
                                                    cp.multiply(flux[self.boundary_slices[0]],
                                                                speed_neg[self.boundary_slices[0]]))
        num_flux[self.boundary_slices[1]] = (cp.multiply(flux[self.boundary_slices[1]],
                                                         speed_pos[self.boundary_slices[1]]) +
                                             cp.multiply(cp.roll(flux[self.boundary_slices[0]], shift=-1, axis=0),
                                                         speed_neg[self.boundary_slices[1]]))

        return basis_product(flux=num_flux, basis_arr=grid.local_basis.numerical, axis=1)
