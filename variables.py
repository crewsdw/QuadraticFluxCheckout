import numpy as np
import cupy as cp
import numpy.polynomial as poly
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt


class Scalar:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order

        # arrays
        self.arr, self.grad = None, None

    def initialize(self, grid, nu):
        # self.arr = nu * cp.sin(grid.device_arr) / (1.0 + 0.5 * np.cos(grid.device_arr))
        self.arr = cp.sin(grid.device_arr) / 2.0

    def conserved_quantity(self, grid):
        """ Compute the conserved quantity integral(0.5 * u^2, dx) """
        # u^2 is order 2(n-1) and needs GL quadrature of order n
        local_order = self.order
        gl_nodes, gl_weights = poly.legendre.leggauss(local_order)
        # Evaluate Legendre polynomials at finer grid
        ps = np.array([sp.legendre(s)(gl_nodes) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.tensordot(grid.local_basis.inv_vandermonde, ps, axes=([0], [0]))
        # Interpolated function at fine points
        interp_poly = np.tensordot(self.arr.get(), ell, axes=([1], [0]))

        # return integral
        return 0.5 * (gl_weights[None, :] * interp_poly ** 2.0).flatten().sum() / grid.J

    def l2_error(self, time, grid):
        """ Compute L2 error given the exact solution """
        local_order = 10 * self.order
        gl_nodes, gl_weights = poly.legendre.leggauss(local_order)
        # Evaluate Legendre polynomials at finer grid
        ps = np.array([sp.legendre(s)(gl_nodes) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.tensordot(grid.local_basis.inv_vandermonde, ps, axes=([0], [0]))
        # Interpolated function at fine points
        interp_poly = np.tensordot(self.arr.get(), ell, axes=([1], [0])).flatten()

        # Exact solution at fine points
        gl_nodes_global = (grid.mid_points[:, None] + gl_nodes[None, :] / grid.J).flatten()
        exact_sol = np.zeros_like(gl_nodes_global)
        for idx, x in enumerate(gl_nodes_global):
            this_sol = opt.fsolve(exact_sol_functional, x0=interp_poly[idx], args=(x, time),
                                  fprime=exact_sol_jacobian)
            exact_sol[idx] = this_sol
        # Visualize error
        # err = np.absolute(interp_poly - exact_sol)
        # plt.figure()
        # plt.semilogy(gl_nodes_global, err, 'o--')
        # plt.ylim([1e-12, 1e-2])
        # plt.title('Point-wise error with aliased flux divergence; e=5 and n=10')
        # plt.ylabel(r'Error $|u - u_a|$'), plt.xlabel(r'Position $x$')
        # plt.grid(True), plt.tight_layout()
        #
        # plt.show()

        # plt.savefig('figs/error_aliased_basis_o10_5e.png')
        # Compute L2 error
        error = interp_poly - exact_sol
        l2_error = np.square(error)
        l2_error = (gl_weights[None, :] * l2_error.reshape(self.res, local_order)).sum() / grid.J
        print('The L2 error is {:0.3e}'.format(np.sqrt(l2_error)))
        print('The max error is {:0.3e}'.format(np.amax(error)))
        quit()

        # plt.show()


def exact_sol_functional(u, x, t):
    return u - 0.5 * np.sin(x - u * t)


def exact_sol_jacobian(u, x, t):
    return 1.0 + 0.5 * t * np.cos(x - u * t)
