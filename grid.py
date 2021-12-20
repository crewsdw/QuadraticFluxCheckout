import numpy as np
import cupy as cp
import basis as b


class Grid:
    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # jacobian
        self.J = 2.0 / self.dx

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        # mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        # self.translation_matrix = cp.asarray(mid_identity +
        #                                      self.local_basis.translation_matrix[None, :, :] /
        #                                      self.J[:, None, None].get())
        #
        # # quad matrix
        # self.fourier_quads = cp.asarray((self.local_basis.weights[None, None, :] *
        #                                  np.exp(-1j * self.modes[:, None, None].get() * self.arr[None, :, :]) /
        #                                  self.J[None, :, None].get()) / self.length)
        # self.grid_phases = cp.exp(1j * self.modes[None, None, :] * self.device_arr[:, :, None])

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])