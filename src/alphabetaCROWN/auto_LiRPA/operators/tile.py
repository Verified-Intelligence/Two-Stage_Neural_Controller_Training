"""BoundTile"""
from torch.nn import Module
from .base import *

class BoundTile(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
    
    def forward(self, x, repeats):
        return x.repeat(repeats.tolist())

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert not self.is_input_perturbed(1)
        repeats = x[1].value

        def _bound_oneside(A):
            if A is None:
                return None
            # block_shape: (specs, d1/r1, r1, d2/r2, r2, ..., dn/rn, rn)
            # Reshaping A to block_shape and sum along the "r" dimensions
            # is equivalent to summing up all block fragments of A.
            block_shape = [A.shape[0]]
            axes_to_sum = []
            for i in range(len(repeats)):
                block_shape.append(A.size(i + 1) // repeats[i].item())
                block_shape.append(repeats[i].item())
                axes_to_sum.append(2 * i + 2)
            reshaped_A = A.reshape(*block_shape)
            next_A = reshaped_A.sum(dim=axes_to_sum)
            return next_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        repeats = x[1].lb.tolist()
        assert repeats[0] == 1, "shouldn't repeat on the batch dimension."
        lb = x[0].lb.repeat(repeats)
        ub = x[0].ub.repeat(repeats)
        repeats.insert(1, 1)    # Add the dim_in to align with lw/uw
        lw = x[0].lw.repeat(repeats)
        uw = x[0].uw.repeat(repeats)
        return LinearBound(lw, lb, uw, ub)