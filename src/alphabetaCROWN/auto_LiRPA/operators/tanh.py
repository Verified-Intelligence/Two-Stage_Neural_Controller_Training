#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""BoundTanh and similar ops."""
import warnings
import torch
from torch.nn import Module
from torch.autograd import Function
from .base import *
from .activation_base import BoundOptimizableActivation


def dtanh(x):
    return  1 - torch.tanh(x).pow(2)

def dsigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def darctan(x):
    return (x.square() + 1.).reciprocal()

def d2tanh(x):
    return -2 * torch.tanh(x) * (1 - torch.tanh(x).pow(2))

def d2sigmoid(x):
    return dsigmoid(x) * (1 - 2 * torch.sigmoid(x))


# TODO refactor BoundTanh into a general op class for convex/concave like nonlinear functions.
class BoundTanh(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('tanh', torch.tanh, dtanh), precompute=True):
        super().__init__(attr, inputs, output_index, options)
        if options is None:
            options = {}
        self.use_default_ibp = True
        self.splittable = True
        self.ibp_intermediate = True
        self.activation = activation
        self.activation_name = activation[0]
        self.activation_forward = activation[1]
        self.activation_backward = activation[2]
        if precompute:
            self.precompute_relaxation(self.activation_forward, self.activation_backward)
            self.precompute_dfunc_values(self.activation_forward, self.activation_backward)
        # TODO make them configurable when implementing a general nonlinear activation.
        # Neurons whose gap between pre-activation bounds is smaller than this
        # threshold will be masked and don't need branching.
        self.split_min_gap = 1e-2 #1e-4
        # Neurons whose pre-activation bounds don't overlap with this range
        # are considered as stable (with values either 0 or 1) and don't need
        # branching.
        self.split_range = (self.range_l, self.range_u)
        # The initialization will be adjusted if the pre-activation bounds are too loose.
        self.loose_threshold = options.get('tanh', {}).get(
            'loose_threshold', None)
        self.convex_concave = None
        self.activation_bound_option = options.get('activation_bound_option', 'adaptive')

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {}
        self.tp_both_upper_init = {}

    def _init_opt_parameters_impl(self, size_spec, name_start, num_params=8):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = l.shape
        # Alpha dimension is (num_params, output_shape, batch, *shape) for Tanh.
        alpha = torch.empty(num_params, size_spec, *shape, device=l.device)
        alpha.data[:4] = (l + u) / 2
        alpha.data[4:6] = self.tp_both_lower_init[name_start]
        alpha.data[6:8] = self.tp_both_upper_init[name_start]
        if num_params > 8:
            alpha.data[8:] = 0
        return alpha

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit=500):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for %s (pre-activation limit: %f)',
                     self.__class__.__name__, x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper),
            check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d >= lower),
            check if the slope at d will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        # Given an upper bound point (>=0), find a line that is guaranteed to be a lower bound of this function.
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        r = torch.zeros_like(upper)
        # Initial guess, the tangent line is at -1.
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to 0.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower = l.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
        lower = -self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        l = torch.zeros_like(upper)
        r = torch.ones_like(upper)
        while True:
            checked = check_upper(lower, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel():
                break
        for _ in range(max_iter):
            m = (l + r) / 2
            checked = check_upper(lower, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        self.d_upper = r.clone()

        logger.debug('Done')

    def precompute_dfunc_values(self, func, dfunc, x_limit=500):
        """
        This function precomputes a list of values for dfunc.
        """
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        self.dfunc_values = dfunc(upper)

    def forward(self, x):
        return self.activation_forward(x)

    def retrieve_from_precompute(self, precomputed_d, input_bound, default_d):
        """
        precomputed_d: The precomputed tangent points.
        input_bound: The input bound of the function.
        default_d: If input bound goes out of precompute range, we will use default_d.
        All of the inputs should share the same shape.
        """

        # divide input bound into number of steps to the inflection point
        index = torch.max(
            torch.zeros(input_bound.numel(), dtype=torch.long, device=input_bound.device),
            (input_bound / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # If precompute range is smaller than input, tanget points will be taken from default.
        # The default value should be a guaranteed bound
        if index.max() >= precomputed_d.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            return torch.where(
                (index < precomputed_d.numel()).view(input_bound.shape),
                torch.index_select(
                    precomputed_d, 0, index.clamp(max=precomputed_d.numel() - 1)
                ).view(input_bound.shape),
                default_d,
            ).view(input_bound.shape)
        else:
            return torch.index_select(precomputed_d, 0, index).view(input_bound.shape)

    def generate_d_lower_upper(self, lower, upper):
        # Indices of neurons with input upper bound >=0, whose optimal slope to
        # lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0,
        # they will be masked later.
        d_lower = self.retrieve_from_precompute(self.d_lower, upper, lower)

        # Indices of neurons with lower bound <=0, whose optimal slope to upper
        # bound the function was pre-computed.
        d_upper = self.retrieve_from_precompute(self.d_upper, -lower, upper)
        return d_lower, d_upper

    def retrieve_d_from_k(self, k, func):
        d_indices = torch.searchsorted(torch.flip(self.dfunc_values, [0]), k, right=False)
        d_indices = self.num_points_pre - d_indices + 4
        d_left = d_indices * self.step_pre
        d_right = d_left + self.step_pre
        y_left = func(d_left)
        y_right = func(d_right)
        k_left = self.dfunc_values[d_indices]
        k_right = self.dfunc_values[torch.clamp(d_indices+1, max=self.dfunc_values.shape[0]-1)]
        # We choose the intersection of two tangent lines
        d_return = (k_left * d_left - k_right * d_right - y_left + y_right) / (k_left - k_right).clamp(min=1e-8)
        mask_almost_the_same = abs(k_left - k_right) < 1e-5
        d_return[mask_almost_the_same] = d_left[mask_almost_the_same]
        y_d = k_left * (d_return - d_left) + y_left
        return d_return, y_d

    def bound_relax_impl_same_slope(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)
        mask_almost_the_same = abs(upper - lower) < 1e-4
        k_direct[mask_almost_the_same] = dfunc(lower)[mask_almost_the_same]

        mask_direct_lower = k_direct <= dfunc(lower)
        mask_direct_upper = k_direct <= dfunc(upper)

        # We now find the tangent line with the same slope of k_direct
        # In the case of "mask_direct_lower(or upper)", there should be only one possible tangent point
        # at which we obtain the same slope within the interval [lower, upper]
        d, y_d = self.retrieve_d_from_k(k_direct, func)
        d[lower + upper < 0] *= -1  # This is the case "direct upper"
        y_d[lower + upper < 0] = 2 * func(torch.tensor(0)) - y_d[lower + upper < 0]
        d_clamped = torch.clamp(d, min=lower, max=upper)
        y_d[d_clamped != d] = func(d_clamped[d_clamped != d])
        self.add_linear_relaxation(
            mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l
        )
        self.add_linear_relaxation(
            mask=mask_direct_lower, type='upper', k=k_direct, x0=d_clamped, y0=y_d
        )
        self.add_linear_relaxation(
            mask=mask_direct_upper, type='upper', k=k_direct, x0=upper, y0=y_u
        )
        self.add_linear_relaxation(
            mask=mask_direct_upper, type='lower', k=k_direct, x0=d_clamped, y0=y_d
        )
        # Now we turn to the case where no direct line can be used
        d_lower, d_upper = self.generate_d_lower_upper(lower, upper)
        mask_both = torch.logical_not(mask_direct_upper + mask_direct_lower)
        # To make sure upper and lower bounds have the same slope,
        # we need the two tangents to be symmetrical
        d_same_slope = torch.max(torch.abs(d_lower), torch.abs(d_upper))
        k = dfunc(d_same_slope)
        y_d_same_slope = func(d_same_slope)
        y_d_same_slope_opposite = 2*func(torch.tensor(0)) - y_d_same_slope
        self.add_linear_relaxation(
            mask=mask_both, type='upper', k=k, x0=d_same_slope, y0=y_d_same_slope
        )
        self.add_linear_relaxation(
            mask=mask_both, type='lower', k=k, x0=-d_same_slope, y0=y_d_same_slope_opposite
        )


    def bound_relax_impl(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_neg, type='upper', k=k_direct, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_pos, type='lower', k=k_direct, x0=lower, y0=y_l)

        d_lower, d_upper = self.generate_d_lower_upper(lower, upper)

        if self.convex_concave is None:
            mask_direct_lower = k_direct < dfunc(lower)
            mask_direct_upper = k_direct < dfunc(upper)
        else:
            mask_direct_lower = torch.where(
                self.convex_concave,
                k_direct < dfunc(lower), k_direct > dfunc(upper))
            mask_direct_upper = torch.where(
                self.convex_concave,
                k_direct < dfunc(upper), k_direct > dfunc(lower))
        mask_direct_lower = torch.logical_and(mask_direct_lower, self.mask_both)
        mask_direct_upper = torch.logical_and(mask_direct_upper, self.mask_both)

        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            # Clipping is done here rather than after `opt.step()` call
            # because it depends on pre-activation bounds
            self.alpha[ns].data[0:2] = torch.max(
                torch.min(self.alpha[ns][0:2], upper), lower)
            self.alpha[ns].data[2:4] = torch.max(
                torch.min(self.alpha[ns][2:4], upper), lower)
            if self.convex_concave is None:
                self.alpha[ns].data[4:6] = torch.min(
                    self.alpha[ns][4:6], d_lower)
                self.alpha[ns].data[6:8] = torch.max(
                    self.alpha[ns][6:8], d_upper)
            else:
                self.alpha[ns].data[4:6, :] = torch.where(
                    self.convex_concave,
                    torch.max(lower, torch.min(self.alpha[ns][4:6, :], d_lower)),
                    torch.min(upper, torch.max(self.alpha[ns][4:6, :], d_lower))
                )
                self.alpha[ns].data[6:8, :] = torch.where(
                    self.convex_concave,
                    torch.min(upper, torch.max(self.alpha[ns][6:8, :], d_upper)),
                    torch.max(lower, torch.min(self.alpha[ns][6:8, :], d_upper))
                )

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0:2]  # For upper bound relaxation
            tp_neg = self.alpha[ns][2:4]  # For lower bound relaxation
            tp_both_lower = self.alpha[ns][4:6]
            tp_both_upper = self.alpha[ns][6:8]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            self.add_linear_relaxation(
                mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_lower), type='lower',
                k=dfunc(tp_both_lower), x0=tp_both_lower, y0=func(tp_both_lower))

            self.add_linear_relaxation(
                mask=mask_direct_upper, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_upper), type='upper',
                k=dfunc(tp_both_upper), x0=tp_both_upper, y0=func(tp_both_upper))

            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower', k=dfunc(tp_neg),
                x0=tp_neg, y0=func(tp_neg))
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper', k=dfunc(tp_pos),
                x0=tp_pos, y0=func(tp_pos))
        else:
            if self.opt_stage == 'init':
                # Initialize optimizable slope.
                tp_both_lower_init = d_lower.detach()
                tp_both_upper_init = d_upper.detach()

                if self.loose_threshold is not None:
                    # We will modify d_lower and d_upper inplace.
                    # So make a copy for these two.
                    tp_both_lower_init = tp_both_lower_init.clone()
                    tp_both_upper_init = tp_both_upper_init.clone()
                    # A different initialization if the pre-activation bounds
                    # are too loose
                    loose = torch.logical_or(lower < -self.loose_threshold,
                                            upper > self.loose_threshold)
                    d_lower[loose] = lower[loose]
                    d_upper[loose] = upper[loose]
                    # tp_both_lower_init[loose] = lower[loose]
                    # tp_both_upper_init[loose] = upper[loose]

                ns = self._start
                self.tp_both_lower_init[ns] = tp_both_lower_init
                self.tp_both_upper_init[ns] = tp_both_upper_init

            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
            k = dfunc(d_lower)
            # Another possibility is to use the direct line as the lower bound, when this direct line does not intersect with f.
            # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
            self.add_linear_relaxation(mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_lower),
                type='lower', k=k, x0=d_lower, y0=func(d_lower))

            # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
            k = dfunc(d_upper)
            self.add_linear_relaxation(
                mask=mask_direct_upper, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_upper),
                type='upper', k=k, x0=d_upper, y0=func(d_upper))
        
        # If lower and upper is too close, we just use IBP bounds to avoid numerical issues.
        mask_very_close = upper - lower < 1e-6
        self.add_linear_relaxation(
            mask=mask_very_close, type='lower', k=0, x0=lower, y0=y_l)
        self.add_linear_relaxation(
            mask=mask_very_close, type='upper', k=0, x0=upper, y0=y_u)


    def bound_relax_impl_post(self, x, func, dfunc):
        if self.opt_stage not in ['opt', 'reuse']:
            lower, upper = x.lower, x.upper
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            d_lower, d_upper = self.generate_d_lower_upper(lower, upper)

            self.add_linear_relaxation(
                mask=torch.logical_and(self.mask_both, d_lower >= m),
                type='lower', k=k, x0=m, y0=y_m)
            self.add_linear_relaxation(
                mask=torch.logical_and(self.mask_both, d_upper < m),
                type='upper', k=k, x0=m, y0=y_m)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        if self.activation_bound_option == 'same-slope':
            self.bound_relax_impl_same_slope(x, self.activation_forward, self.activation_backward)
        else:
            self.bound_relax_impl(x, self.activation_forward, self.activation_backward)
            self.bound_relax_impl_post(x, self.activation_forward, self.activation_backward)

    def get_split_mask(self, lower, upper, input_index):
        assert input_index == 0
        return torch.logical_and(
            upper - lower >= self.split_min_gap,
            torch.logical_or(upper >= self.split_range[0],
                             lower <= self.split_range[1])
        )

    def build_gradient_node(self, grad_upstream):
        if self.activation_name == 'tanh':
            node_grad = TanhGrad()
            grad_input = (grad_upstream, self.inputs[0].forward_value)
            grad_extra_nodes = [self.inputs[0]]
            return [(node_grad, grad_input, grad_extra_nodes)]
        elif self.activation_name == 'sigmoid':
            node_grad = SigmoidGrad()
            grad_input = (grad_upstream, self.inputs[0].forward_value)
            grad_extra_nodes = [self.inputs[0]]
            return [(node_grad, grad_input, grad_extra_nodes)]
        elif self.activation_name == 'arctan':
            node_grad = AtanGrad()
            grad_input = (grad_upstream, self.inputs[0].forward_value)
            grad_extra_nodes = [self.inputs[0]]
            return [(node_grad, grad_input, grad_extra_nodes)]
        else:
            raise NotImplementedError(self.activation_name)


class TanhGradOp(Function):
    @staticmethod
    def symbolic(_, preact):
        return _.op('grad::Tanh', preact).setType(preact.type())
    
    @staticmethod
    def forward(ctx, preact):
        return 1 - torch.tanh(preact)**2
    

class TanhGrad(Module):
    def forward(self, g, preact):
        return g * TanhGradOp.apply(preact).unsqueeze(1)


class BoundTanhGrad(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('tanh', dtanh, d2tanh), precompute=True):
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [0]
        # The inflection point is where d2f/dx2 = 0.
        self.inflection_point = 0.6585026
        self.func = activation[1]
        self.dfunc = activation[2]
        if precompute:
            self.precompute_relaxation()

    def forward(self, x):
        return self.func(x)

    def interval_propagate(self, *v):
        lower, upper = v[0]
        f_lower = self.func(lower)
        f_upper = self.func(upper)
        next_lower = torch.min(f_lower, f_upper)
        next_upper = torch.max(f_lower, f_upper)
        mask_both = torch.logical_and(lower < 0, upper > 0)
        next_upper[mask_both] = self.func(torch.tensor(0))
        return next_lower, next_upper
    
    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        return self.bound_relax_impl(x)
    
    def precompute_relaxation(self, x_limit=500):
        """
        This function precomputes the tangent lines that will be used as
        the lower/upper bounds for bell-shaped functions.
        Three tensors are precomputed:
        - self.precompute_x: The x values of the upper preactivation bound.
        - self.d_lower: The tangent points of the lower bound.
        - self.d_upper: The tangent points of the upper bound.
        """

        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)

        max_iter = 100
        func, dfunc = self.func, self.dfunc

        logger.debug('Precomputing relaxation for %s (pre-activation limit: %f)',
                     self.__class__.__name__, x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper),
            check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d <= lower),
            check if the slope at d will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        self.precompute_x = torch.arange(-self.x_limit, self.x_limit + self.step_pre, self.step_pre, device=self.device)
        self.d_lower = torch.zeros_like(self.precompute_x)
        self.d_upper = torch.zeros_like(self.precompute_x)

        # upper point that needs lower precomputed tangent line
        mask_need_d_lower = self.precompute_x >= -self.inflection_point
        upper = self.precompute_x[mask_need_d_lower]
        # 1. Initiasl guess, the tangent is at -1 (should be between (-inf, -inflection_point))
        r = -self.inflection_point * torch.ones_like(upper)
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to upper.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower[mask_need_d_lower] = l.clone()

        # upper point that needs upper precomputed tangent line
        mask_need_upper_d = self.precompute_x >= self.inflection_point
        upper = self.precompute_x[mask_need_upper_d]
        # 1. Initiasl guess, the tangent is at inflection_point/2 (should be between (0, inflection_point))
        r = self.inflection_point * torch.ones_like(upper)
        l = self.inflection_point / 2 * torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an upper bound at f(upper).
            checked = check_upper(upper, l).int()
            # If the initial guess is not smaller enough, then reduce it.
            l = checked * l + (1 - checked) * (l / 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an upper bound at f(upper).
        # We want to further tighten this bound by moving it closer to upper.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_upper(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to upper bound the function.
        self.d_upper[mask_need_upper_d] = l.clone()

    def retrieve_from_precompute(self, x, flip=False):
        if not flip:
            if x.max() > self.x_limit:
                warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Take the left endpoint of the interval
            x_indices = torch.searchsorted(self.precompute_x, x, right=True) - 1
            return self.d_lower[x_indices], self.d_upper[x_indices]
        else:
            if x.min() < -self.x_limit:
                warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Take the right endpoint of the interval
            x_indices = torch.searchsorted(self.precompute_x, -x, right=False)
            return -self.d_lower[x_indices], -self.d_upper[x_indices]
            

    def bound_relax_impl(self, x):
        lower, upper = x.lower, x.upper
        func, dfunc = self.func, self.dfunc
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # The tangent line at the midpoint can be a good approximation
        midpoint = (lower + upper) / 2
        k_midpoint = dfunc(midpoint)
        y_midpoint = func(midpoint)

        # If -inflection_point <= lower < upper <= inflection_point,
        # we call it "completely concave" region.
        mask_completely_concave = torch.logical_and(
            lower >= -self.inflection_point,
            upper <= self.inflection_point
        )
        self.add_linear_relaxation(
            mask=mask_completely_concave, type='lower', k=k_direct, x0=lower, y0=y_l)
        self.add_linear_relaxation(
            mask=mask_completely_concave, type='upper', k=k_midpoint, x0=midpoint, y0=y_midpoint)
        
        # From now on, we assume at least one of the bounds is outside the completely concave region.
        # Without loss of generality, we assume upper > inflection_point (indicated by mask_right).
        mask_right = lower + upper >= 0

        dl, du = self.retrieve_from_precompute(upper, flip=False)
        dl_, du_ = self.retrieve_from_precompute(lower, flip=True)

        # Case 1: Similar to a convex function
        mask_case1 = torch.logical_or(
            torch.logical_and(mask_right, lower >= du),
            torch.logical_and(torch.logical_not(mask_right), upper <= du_)
        )
        self.add_linear_relaxation(
            mask=mask_case1, type='upper', k=k_direct, x0=lower, y0=y_l)
        self.add_linear_relaxation(
            mask=mask_case1, type='lower', k=k_midpoint, x0=midpoint, y0=y_midpoint)
        
        # Case 2: Similar to a S-shaped function
        mask_case2_right = torch.logical_and(mask_right, lower < du)
        # The upper tangent point is lineraly interpolated between 0 and du,
        # given lower ranging between -upper and du.
        d_mask_case2_right_upper = du * (lower + upper) / (du + upper)
        k_mask_case2_right_upper = dfunc(d_mask_case2_right_upper)
        y_mask_case2_right_upper = func(d_mask_case2_right_upper)
        self.add_linear_relaxation(
            mask=mask_case2_right, type='upper',
            k=k_mask_case2_right_upper, x0=d_mask_case2_right_upper, y0=y_mask_case2_right_upper)
        # The lower tangent point is found based on lower.
        d_mask_case2_right_lower = (dl_ + upper) / 2
        k_mask_case2_right_lower = dfunc(d_mask_case2_right_lower)
        y_mask_case2_right_lower = func(d_mask_case2_right_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_right, dl_ < upper), type='lower',
            k=k_mask_case2_right_lower, x0=d_mask_case2_right_lower, y0=y_mask_case2_right_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_right, dl_ >= upper), type='lower',
            k=k_direct, x0=lower, y0=y_l)

        mask_case2_left = torch.logical_and(torch.logical_not(mask_right), upper > du_)
        # The upper tangent point is lineraly interpolated between du_ and 0,
        # given upper ranging between du_ and -lower.
        d_mask_case2_left_upper = du_ * (upper + lower) / (du_ + lower)
        k_mask_case2_left_upper = dfunc(d_mask_case2_left_upper)
        y_mask_case2_left_upper = func(d_mask_case2_left_upper)
        self.add_linear_relaxation(
            mask=mask_case2_left, type='upper',
            k=k_mask_case2_left_upper, x0=d_mask_case2_left_upper, y0=y_mask_case2_left_upper)
        # The lower tangent point is found based on upper.
        d_mask_case2_left_lower = (dl + lower) / 2
        k_mask_case2_left_lower = dfunc(d_mask_case2_left_lower)
        y_mask_case2_left_lower = func(d_mask_case2_left_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_left, dl > lower), type='lower',
            k=k_mask_case2_left_lower, x0=d_mask_case2_left_lower, y0=y_mask_case2_left_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_left, dl <= lower), type='lower',
            k=k_direct, x0=upper, y0=y_u)
        
        # If the lower and upper bounds are too close, we just use IBP bounds to avoid numerical issues.
        mask_very_close = upper - lower < 1e-6
        if mask_very_close.any():
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_neg), type='lower', k=0, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_neg), type='upper', k=0, x0=upper, y0=y_u)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_pos), type='lower', k=0, x0=upper, y0=y_u)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_pos), type='upper', k=0, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_both), type='lower', k=0, x0=lower, y0=torch.min(y_l, y_u))
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_both), type='upper', k=0, x0=upper, y0=torch.full_like(y_l, func(torch.tensor(0))))


class BoundSigmoid(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('sigmoid', torch.sigmoid, dsigmoid))


class SigmoidGradOp(Function):
    @staticmethod
    def symbolic(_, preact):
        return _.op('grad::Sigmoid', preact).setType(preact.type())
    
    @staticmethod
    def forward(ctx, preact):
        sigmoid_x = torch.sigmoid(preact)
        return sigmoid_x * (1 - sigmoid_x)
    

class SigmoidGrad(Module):
    def forward(self, g, preact):
        return g * SigmoidGradOp.apply(preact).unsqueeze(1)


class BoundSigmoidGrad(BoundTanhGrad):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('sigmoid', dsigmoid, d2sigmoid))
        self.inflection_point = 1.3169614


class BoundAtan(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('arctan', torch.arctan, darctan))
        self.split_range = (-torch.inf, torch.inf)


class AtanGrad(Module):
    def forward(self, g, preact):
        # arctan'(x) = 1 / (1 + x^2)
        return g / (1 + preact.square()).unsqueeze(1)


class BoundTan(BoundAtan):
    """
    The implementation of BoundTan is based on the S-shaped BoundAtan. We use the bounds from its
    inverse function and directly convert the bounds of the inverse function to bounds of the original
    function. This trick allows us to quickly implement bounds on inverse functions.
    """

    def forward(self, x):
        return torch.tan(x)

    def _check_bounds(self, lower, upper):
        # Lower and upper bounds must be within the same [-½π, ½π] region.
        lower_periods = torch.floor((lower + 0.5 * torch.pi) / torch.pi)
        upper_periods = torch.floor((upper + 0.5 * torch.pi) / torch.pi)
        if not torch.allclose(lower_periods, upper_periods):
            print('Tan preactivation lower bounds:\n', lower)
            print('Tan preactivation upper bounds:\n', upper)
            raise ValueError("BoundTan received pre-activation bounds that produce infinity. "
                    "The preactivation bounds are too loose. Try to reduce perturbation region.")
        # Return the period number for each neuron.
        # Period is 0 => bounds are within [-½π, ½π],
        # Period is 1 => bounds are within [-½π + π, ½π + π]
        # Period is -1 => bounds are within [-½π - π, ½π - π]
        return lower_periods

    def _init_masks(self, x):
        # The masks now must consider the periodicity.
        lower = torch.remainder(x.lower + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        upper = torch.remainder(x.upper + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        self.mask_pos = lower >= 0
        self.mask_neg = upper <= 0
        self.mask_both = torch.logical_not(torch.logical_or(self.mask_pos, self.mask_neg))

    def interval_propagate(self, *v):
        # We need to check if the input lower and upper bounds are within the same period.
        # Otherwise the bounds become infinity.
        concrete_lower, concrete_upper = v[0][0], v[0][1]
        self._check_bounds(concrete_lower, concrete_upper)
        return super().interval_propagate(*v)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        periods = self._check_bounds(x.lower, x.upper)
        periods = torch.pi * periods
        # Create a fake x with inversed lower and upper.
        inverse_x = lambda: None
        inverse_x.lower = torch.tan(x.lower)
        inverse_x.upper = torch.tan(x.upper)
        super().bound_relax(inverse_x, init=init, dim_opt=dim_opt)
        # Lower slope, lower bias, upper slope and upper bias are saved to
        # self.lw, self.lb, self.uw, self.ub. We need to reverse them.
        # E.g., y = self.lw * x + self.lb, now becomes x = 1./self.lw * y - self.lb / self.lw
        # Additionally, we need to add the missing ½π periods.
        new_upper_slope = 1. / self.lw
        new_upper_bias = - self.lb / self.lw - periods / self.lw
        new_lower_slope = 1. / self.uw
        new_lower_bias = - self.ub / self.uw - periods / self.uw

        # NaN can happen if lw=0 or uw=0 when the pre-activation bounds are too close
        # Replace the bounds with interval bounds.
        if (self.lw == 0).any():
            mask = self.lw == 0
            new_upper_slope[mask] = 0
            new_upper_bias[mask] = inverse_x.upper[mask]
        if (self.uw == 0).any():
            mask = self.uw == 0
            new_lower_slope[mask] = 0
            new_lower_bias[mask] = inverse_x.lower[mask]

        self.lw = new_lower_slope
        self.lb = new_lower_bias
        self.uw = new_upper_slope
        self.ub = new_upper_bias
