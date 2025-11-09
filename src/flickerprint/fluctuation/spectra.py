#!/usr/bin/env python

r"""
Faster version of the spectrum fitting code
===========================================

Due to the need to precalculate the constant values before the fitting, this doesn't fit into the old workflow
and so it has been moved into a separate file.

 We want to minimise the function

 F(\sigma_bar, \kappa, q) = sum_{l=q}^{q_max} f_ql kappa_bar

 f_ql(\sigma_bar, kappa, q) = \frac{N_ql}{(l-1)(l+2)[l(l+1) - \sigma_bar]}

We decompose this into fractions

 f_ql = \frac{A(q, l)}{B(l) + C(l) \sigma_bar}, where,

   - A(q, l) = N_ql
   - B(l) = l(l+1)(l-1)(l+2)
   - C(l) = (l-1)(l+2)
"""

from typing import Callable

import numpy as np
from scipy.special import factorial, lpmv
from typing import Tuple
from scipy.optimize import least_squares


class SpectrumFitterBuilder:
    """
    This class contains the precomputed values used in the theoretical spectrum calculation.
    Supports vectorised sigma_bar and kappa_bar (floats or 2D numpy arrays).
    """

    def __init__(self, q_max: int = 15, l_max: int = 75):
        self.q_max = q_max
        self.a_ql = self._get_numerator_factor(q_max, l_max)
        self.b_l = self._get_constant_l_terms(l_max)
        self.c_l = self._get_base_l_terms(l_max)

    def get_spectra(self, sigma_bar, kappa_bar):
        """
        Calculate the theoretical spectrum for the given σ and κ.
        sigma_bar and kappa_bar can be floats or 2D numpy arrays of shape (N, M).
        Returns array of shape (N, M, q_max-1) if inputs are arrays, else (q_max-1,)
        """
        b_l = self.b_l[None, None, :]
        c_l = self.c_l[None, None, :]
        a_ql = self.a_ql[None, None, :, :]

        sigma_bar = np.asarray(sigma_bar)
        kappa_bar = np.asarray(kappa_bar)

        # Expand dims if scalar
        if sigma_bar.ndim == 0:
            sigma_bar = sigma_bar[None, None]
        if kappa_bar.ndim == 0:
            kappa_bar = kappa_bar[None, None]

        # Broadcast denominator and numerator
        denominator = kappa_bar[..., None, None] * (b_l + c_l * sigma_bar[..., None, None])
        out = a_ql / denominator
        # Sum over l axis (axis=2)
        spectra = out.sum(axis=-1)
        if spectra.shape[0] == 1 and spectra.shape[1] == 1:
            return spectra[0, 0]
        return spectra

    def create_fitting_function(
        self, spectrum_experimental: np.ndarray
    ) -> Tuple[Callable, Callable]:
        """Create a function that returns the error between the experimental and theoretical spectrum."""

        def residuals(fitting_params):
            sigma_bar, kappa_bar = fitting_params
            spectrum_theory = self.get_spectra(sigma_bar, kappa_bar)
            relative_errors = np.log10(np.abs(spectrum_theory / spectrum_experimental))
            return relative_errors

        def fitting_error(fitting_params):
            residual_vals = residuals(fitting_params)
            return np.sum(residual_vals**2, axis=-1)
        return residuals, fitting_error

    @classmethod
    def _get_constant_l_terms(cls, l_max: int = 60):
        """The constant terms on the denominator, given by B(l) above."""
        l = np.arange(2, l_max + 1)
        return l * (l + 1) * (l - 1) * (l + 2)

    @classmethod
    def _get_base_l_terms(cls, l_max: int = 60):
        l = np.arange(2, l_max + 1)
        return (l - 1) * (l + 2)

    @classmethod
    def _get_numerator_factor(cls, q_max: int = 15, l_max: int = 60) -> np.ndarray:
        """
        Returns an array of the A_lq terms as above. 

        This doesn't depend on the physical properties and so can be cached.

        Returns an array of shape
            (2 <= l <= L_MAX, l <= Q <= Q_MAX)
        """
        q_vector = np.arange(2, q_max + 1)
        l_vector = np.arange(2, l_max + 1)

        return _numerator_factor(q_vector, l_vector)
    

    def grid_scan_points(self, error_function: Callable, n_points: int = 18, n_best_points: int = 3):
        """
        Looks for the best starting point in ``n_points`` × ``n_points`` array and return the ``n_best_points`` options.

        Returns an array of shape (``n_best_points`` × 3) with the (fitting_error, sigma, kappa) values
        """
        sigma_values = np.logspace(-5, 7, n_points)
        kappa_values = np.logspace(-7, 5, n_points)

        ss, kk = np.meshgrid(sigma_values, kappa_values, indexing="ij")
        value_array = np.stack([ss, kk])
        fitting_error_array = np.apply_along_axis(
            error_function, axis=0, arr=value_array
        )

        # Trick to get the index of the ``k`` lowest elements, note they're not guaranteed to be ordered
        # This indexes into the flattened array, so we use this for the best sigma and kappa values
        best_indicies = np.argpartition(fitting_error_array, n_best_points, axis=None)[
            :n_best_points
        ]

        best_sigma = ss.ravel()[best_indicies]
        best_kappa = kk.ravel()[best_indicies]
        best_cost = fitting_error_array.ravel()[best_indicies]

        summary = np.stack([best_cost, best_sigma, best_kappa]).T
        return summary

    def minimiser(self, residual_function: Callable, error_function: Callable, n_starting_points: int = 18, n_best_points: int = 3):
        
        starting_points = self.grid_scan_points(
            error_function=error_function, n_points=n_starting_points, n_best_points=n_best_points
        )

        best_fit_value = np.inf
        parameters = None
        errors = None

        for starting_point in starting_points:
            x0 = starting_point[1:]
            minimiser_result = least_squares(
                fun=residual_function,
                x0=x0,
                bounds = ([0.0,0.0], [np.inf, np.inf]),
            )

            if 2*minimiser_result["cost"] < best_fit_value: # The definition used in least_squares has a factor of 1/2 in it that we don't use in our minimisation.
                best_fit_value = 2*minimiser_result["cost"]
                parameters = minimiser_result["x"]
                pcov = np.linalg.inv(minimiser_result["jac"].T.dot(minimiser_result["jac"]))

                S_sq = 2*minimiser_result["cost"] / (self.q_max-x0.size)

                errors = np.sqrt(np.abs(np.diag(pcov*S_sq)))


        if parameters is None:
            return None
    
        return {
            "sigma_bar": parameters[0],
            "sigma_bar_err": errors[0],
            "kappa_scale": parameters[1],
            "kappa_scale_err": errors[1],
            "fitting_error": best_fit_value,
        }
    
class SpectrumFitterBuilder_ST_Only(SpectrumFitterBuilder):

    # This one is even easier as we can precompute just about everything.
    def __init__(self, q_max: int = 15, l_max: int = 75):
        self.q_max = q_max
        self.a_ql = self._get_numerator_factor(self.q_max, l_max)
        self.c_l = self._get_base_l_terms(l_max)


    def get_spectra(self, sigma_bar: float):
        """Calculate the given theoretical spectrum for the given σ and κ."""
        out = self.a_ql/(self.c_l * sigma_bar)
        return out.sum(axis=1)
    
    def create_fitting_function(
        self, spectrum_experimental: np.ndarray
    ) -> Tuple[Callable, Callable]:
        """Create a function that returns the error between the experimental and theoretical spectrum."""

        def residuals(fitting_params):
            sigma_bar = fitting_params
            spectrum_theory = self.get_spectra(sigma_bar)
            relative_errors = np.log10(np.abs(spectrum_theory / spectrum_experimental))
            return relative_errors

        def fitting_error(fitting_params):
            residual_vals = residuals(fitting_params)
            return np.sum(residual_vals**2, axis=-1)
        return residuals, fitting_error
    
    def grid_scan_points(self, error_function: Callable, n_points: int = 18, n_best_points: int = 3):
        """
        Looks for the best starting point in ``n_points`` × ``n_points`` array and return the ``n_best_points`` options.

        Returns an array of shape (``n_best_points`` × 3) with the (fitting_error, sigma, kappa) values
        """
        sigma_values = np.logspace(-5, 7, n_points)
        fitting_error_array = np.array([error_function(np.array([s])) for s in sigma_values])
        best_indices = np.argpartition(fitting_error_array, n_best_points)[:n_best_points]
        best_sigma = sigma_values[best_indices]
        best_cost = fitting_error_array[best_indices]
        summary = np.stack([best_cost, best_sigma]).T
        return summary

    def minimiser(self, residual_function: Callable, error_function: Callable, n_starting_points: int = 18, n_best_points: int = 3):
        
        starting_points = self.grid_scan_points(
            error_function=error_function, n_points=n_starting_points, n_best_points=n_best_points
        )

        best_fit_value = np.inf
        parameters = None
        errors = None

        for starting_point in starting_points:
            x0 = starting_point[1:]
            minimiser_result = least_squares(
                fun=residual_function,
                x0=x0,
                bounds = ([0.0], [np.inf]),
            )

            if 2*minimiser_result["cost"] < best_fit_value:
                best_fit_value = 2*minimiser_result["cost"]
                parameters = minimiser_result["x"]
                pcov = np.linalg.inv(minimiser_result["jac"].T.dot(minimiser_result["jac"]))

                S_sq = 2*minimiser_result["cost"] / (self.q_max-x0.size)

                errors = np.sqrt(np.abs(np.diag(pcov*S_sq)))


        if parameters is None:
            return None
    
        return {
            "sigma_ST_bar": parameters[0],
            "sigma_ST_bar_err": errors[0],
            "fitting_error_ST": best_fit_value,
        }


def _numerator_factor(q_vec: np.ndarray, l_vec: np.ndarray):
    ll, qq = np.meshgrid(l_vec, q_vec, indexing="ij")
    valid = ll >= qq

    # Only the terms for which l - q are even are non-zero (odd function otherwise)
    even_terms = (ll + qq) % 2 == 0
    valid = np.logical_and(valid, even_terms)

    def numerator(l, q):
        return _norm_vec(l, q) * lpmv(q, l, 0) ** 2

    out = np.zeros_like(valid, dtype=float)
    out[valid] = numerator(ll[valid], qq[valid])

    return out.T


def _norm_vec(l, q):
    """Squared normalisation used in Häckl (doesn't break with numpy)"""
    return (2.0 * l + 1) / (4 * np.pi) * factorial(l - q) / factorial(l + q)


def calculate_durbin_watson(
    experimental_spectrum: np.ndarray, best_fit_spectrum: np.ndarray
):
    residuals = np.log10(experimental_spectrum / best_fit_spectrum)
    durbin_watson_stat = np.sum(np.square(residuals[1:] - residuals[:1])) / np.sum(
        np.square(residuals)
    )
    return durbin_watson_stat
