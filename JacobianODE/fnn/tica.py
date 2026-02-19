"""
Time-structure Independent Component Analysis (tICA).

Vendored from MSMBuilder via https://github.com/williamgilpin/fnn
Original authors: Christian Schwantes, Robert McGibbon, Kyle A. Beauchamp,
                  Muneeb Sultan, Brooke Husic
Copyright (c) 2014, Stanford University. All rights reserved.

Only import paths have been adjusted for packaging.
"""

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.linalg
import warnings
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator as SklearnBaseEstimator

__all__ = ['tICA']


class BaseEstimator(SklearnBaseEstimator):
    def summarize(self):
        return 'NotImplemented'


def check_iter_of_sequences(sequences, allow_trajectory=False, ndim=2,
                            max_iter=None):
    value = True
    for i, X in enumerate(sequences):
        if not isinstance(X, np.ndarray):
            value = False
            break
        if X.ndim != ndim:
            value = False
            break
        if max_iter is not None and i >= max_iter:
            break
    if not value:
        raise ValueError('sequences must be a list of sequences')


def _assert_all_finite(X):
    X = np.asanyarray(X)
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def array2d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = X_2d.copy()
    return X_2d


class tICA(BaseEstimator, TransformerMixin):
    """Time-structure Independent Component Analysis (tICA).

    Linear dimensionality reduction using an eigendecomposition of the
    time-lag correlation matrix and covariance matrix of the data.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep.
    lag_time : int
        Delay time for time-lagged correlations.
    shrinkage : float, optional
        Covariance shrinkage intensity (range 0-1).
    kinetic_mapping : bool
        Weight projections by eigenvalues for kinetic distances.
    commute_mapping : bool
        Scale projections for commute distances.
    """

    def __init__(self, n_components=None, lag_time=1, shrinkage=None,
                 kinetic_mapping=False, commute_mapping=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.shrinkage = shrinkage
        self.shrinkage_ = None
        self.kinetic_mapping = kinetic_mapping
        self.commute_mapping = commute_mapping
        if self.kinetic_mapping and self.commute_mapping:
            raise ValueError("Can't have both kinetic mapping and commute mapping.")
        self.n_features = None
        self.n_observations_ = None
        self.n_sequences_ = None

        self._initialized = False
        self._outer_0_to_T_lagged = None
        self._sum_0_to_TminusTau = None
        self._sum_tau_to_T = None
        self._sum_0_to_T = None
        self._outer_0_to_TminusTau = None
        self._outer_offset_to_T = None
        self._components_ = None
        self._eigenvectors_ = None
        self._eigenvalues_ = None
        self._is_dirty = True

    def _initialize(self, n_features):
        if self._initialized:
            return
        if self.n_components is None:
            self.n_components = n_features
        self.n_features = n_features
        self.n_observations_ = 0
        self.n_sequences_ = 0
        self._outer_0_to_T_lagged = np.zeros((n_features, n_features))
        self._sum_0_to_TminusTau = np.zeros(n_features)
        self._sum_tau_to_T = np.zeros(n_features)
        self._sum_0_to_T = np.zeros(n_features)
        self._outer_0_to_TminusTau = np.zeros((n_features, n_features))
        self._outer_offset_to_T = np.zeros((n_features, n_features))
        self._initialized = True

    def _solve(self):
        if not self._is_dirty:
            if len(self._eigenvalues_) >= self.n_components:
                return
        if self.n_observations_ == 0:
            raise RuntimeError('The model must be fit() before use.')

        lhs = self.offset_correlation_
        rhs = self.covariance_

        if not np.allclose(lhs, lhs.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(rhs, rhs.T):
            raise RuntimeError('correlation matrix is not symmetric')

        vals, vecs = scipy.linalg.eigh(
            lhs, b=rhs,
            subset_by_index=(self.n_features - self.n_components, self.n_features - 1)
        )
        ind = np.argsort(vals)[::-1]
        vals = vals[ind]
        vecs = vecs[:, ind]

        self._eigenvalues_ = vals
        self._eigenvectors_ = vecs
        self._is_dirty = False

    @property
    def score_(self):
        self._solve()
        return self._eigenvalues_[:self.n_components].sum()

    @property
    def eigenvectors_(self):
        self._solve()
        return self._eigenvectors_[:, :self.n_components]

    @property
    def eigenvalues_(self):
        self._solve()
        return self._eigenvalues_[:self.n_components]

    @property
    def timescales_(self):
        self._solve()
        return -1. * self.lag_time / np.log(self._eigenvalues_[:self.n_components])

    @property
    def components_(self):
        return self.eigenvectors_[:, 0:self.n_components].T

    @property
    def means_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        means = (self._sum_0_to_TminusTau + self._sum_tau_to_T) / float(two_N)
        return means

    @property
    def offset_correlation_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        term = (self._outer_0_to_T_lagged + self._outer_0_to_T_lagged.T) / two_N
        means = self.means_
        return term - np.outer(means, means)

    @property
    def covariance_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        term = (self._outer_0_to_TminusTau + self._outer_offset_to_T) / two_N
        means = self.means_
        S = term - np.outer(means, means)

        if self.shrinkage is None:
            sigma, self.shrinkage_ = rao_blackwell_ledoit_wolf(S, n=self.n_observations_)
        else:
            self.shrinkage_ = self.shrinkage
            p = self.n_features
            F = (np.trace(S) / p) * np.eye(p)
            sigma = (1 - self.shrinkage) * S + self.shrinkage * F

        return sigma

    def fit(self, sequences, y=None):
        self._initialized = False
        for X in sequences:
            self._fit(X)
        if self.n_sequences_ == 0:
            raise ValueError('All sequences were shorter than '
                             'the lag time, %d' % self.lag_time)
        return self

    def partial_fit(self, X):
        self._fit(X)
        return self

    def transform(self, sequences):
        sequences_new = []
        for X in sequences:
            X = array2d(X)
            if self.means_ is not None:
                X = X - self.means_
            X_transformed = np.dot(X, self.components_.T)

            if self.kinetic_mapping:
                X_transformed *= self.eigenvalues_

            if self.commute_mapping:
                regularized_timescales = 0.5 * self.timescales_ * \
                    np.tanh(np.pi * ((self.timescales_ - self.lag_time)
                                     / self.lag_time) + 1)
                X_transformed *= np.sqrt(regularized_timescales / 2)
                X_transformed = np.nan_to_num(X_transformed)
            sequences_new.append(X_transformed)
        return sequences_new

    def partial_transform(self, features):
        return self.transform([features])[0]

    def fit_transform(self, sequences, y=None):
        self.fit(sequences)
        return self.transform(sequences)

    def _fit(self, X):
        X = np.asarray(array2d(X), dtype=np.float64)
        if X.shape[1] > X.shape[0]:
            warnings.warn(
                "The number of features (%d) is greater than the length of "
                "the data (%d)." % (X.shape[1], X.shape[0])
            )
        self._initialize(X.shape[1])
        if not len(X) > self.lag_time:
            warnings.warn(
                "length of data (%d) is too short for the lag time (%d)"
                % (len(X), self.lag_time)
            )
            return

        self.n_observations_ += X.shape[0]
        self.n_sequences_ += 1

        self._outer_0_to_T_lagged += np.dot(X[:-self.lag_time].T, X[self.lag_time:])
        self._sum_0_to_TminusTau += X[:-self.lag_time].sum(axis=0)
        self._sum_tau_to_T += X[self.lag_time:].sum(axis=0)
        self._sum_0_to_T += X.sum(axis=0)
        self._outer_0_to_TminusTau += np.dot(X[:-self.lag_time].T, X[:-self.lag_time])
        self._outer_offset_to_T += np.dot(X[self.lag_time:].T, X[self.lag_time:])

        self._is_dirty = True

    def score(self, sequences, y=None):
        assert self._initialized
        V = self.eigenvectors_
        m2 = self.__class__(
            shrinkage=self.shrinkage, n_components=self.n_components,
            lag_time=self.lag_time,
        )
        for X in sequences:
            m2.partial_fit(X)

        numerator = V.T.dot(m2.offset_correlation_).dot(V)
        denominator = V.T.dot(m2.covariance_).dot(V)

        try:
            trace = np.trace(numerator.dot(np.linalg.inv(denominator)))
        except np.linalg.LinAlgError:
            trace = np.nan
        return trace

    def summarize(self):
        self.covariance_
        return (
            "tICA\n"
            f"n_components: {self.n_components}\n"
            f"shrinkage: {self.shrinkage_}\n"
            f"lag_time: {self.lag_time}\n"
            f"kinetic_mapping: {self.kinetic_mapping}\n"
            f"Top 5 timescales: {self.timescales_[:5]}\n"
            f"Top 5 eigenvalues: {self.eigenvalues_[:5]}\n"
        )


def rao_blackwell_ledoit_wolf(S, n):
    """Rao-Blackwellized Ledoit-Wolf shrinkage estimator of the covariance matrix.

    Parameters
    ----------
    S : ndarray, shape (p, p)
        Sample covariance matrix.
    n : int
        Number of data points.

    Returns
    -------
    sigma : ndarray, shape (p, p)
    shrinkage : float

    Reference
    ---------
    Chen, Wiesel, and Hero III. "Shrinkage estimation of high dimensional
    covariance matrices." ICASSP (2009).
    """
    p = len(S)
    assert S.shape == (p, p)

    alpha = (n - 2) / (n * (n + 2))
    beta = ((p + 1) * n - 2) / (n * (n + 2))

    trace_S2 = np.sum(S * S)
    U = ((p * trace_S2 / np.trace(S) ** 2) - 1)
    rho = min(alpha + beta / U, 1)

    F = (np.trace(S) / p) * np.eye(p)
    return (1 - rho) * S + rho * F, rho
