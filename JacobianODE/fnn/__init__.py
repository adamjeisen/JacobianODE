"""
FNN: False Nearest Neighbors regularizer for time series embedding.

PyTorch reimplementation of https://github.com/williamgilpin/fnn
Reference: Gilpin, "Deep reconstruction of strange attractors from time series"
           NeurIPS 2020. https://arxiv.org/abs/2002.05909

Quick start (sklearn-style API):
    >>> from JacobianODE.fnn import MLPEmbedding, FNN
    >>> model = MLPEmbedding(n_latent=3, time_window=10, latent_regularizer=FNN(1.0))
    >>> model.fit(time_series, train_steps=200, verbose=1)
    >>> embedding = model.transform(time_series)

For full-featured training with Lightning/W&B/Hydra:
    >>> from JacobianODE.fnn.lightning import LitFNNAutoencoder, create_dataloaders
    >>> from JacobianODE.fnn.networks import MLPAutoencoder
    >>> from JacobianODE.fnn.regularizers import FNN
"""

from .regularizers import Amplification, FNN, DeCov, loss_false, loss_cov, loss_amplification, tangent_space_entropy
from .networks import MLPAutoencoder, LSTMAutoencoder
from .models import (
    MLPEmbedding,
    LSTMEmbedding,
    ETDEmbedding,
    ConstantLagEmbedding,
    AMIEmbedding,
    TICAEmbedding,
)
from .coupling_flows import (
    AdditiveCouplingLayer,
    AnalyticAutoregressiveEncoder,
    AnalyticAutoregressiveLayer,
    CayleyOrthogonal,
    CouplingEncoder,
    DirectSumCouplingEncoder,
    FixedOrthogonal,
    FixedPermutation,
    MADE,
    MaskedLinear,
    SplineAutoregressiveEncoder,
    SplineAutoregressiveLayer,
)
from .sequence_networks import (
    SequenceAutoencoder,
    TransformerSequenceEncoder,
    SSMSequenceEncoder,
    TCNSequenceEncoder,
    TCNSpatialSequenceEncoder,
    StepDecoder,
    build_transformer,
    build_ssm,
    build_tcn,
    build_tcn_spatial,
)
from .sequence_models import (
    TransformerEmbedding,
    SSMEmbedding,
    TCNEmbedding,
    TCNSpatialEmbedding,
)
from .utils import hankel_matrix, standardize_ts, compute_variances, compute_s_dim
