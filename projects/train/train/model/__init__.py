from .autoencoder import AutoencoderAframe
from .base import AframeBase
from .supervised import (
    SupervisedAframe,
    SupervisedAframeS4,
    SupervisedMultiModalAframe,
    SupervisedTimeSpectrogramAframe,
)
from .regression import LitS4DGaussianNLL, LitLinOSSGaussianNLL
