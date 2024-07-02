"""
Abstract gradient training (AGT) is a framework for training neural networks with certified robustness guarantees.
"""

import logging
from abstract_gradient_training.certified_training import (
    poison_certified_training,
    privacy_certified_training,
    unlearning_certified_training,
    AGTConfig,
    utils,
)
from abstract_gradient_training import test_metrics
from abstract_gradient_training import bounds

logging.basicConfig(
    level=logging.INFO,
    format="[AGT] [%(levelname)-8s] [%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
