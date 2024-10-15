"""
Abstract gradient training (AGT) is a framework for training neural networks with certified robustness guarantees.
"""

import logging

from abstract_gradient_training import (bounds, certified_training_utils,
                                        interval_arithmetic, nominal_pass,
                                        privacy_utils, test_metrics)
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training.poisoning import poison_certified_training
from abstract_gradient_training.privacy import privacy_certified_training
from abstract_gradient_training.unlearning import unlearning_certified_training

logger = logging.getLogger("abstract_gradient_training")
logger.handlers.clear()
formatter = logging.Formatter("[AGT] [%(levelname)-8s] [%(asctime)s] %(message)s", datefmt="%H:%M:%S")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
