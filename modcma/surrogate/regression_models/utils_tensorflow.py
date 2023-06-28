from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from modcma.surrogate.regression_models.model_gp import tfb


def create_positive_variable(default, dtype=tf.float64, name=None):
    """ creates positive variable though bijectors """

    if isinstance(default, (float, int)):
        default = tf.constant(default, dtype=dtype)

    bijector = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    return tfp.util.TransformedVariable(
        initial_value=default,
        bijector=bijector,
        dtype=dtype,
        name=name,
    )


def create_constant(default, dtype=tf.float64, name: Optional[str] = None):
    """ creates tf constant defaults to float64 """
    return tf.constant(default, dtype=dtype, name=name)
