'''
multiclass_svm.py : perform calculation for multiclass SVM loss
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.framework import ops

'''
labels: [batchSize]
logits: [batchSize, tagTotal]
'''
def multiclass_svm(labels, logits):

    # Read batch size and number of tags from logits
    shape = array_ops.shape(logits)
    batchSize = shape[0]
    tagTotal = shape[1]
    logits = math_ops.to_float(logits)

    with ops.control_dependencies([check_ops.assert_less(labels, math_ops.cast(tagTotal, labels.dtype))]):
        labels = array_ops.reshape(labels, shape=[-1])

    example_indices = array_ops.reshape(math_ops.range(batchSize), shape=[batchSize, 1])
    indices = array_ops.concat(
        [
            example_indices,
            array_ops.reshape(
                math_ops.cast(labels, example_indices.dtype),
                shape=[batchSize, 1])
        ],
        axis=1)
    label_logits = array_ops.reshape(
        array_ops.gather_nd(params=logits, indices=indices),
        shape=[batchSize, 1])

    one_cold_labels = array_ops.one_hot(
        indices=labels, depth=tagTotal, on_value=0.0, off_value=1.0)

    # Compute SVM margin
    margin = logits - label_logits + one_cold_labels


    margin = nn_ops.relu(margin)
    loss = math_ops.reduce_max(margin, axis=1)
    return losses.compute_weighted_loss(loss)