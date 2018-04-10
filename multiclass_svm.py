'''
multiclass_svm.py : perform calculation for multiclass SVM loss
'''

from tensorflow.python.ops import array_ops, math_ops, nn_ops
from tensorflow.python.ops.losses import losses

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
    labels = array_ops.reshape(labels, shape=[-1])

    # Target indices
    targetIndex = array_ops.reshape(math_ops.range(batchSize), shape=[batchSize, 1])
    indices = array_ops.concat(
        [
            targetIndex,
            array_ops.reshape(
                math_ops.cast(labels, targetIndex.dtype),
                shape=[batchSize, 1])
        ],
        axis=1)

    # logits with the label class value
    labelLogits = array_ops.reshape(array_ops.gather_nd(params=logits, indices=indices),
        shape=[batchSize, 1])

    marginDelta = array_ops.one_hot(
        indices=labels, depth=tagTotal, on_value=0.0, off_value=1.0)

    # Compute SVM margin
    margin = logits - labelLogits + marginDelta

    # Use rectifier for max(features, 0)
    margin = nn_ops.relu(margin)

    # Maximize margin
    loss = math_ops.reduce_max(margin, axis=1)
    return losses.compute_weighted_loss(loss)