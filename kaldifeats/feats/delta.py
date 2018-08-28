import numpy


def delta(feat: numpy.ndarray, window: int=2) -> numpy.ndarray:
    """Compute delta features from a feature vector sequence.

    Args:
        feat(numpy.ndarray): A numpy array of size
            (NUMFRAMES by number of features) containing features.
            Each row holds 1 feature vector.
        window(int): For each frame, calculate delta
            features based on preceding and following N frames
    Returns:
        A numpy array of size (NUMFRAMES by number of features) containing
        delta features. Each row holds 1 delta feature vector.
    """
    assert window > 0
    delta_feat = numpy.zeros_like(feat)
    for i in range(1, window + 1):
        delta_feat[:-i] += i * feat[i:]
        delta_feat[i:] += -i * feat[:-i]
        # Padding
        delta_feat[-i:] += i * feat[-1]
        delta_feat[:i] += -i * feat[0]
    delta_feat /= 2 * sum(i ** 2 for i in range(1, window + 1))
    return delta_feat


def add_deltas(x: numpy.ndarray, window: int=2, order: int=2) -> numpy.ndarray:
    feats = [x]
    for _ in range(order):
        feats.append(delta(feats[-1], window))
    return numpy.concatenate(feats, axis=1)
