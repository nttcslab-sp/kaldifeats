def round_up_to_nearest_power_of_two(n: int) -> int:
    """
    Args:
        n (int):
    Returns:
        N (int)
    Example:
        >>> round_up_to_nearest_power_of_two(123)
        128
        >>> round_up_to_nearest_power_of_two(5)
        8
        >>> round_up_to_nearest_power_of_two(1023)
        1024
    """
    assert n > 0
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1
