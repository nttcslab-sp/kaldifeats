from functools import lru_cache
from typing import List, Union

import numpy

from kaldifeats.utils.roundup import round_up_to_nearest_power_of_two


def inv_melscale(mel_freq: Union[float, numpy.ndarray])\
        -> Union[float, numpy.ndarray]:
    return 700. * (numpy.exp(mel_freq / 1127.) - 1.)


def melscale(freq: Union[float, numpy.ndarray]) -> Union[float, numpy.ndarray]:
    return 1127. * numpy.log(1. + freq / 700.)


def vtln_warp_freq(vtln_low_cutoff: float,
                   vtln_high_cutoff: float,
                   low_freq: float,
                   high_freq: float,
                   vtln_warp_factor: float,
                   freq_array: numpy.ndarray):
    """Apply vtln warping to the input freq_array inplace.

    This computes a VTLN warping function that is not the same as HTK's one,
    but has similar inputs (this function has the advantage of never producing
    empty bins).

    This function computes a warp function F(freq), defined between low_freq and
    high_freq inclusive, with the following properties:
    F(low_freq) == low_freq
    F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection
     points.
    The lower inflection point (measured in terms of the unwarped
    frequency) is at frequency l, determined as described below.
    The higher inflection point is at a frequency h, determined as
     described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped
     frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
     Since (by the last point) F(h) == h/vtln_warp_factor, then
     max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
     h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
       = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped
     frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
     This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                         = vtln_low_cutoff * max(1, vtln_warp_factor)
    """
    # For out-of-range frequencies, just return the freq.
    assert vtln_low_cutoff > low_freq,\
        'be sure to set the --vtln-low option higher than --low-freq'
    assert vtln_high_cutoff < high_freq,\
        'be sure to set the --vtln-high '\
        'option lower than --high-freq [or negative]'

    one: float = 1.0
    l: float = vtln_low_cutoff * max(one, vtln_warp_factor)
    h: float = vtln_high_cutoff * min(one, vtln_warp_factor)
    scale: float = 1.0 / vtln_warp_factor
    fl: float = scale * l  # F(l)
    fh: float = scale * h  # F(h)
    assert l > low_freq and h < high_freq
    # slope of left part of the 3-piece linear function
    # slope of center part is just "scale"
    # slope of right part of the 3-piece linear function
    scale_left: float = (fl - low_freq) / (l - low_freq)
    scale_right: float = (high_freq - fh) / (high_freq - h)

    inds1 = low_freq <= freq_array
    inds2 = freq_array < l
    inds = inds1 * inds2
    freq_array[inds] = low_freq + scale_left *\
        (numpy.compress(inds, freq_array) - low_freq)

    inds3 = freq_array < h
    inds = ~inds2 * inds3
    freq_array[inds] = scale * numpy.compress(inds, freq_array)

    inds4 = freq_array <= high_freq
    inds3 = ~inds3 * inds4
    freq_array[inds3] = high_freq + scale_right *\
        (numpy.compress(inds3, freq_array) - high_freq)


@lru_cache(typed=True)
def get_mel_filterbanks(
        num_mel_bins: int=23,
        sample_frequency: Union[int, float]=16000.,
        frame_length: float=400.,
        low_freq: float=20.,
        high_freq: float=0.,  # 0.5 * sample_frequency + high_freq
        vtln_warp_factor: float=1.0,
        vtln_low: float=100.,
        vtln_high: float=-500.,
        round_to_power_of_two: bool=True,
        htk_compat: bool=False,
        dtype=None) -> numpy.ndarray:
    """Compute mel-filterbanks. Once returns, the output is cached in memory.

    Args:
        num_mel_bins (int): Number of triangular mel_frequency bins
        sample_frequency (float): Waveform data sample frequency
        frame_length:
        low_freq (float): Low cutoff frequency for mel bins
        high_freq (float): High cutoff frequency for mel bins
        vtln_warp_factor (float): Vtln warp factor
            (only applicable if vtln-map not specified)
            (if < 0, offset from Nyquist)
        vtln_low (float): Low inflection point in piecewise linear VTLN warping
            function
        vtln_high (float):
            High inflection point in piecewise linear VTLN warping function
             (if negative, offset from high-mel-freq
        round_to_power_of_two (bool):
            If true, round window size to power of two.
        htk_compat (bool): If true, put energy last.
            Warning: not sufficient to get HTK compatible features
            (need to change other parameters).
        dtype (type): The type of the output array
    Returns:
        out (numpy.ndarray):
            out[fft_bin, mel_freq_bin]
    """

    assert num_mel_bins >= 3, 'Must have at least 3 mel bins'
    if round_to_power_of_two:
        window_length_padded: int =\
            round_up_to_nearest_power_of_two(int(frame_length))
    else:
        window_length_padded: int = frame_length
    assert window_length_padded % 2 == 0

    num_fft_bins = window_length_padded // 2 + 1
    nyquist: float = 0.5 * sample_frequency

    if high_freq <= 0:
        high_freq = nyquist + high_freq

    assert nyquist > low_freq >= 0.0 and \
        nyquist >= high_freq > 0.0 and high_freq > low_freq,\
        f'Bad values: low-freq {low_freq} '\
        f'and high-freq {high_freq} vs. nyquist {nyquist}'

    mel_low_freq: float = melscale(low_freq)
    mel_high_freq: float = melscale(high_freq)

    # Divide by num_mel_bins+1 in next line
    # because of end-effects where the bins spread out to the sides.
    mel_freq_delta: float = (mel_high_freq - mel_low_freq) / (num_mel_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or\
        (vtln_low >= 0.0 and high_freq > vtln_low > low_freq and
         vtln_high > 0.0 and high_freq > vtln_high > vtln_low),\
        f'Bad values: vtln-low {vtln_low} and vtln-high {vtln_high} '\
        f'versus low-freq {low_freq} and high-freq {high_freq}'

    mel_bins = mel_low_freq + numpy.arange(num_mel_bins + 2) * mel_freq_delta
    if vtln_warp_factor != 1.0:
        mel_bins = inv_melscale(mel_bins)
        vtln_warp_freq(vtln_low,
                       vtln_high,
                       low_freq,
                       high_freq,
                       vtln_warp_factor,
                       mel_bins)
        mel_bins = melscale(mel_bins)

    # Fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width: float = sample_frequency / window_length_padded

    out = numpy.zeros((num_fft_bins, num_mel_bins), dtype=dtype)
    for ibin in range(num_mel_bins):
        # This_bin will be a vector of coefficients that is only
        # nonzero where this mel bin is active.
        start_bin = None
        this_bin: List[float] = []

        left_mel = mel_bins[ibin]
        center_mel = mel_bins[ibin + 1]
        right_mel = mel_bins[ibin + 2]
        for i in range(num_fft_bins):
            # Center mel-frequency of this fft bin.
            mel = melscale(fft_bin_width * i)
            if mel >= right_mel:
                break
            if mel > left_mel:
                if start_bin is None:
                    start_bin = i
                if mel <= center_mel:
                    weight: float = (mel - left_mel) / (center_mel - left_mel)
                    assert weight >= 0
                else:
                    weight: float =\
                        (right_mel - mel) / (right_mel - center_mel)
                this_bin.append(weight)
        assert len(this_bin) > 0, \
            f'You may have set num_mel_bins too large: {num_mel_bins}'

        # Replicate a bug in HTK, for testing purposes.
        if htk_compat and ibin == 0 and mel_low_freq != 0.0:
            this_bin[0] = 0.0

        out[start_bin:start_bin + len(this_bin), ibin] = this_bin
    return out


if __name__ == '__main__':
    import time
    keywords = {'num_mel_bins': 60,
                'sample_frequency': 16000,
                'frame_length': 400,
                'vtln_warp_factor': 1.0,
                'dtype': float}
    s = time.perf_counter()
    get_mel_filterbanks(**keywords)
    print(f'First: {time.perf_counter() - s}')

    s = time.perf_counter()
    get_mel_filterbanks(**keywords)
    print(f'First: {time.perf_counter() - s}')
