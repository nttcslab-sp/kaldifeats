from typing import Union, Optional

import numpy

from kaldifeats.feats.mel_compute import get_mel_filterbanks
from kaldifeats.signal.spectral import spectrogram


def fbank_feats(
        x: numpy.ndarray,
        window_type: Union[str, tuple, numpy.ndarray]= 'povey',
        sample_frequency: float=16000.,
        frame_length: Optional[int]=400,
        frame_shift: Optional[int]=160,
        nfft: int=None,
        detrend: Union[str, callable, bool]='constant',
        return_onesided: bool=True,
        snip_edges: bool=True,
        dither: float=0.,
        dither_seed: Union[numpy.random.RandomState, int]=None,
        preemphasis_coefficient: float=0.97,
        mode: str='kaldi_power',
        num_mel_bins: int=23,
        low_freq: float=20.,
        high_freq: float=0.,
        vtln_warp_factor: float=1.0,
        vtln_low: float=100.,
        vtln_high: float=-500.,
        round_to_power_of_two: bool=True,
        htk_compat: bool=False,
        use_energy: bool=False,
        raw_energy: bool=True,
        log_energy_floor: float=0.0,
        ) -> numpy.ndarray:
    """Compute log-fbank-feats. 

    Args:
        x (numpy.ndarray):
        frame_length (int):
            Length of each segment.
        frame_shift (int):
        window_type (Union[str, Tuple[str, Union[float, int]], numpy.ndarray]):
            Desired window to use. If `window_type` is a string or tuple, it is
            passed to `get_window` to generate the window values, which are
            DFT-even by default. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length must be frame_length.
        nfft (Optional[int]):
            Length of the FFT used, if a zero padded FFT is desired. If
            `None`, the FFT length is `frame_length`.
        detrend (Union[str, bool, Callable[[numpy.ndarray], numpy.ndarray],
                       None]):
            Specifies how to detrend each segment. If `detrend` is a
            string, it is passed as the `type` argument to the `detrend`
            function. If it is a function, it takes a segment and returns a
            detrended segment. If `detrend` is `False`, no detrending is
            done. Defaults to 'constant'.
        return_onesided (bool):
            If `True`, return a one-sided spectrum for real data. If
            `False` return a two-sided spectrum. Note that for complex
            data, a two-sided spectrum is always returned.
        snip_edges (bool):
            If True, end effects will be handled by outputting
            only frames that completely fit in the file,
            and the number of frames depends on the frame-length.
        dither (Optional[float]):
        dither_seed (Optional[Union[numpy.random.RandomState, int]]):
            The random state is used for generation of random noise value
            in dithering function.
        preemphasis_coefficient (Optional[float]):
        round_to_power_of_two (bool):
            If true, round nfft size to power of two.
        mode (str):
            'psd'('power'), 'magnitude', 'kaldi_psd'('kaldi_power')
        num_mel_bins (int): Number of triangular mel_frequency bins
        sample_frequency (float): Waveform data sample frequency
        frame_length:
        low_freq (float): Low cutoff frequency for mel bins
        high_freq (float): High cutoff frequency for mel bins
            If high_freq is 0, then change to 0.5 * sample_freq + high_freq
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
        use_energy (bool): Add an extra dimension with energy to
            the FBANK output.
        raw_energy (bool): If true, compute energy before
            preemphasis and windowing
        log_energy_floor (float): Floor on energy (absolute, not relative)
    Returns:
        out (numpy.ndarray):
            out[fft_bin, mel_freq_bin]
    """
    if use_energy:
        if raw_energy:
            kwargs = {'return_energy': False,
                      'return_raw_energy': True}
        else:
            kwargs = {'return_energy': True,
                      'return_raw_energy': False}
    else:
        kwargs = {'return_energy': False,
                  'return_raw_energy': False}
    returns = spectrogram(
        x,
        window_type=window_type,
        frame_length=frame_length,
        frame_shift=frame_shift,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        snip_edges=snip_edges,
        dither=dither,
        dither_seed=dither_seed,
        preemphasis_coefficient=preemphasis_coefficient,
        mode=mode, **kwargs)

    if use_energy:
        assert len(returns) == 2
        spec, energy = returns
    else:
        spec = returns
    del returns

    fbanks = get_mel_filterbanks(
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_frequency,
        vtln_warp_factor=vtln_warp_factor,
        low_freq=low_freq,
        high_freq=high_freq,
        vtln_low=vtln_low,
        vtln_high=vtln_high,
        round_to_power_of_two=round_to_power_of_two,
        htk_compat=htk_compat,
        frame_length=frame_length,
        dtype=spec.dtype,)
    feat = numpy.log(numpy.maximum(numpy.dot(spec, fbanks),
                                   numpy.finfo(spec.dtype).eps))
    del spec
    del fbanks

    if use_energy:
        log_energy = numpy.log(numpy.maximum(energy,
                                             numpy.finfo(energy.dtype).eps))
        if log_energy_floor != 0.0:
            log_energy = numpy.maximum(log_energy, log_energy_floor)

        log_energy = log_energy[..., numpy.newaxis]
        if htk_compat:
            return numpy.concatenate([feat, log_energy], axis=-1)
        else:
            return numpy.concatenate([log_energy, feat], axis=-1)
    else:
        return feat
