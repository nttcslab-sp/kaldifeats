import warnings
from typing import Union, Callable, Optional, Tuple, List

import numpy
import scipy.fftpack
import scipy.signal
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext

from kaldifeats.signal.get_window import get_window
from kaldifeats.utils.roundup import round_up_to_nearest_power_of_two


def spectrogram(
        x: numpy.ndarray,
        window_type: Union[str, tuple, numpy.ndarray]= 'povey',
        frame_length: Optional[int]=400,
        frame_shift: Optional[int]=160,
        nfft: Optional[int]=None,
        detrend: Union[str, callable, bool]='constant',
        return_onesided: bool=True,
        snip_edges: bool=True,
        dither: float=0.,
        dither_seed: Union[numpy.random.RandomState, int]=None,
        preemphasis_coefficient: float=0.,
        round_to_power_of_two: bool=True,
        return_energy: bool=False,
        return_raw_energy: bool=False,
        mode: str='kaldi_power',
        ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
    """ Compute the spectrogram from the input array. 
    This function is created taking care of compatibility with Kaldi.
    
    Args:
        x (array_like):
            Array or sequence containing the data to be analyzed.
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
        return_energy (bool):
            return a tuple (spectrogram, energy)
            Energy means the sum of square root of each window at time space.
        return_raw_energy (bool):
            return a tuple (spectrogram, raw_energy)
            Raw_energy means the energy before applying pre-emphasis
            (after dithering, dc-offset before pre-emphasis, matmul-window)
        mode (str): 
            'psd'('power'), 'magnitude', 
            'kaldi_psd'('kaldi_power')
    """
    if mode not in ('psd', 'power', 'magnitude', 'kaldi_psd', 'kaldi_power'):
        raise ValueError(f"Unsupported mode: {mode}, select one "
                         "'psd', 'power', 'magnitude', 'kaldi_psd'"
                         "'kaldi_power', 'kaldi_magnitude'")
    if not snip_edges:
        # Reflect around the beginning or end of the wave.

        # This is altenative ways of "boundary" and "paddped"
        # done in the feature extraction of Kaldi.
        # thus the boundary and padded values in stft arguments
        # should be given as False in this case.
        head_ext = frame_length // 2 - frame_shift // 2
        head = numpy.flip(x[..., :head_ext], -1)

        tail_ext = (-(x.shape[-1] + head_ext - frame_length) % frame_shift)\
            % frame_length
        tail = numpy.flip(x[..., -tail_ext:], -1)
        x = numpy.concatenate([head, x, tail], axis=-1)
    if nfft is None:
        nfft = frame_length
    elif nfft < frame_length:
        raise ValueError('nfft must be greater than or equal to frame_length.')
    else:
        nfft = int(nfft)
    if round_to_power_of_two:
        nfft = round_up_to_nearest_power_of_two(nfft)

    returns =\
        stft(x,
             frame_length=frame_length,
             frame_shift=frame_shift,
             window_type=window_type,
             nfft=nfft,
             detrend=detrend,
             return_onesided=return_onesided,
             boundary=None,
             padded=False,
             dither=dither,
             dither_seed=dither_seed,
             preemphasis_coefficient=preemphasis_coefficient,
             return_energy=return_energy,
             return_raw_energy=return_raw_energy,
             round_to_power_of_two=round_to_power_of_two
             )
    if isinstance(returns, tuple):
        spec = returns[0]
    else:
        spec = returns

    if mode == 'magnitude':
        spec = numpy.abs(spec)
    elif mode in ('psd', 'power'):
        spec = numpy.conjugate(spec) * spec
        if return_onesided:
            # Last point is unpaired Nyquist freq point, don't double
            if nfft % 2:
                spec[..., 1:] *= 2
            else:
                spec[..., 1:-1] *= 2
            spec = spec.real
    elif mode in ('kaldi_psd', 'kaldi_power'):
        spec = numpy.conjugate(spec) * spec
        if return_onesided:
            # Kaldi doesn't care about the unpaired nyquist freq points
            spec = spec.real

    if isinstance(returns, tuple):
        return (spec,) + returns[1:]
    else:
        return spec


def stft(
        x: numpy.ndarray,
        frame_length: int,
        frame_shift: int,
        window_type: Union[str,
                           Tuple[str, Union[float, int]],
                           numpy.ndarray]='hann',
        nfft: int=None,
        detrend: Union[str,
                       bool,
                       Callable[[numpy.ndarray], numpy.ndarray],
                       None]='constant',
        return_onesided: bool=True,
        boundary: str=None,
        padded: bool=False,
        dither: float=None,
        dither_seed: Union[numpy.random.RandomState, int]=None,
        preemphasis_coefficient: float=0.,
        return_energy: bool=False,
        return_raw_energy: bool=False,
        round_to_power_of_two: bool=True,
        dtype=None,
        deepcopy_input=False,
        ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
    """Compute short-time Fourier transform 
    
    This function is originated from scipy.signal.spectral and 
    some option are appended for kaldi compatibility.
    
    Args:
        x (numpy.ndarray):
            Array or sequence containing the data to be analyzed.
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
        boundary (Optional[str]):
            Specifies whether the input signal is extended at both ends, and
            how to generate the new values, in order to center the first
            windowed segment on the first input point. This has the benefit
            of enabling reconstruction of the first input point when the
            employed window function starts at zero. Valid options are
            ``['even', 'odd', 'constant', 'zeros', None]``. 
        padded (bool):
            Specifies whether the input signal is zero-padded at the end to
            make the signal fit exactly into an integer number of window
            segments, so that all of the signal is included in the output.
            Defaults to `False`. Padding occurs after boundary extension, if
            `boundary` is not `None`, and `padded` is `True`.
        dither (Optional[float]):
            Dithering constant (0.0 means no dither)
        dither_seed (Optional[Union[numpy.random.RandomState, int]]):
            The random state is used for generation of random noise value 
            in dithering function.
        preemphasis_coefficient (Optional[float]):
        return_energy (bool):
            return a tuple (spectrogram, energy)
            Energy means the sum of square root of each window at time space.
        return_raw_energy (bool):
            return a tuple (spectrogram, raw_energy)
            Raw_energy means the energy before applying pre-emphasis
            (after dithering, dc-offset before pre-emphasis, matmul-window)
        dtype (numpy.dtype): 
            The array precision for this computing and the output dtype.
            Default to None, indicating try to using the input dtype as it is,
            but if the array type is integer, convert it to double type.
        deepcopy_input (bool):
            If False, try to avoid copying of the input array, so the 
            input array could be changed.
            Even if False is specified and the input array dtype differs from 
            the computation dtype(See the avobe argument), deep-copying is done
        round_to_power_of_two (bool):
            If true, round nfft size to power of two. 
    
    Returns:
        freqs : ndarray
            Array of sample frequencies.
        t : ndarray
            Array of times corresponding to each data segment
        result : ndarray
            Array of output data, contents dependant on *mode* kwarg.
    References:
        .. [1] Stack Overflow, "Rolling window for 1D arrays in Numpy?",
               http://stackoverflow.com/a/6811241
        .. [2] Stack Overflow, "Using strides for an efficient moving
               average filter", http://stackoverflow.com/a/4947453
    """
    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError('Unknown boundary option "{0}", must be one of: {1}'
                         .format(boundary, list(boundary_funcs.keys())))

    x_org = x
    x = numpy.asarray(x, dtype=dtype)

    if x.size == 0:
        raise ValueError('Input array size is zero')
    if frame_length < 1:
        raise ValueError('frame_length must be a positive integer')
    if frame_length > x.shape[-1]:
        raise ValueError('frame_length is greater than input length')
    if 0 >= frame_shift:
        raise ValueError('frame_shift must be greater than 0')

    # parse window; if array like, then set frame_length = win.shape
    if isinstance(window_type, str) or isinstance(window_type, tuple):
        win = get_window(window_type, frame_length)
    else:
        win = numpy.asarray(window_type)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if x.shape[-1] < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if frame_length != win.shape[0]:
            raise ValueError(
                'value specified for frame_length is '
                'different from length of window')

    if nfft is None:
        nfft = frame_length
    elif nfft < frame_length:
        raise ValueError('nfft must be greater than or equal to frame_length.')
    else:
        nfft = int(nfft)
    if round_to_power_of_two:
        nfft = round_up_to_nearest_power_of_two(nfft)

    if return_onesided and numpy.iscomplexobj(x):
        warnings.warn('Input data is complex, switching to '
                      'return_onesided=False')
        return_onesided = False

    if x.dtype.kind == 'i':
        x = x.astype(numpy.float64)
    outdtype = numpy.result_type(x, numpy.complex64)
    if deepcopy_input and x is x_org:
        x = numpy.array(x)
        assert x is not x_org
    del x_org

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, frame_length // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        zeros_shape = x.shape[:-1] + (nadd,)
        x = numpy.concatenate((x, numpy.zeros(zeros_shape, dtype=x.dtype)),
                              axis=-1)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., numpy.newaxis]
    else:
        shape = x.shape[:-1] + \
                ((x.shape[-1] - frame_length) // frame_shift + 1, frame_length)
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = numpy.lib.stride_tricks.as_strided(x, shape=shape,
                                                    strides=strides)
    # else:
    #     nwindow = (x.shape[-1] - frame_length) // frame_shift + 1
    #     result = [x[..., i * frame_shift:i * frame_shift + frame_length]
    #               for i in range(nwindow)]
    #     result = numpy.concatenate(result, axis=-1)
    #     result = result.reshape(result.shape[:-1] + (-1, frame_length))
    del x

    if dither is not None and dither != 0.0:
        dithering(result, dither_value=dither,
                  state_or_seed=dither_seed)

    if detrend is not None and detrend:
        if callable(detrend):
            result = detrend(result)
        else:
            assert isinstance(detrend, str)
            result = scipy.signal.signaltools.detrend(result,
                                                      type=detrend, axis=-1)

    if return_raw_energy:
        raw_energy = numpy.sum(result ** 2, axis=1)

    if preemphasis_coefficient is not None \
            and preemphasis_coefficient != 0.0:
        pre_emphasis_filter(result, p=preemphasis_coefficient)

    if numpy.result_type(win, numpy.complex64) != outdtype:
        win = win.astype(outdtype)
    result = win * result

    if return_energy:
        energy = numpy.sum(result ** 2, axis=1)

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if not return_onesided:
        fft_func = scipy.fftpack.fft
    else:
        result = result.real
        fft_func = numpy.fft.rfft
    result = fft_func(result, n=nfft)

    if not return_energy and not return_raw_energy:
        return result
    else:
        returns: List[numpy.ndarray] = [result]
        if return_energy:
            returns.append(energy)
        if return_raw_energy:
            returns.append(raw_energy)
        return tuple(returns)


def dithering(wave: numpy.ndarray,
              dither_value: float=1.0,
              state_or_seed: Union[numpy.random.RandomState, int]=None)\
        -> None:
    if dither_value == 0.0:
        return
    if state_or_seed is None:
        state = numpy.random.RandomState()
    elif isinstance(state_or_seed, int):
        state = numpy.random.RandomState(state_or_seed)
    else:
        state = state_or_seed
    rand_gauss = numpy.sqrt(-2 * numpy.log(state.uniform(0, 1,
                                                         size=wave.shape))) * \
        numpy.cos(2 * numpy.pi * state.uniform(0, 1, size=wave.shape))
    wave += rand_gauss * dither_value


def pre_emphasis_filter(signal: numpy.ndarray, p=0.97) -> None:
    """Apply pre-emphasis filter to the input array inplace
    
    To implement pre emphasis fitler using scipy.signal.lfilter,
    
    >>> signal = scipy.signal.lfilter([1.0, -p], 1, signal)
        
    and this is equivalent to
    
    >>> signal[..., 1:] -= p * signal[..., :-1]
    
    The process only for the 0 index is different from this function.
    """

    signal[..., 1:] -= p * signal[..., :-1]
    signal[..., 0] -= p * signal[..., 0]


