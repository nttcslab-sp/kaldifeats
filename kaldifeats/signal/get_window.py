from typing import Union, Tuple

import numpy
import scipy.signal


def get_window(window: Union[str, Tuple[str, Union[int, float]]],
               Nx: int, fftbins: bool=True) -> numpy.ndarray:
    """Return a window.
    
    This function depends on scipy.signal.get_window and basically 
    has compatible arguments exception with 
    
    1. 'povey', which is a window function developpend by Dan-povey 
    2. Suport the optional value of coefficiency for "blackman" window
    
    Parameters:
        window (string, float, or tuple):
            The type of window to create. See below for more details.
        Nx (int):
            The number of samples in the window.
        fftbins (bool):
            If True, create a "periodic" window ready to use with ifftshift
            and be multiplied by the result of an fft (SEE ALSO fftfreq).
    Returns:
        get_window (numpy.ndarray):
            Returns a window of length `Nx` and type `window`
        
    Notes:
        Window types:
            povey, 
            boxcar, triang, blackman, hamming, hann, bartlett, flattop,
            parzen, bohman, blackmanharris, nuttall, barthann,
            kaiser (needs beta), gaussian (needs std),
            general_gaussian (needs power, width),
            slepian (needs width), chebwin (needs attenuation)
        If the window requires no parameters, then `window` can be a string.
        If the window requires parameters, then `window` must be a tuple
        with the first argument the string name of the window, and the next
        arguments the needed parameters.
        If `window` is a floating point number, it is interpreted as the beta
        parameter of the kaiser window.
        Each of the window types listed above is also the name of
        a function that can be called directly to create a window of
        that type.
        
    Examples:
        >>> get_window('triang', 7)
        array([ 0.25,  0.5 ,  0.75,  1.  ,  0.75,  0.5 ,  0.25])
        >>> get_window(('kaiser', 4.0), 9)
        array([ 0.08848053,  0.32578323,  0.63343178,  0.89640418,  1.        ,
                0.89640418,  0.63343178,  0.32578323,  0.08848053])
        >>> get_window(4.0, 9)
        array([ 0.08848053,  0.32578323,  0.63343178,  0.89640418,  1.        ,
                0.89640418,  0.63343178,  0.32578323,  0.08848053])
    """
    if isinstance(window, tuple):
        if len(window) == 2 and window[0] in ['blackman', 'black', 'blk']:
            window, coeff = window
            return blackman(Nx, coeff=coeff, sym=fftbins)
    elif window in ['povey', 'danpovey']:
        return povey(Nx, sym=fftbins)
    else:
        return scipy.signal.get_window(window, Nx=Nx, fftbins=fftbins)


def povey(M: int, sym: bool=True) -> numpy.ndarray:
    """
    "povey" is a window dan-povey made to be similar to Hamming 
    but to go to zero at the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
    He just don't think the Hamming window makes sense as a windowing function.
     
    Args:
        M (int)
            Number of points in the output window. If zero or less, an empty
            array is returned.
        sym (bool):
            When True (default), generates a symmetric window, 
            for use in filter design.
            When False, generates a periodic window,
            for use in spectral analysis.
        
    Returns:
        w (numpy.ndarray):
            The window, with the maximum value normalized to 
            1 (though the value 1 does not appear if
             `M` is even and `sym` is True).   
    """
    # Docstring adapted from NumPy's blackman function
    if M < 1:
        return numpy.array([])
    if M == 1:
        return numpy.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M += 1
    n = numpy.arange(0, M)
    w = numpy.power(0.5 - 0.5 * numpy.cos(2.0 * numpy.pi * n / (M - 1)),
                    0.85)
    if not sym and not odd:
        w = w[:-1]
    return w


def blackman(M: int, coeff: float=0.42, sym: bool=True) -> numpy.ndarray:
    """Originated from scipy.signal.windows.blackman 
    with additionally parameter "coeff" for comapatibility of kaldi 
    
    Return a Blackman window.
    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.
    
    Args:
        M (int)
            Number of points in the output window. If zero or less, an empty
            array is returned.
        coeff (float): The coefficient value for blackman function
        sym (bool):
            When True (default), generates a symmetric window, 
            for use in filter design.
            When False, generates a periodic window, 
            for use in spectral analysis.
        
    Returns:
        w (numpy.ndarray):
            The window, with the maximum value normalized to 
            1 (though the value 1 does not appear if
             `M` is even and `sym` is True).
    Notes:
        The Blackman window is defined as
        .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)
        Most references to the Blackman window come from the signal processing
        literature, where it is used as one of many windowing functions for
        smoothing values.  It is also known as an apodization (which means
        "removing the foot", i.e. smoothing discontinuities at the beginning
        and end of the sampled signal) or tapering function. It is known as a
        "near optimal" tapering function, almost as good (by some measures)
        as the Kaiser window.
        References
        ----------
        .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
               spectra, Dover Publications, New York.
        .. [2] Oppenheim, A.V., and R.W. Schafer. 
                Discrete-Time Signal Processing.
               Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    """
    # Docstring adapted from NumPy's blackman function
    if M < 1:
        return numpy.array([])
    if M == 1:
        return numpy.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = numpy.arange(0, M)
    w = (coeff - 0.5 * numpy.cos(2.0 * numpy.pi * n / (M - 1)) +
         (1 - coeff) * numpy.cos(4.0 * numpy.pi * n / (M - 1)))
    if not sym and not odd:
        w = w[:-1]
    return w