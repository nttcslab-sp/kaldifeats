import shutil
from io import BytesIO
from subprocess import PIPE, Popen
from typing import Sequence

import kaldiio
import numpy
import scipy.io.wavfile


for _cmd in ['compute-mfcc-feats',
             'compute-fbank-feats',
             'compute-spectrogram-feats']:
    if shutil.which(_cmd) is None:
        raise RuntimeError(
            f'Command not found: {_cmd}. This tool depends on kaldi')


DUMMY_KEY = b'kaldifeats_commands_feats '


def _tpl_command(wave: numpy.ndarray,
                 cmd: Sequence[str],
                 sample_rate: float,
                 buffering: int=-1,
                 verbose: int=-1) -> numpy.ndarray:
    if wave.dtype not in (numpy.int8, numpy.int16, numpy.int32):
        raise ValueError(
            'Can read only PCM data. Input as int8, int16 or int32 type'
            '(int24 is not supported by numpy).')
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f'Command not found: {cmd[0]}')

    with Popen(cmd, stdin=PIPE, stdout=PIPE,
               stderr=None if verbose > -1 else PIPE,
               bufsize=buffering) as p:
        with BytesIO() as fin:
            scipy.io.wavfile.write(fin, int(sample_rate), wave)
            stdin = DUMMY_KEY + fin.getvalue()
        stdout, stderr = p.communicate(input=stdin)
        if p.returncode != 0:
            if stderr is not None:
                ms = stderr.decode()
            else:
                ms = f'Fail: {" ".join(cmd)}'
            raise RuntimeError(ms)
        fout = BytesIO(stdout)
        return next(kaldiio.load_ark(fout))[1]


def _check_length(wave_length,
                  frame_length_ms, frame_shift_ms, sample_frequency,
                  snip_edge):
    frame_length = calc_frame_length(frame_length_ms, sample_frequency)
    frame_shift = calc_frame_shift(frame_shift_ms, sample_frequency)
    if snip_edge:
        if wave_length < frame_length:
            raise ValueError(f'Wave length is too short: {wave_length}')
    else:
        if wave_length < frame_shift / 2:
            raise ValueError(f'Wave length is too short: {wave_length}')


def calc_frame_length(frame_length_ms: float, sample_frequency: float)\
        -> float:
    return sample_frequency * 0.001 * frame_length_ms


def calc_frame_shift(frame_shift_ms: float, sample_frequency: float)\
        -> float:
    return sample_frequency * 0.001 * frame_shift_ms


def calc_frame_length_ms(frame_length: float, sample_frequency: float)\
        -> float:
    return 1000. * frame_length / sample_frequency


def calc_frame_shift_ms(frame_shift: float, sample_frequency: float)\
        -> float:
    return 1000. * frame_shift / sample_frequency


def mfcc_feats(wave: numpy.ndarray,
               axis: int=-1,
               blackman_coeff: float=0.42,
               cepstral_lifter: float=22.,
               dither: float=1.,
               energy_floor: float=0.,
               frame_length_ms: float=25.,
               frame_shift_ms: float=10.,
               high_freq: float=0.,
               htk_compat: bool=False,
               low_freq: float=20.,
               num_ceps: int=13,
               num_mel_bins: int=23,
               preemphasis_coefficient: float=0.97,
               raw_energy: bool=True,
               remove_dc_offset: bool=True,
               round_to_power_of_two: bool=True,
               sample_frequency: float=16000.,
               snip_edges: bool=True,
               subtract_mean: bool=False,
               use_energy: bool=True,
               utt2spk: str='',
               vtln_high: float=-500.,
               vtln_low: float=100.,
               vtln_map: str='',
               vtln_warp: float=1.,
               window_type: str='povey') -> numpy.ndarray:
    """Computes mfcc using 'compute-mfcc-feats' command
    
    Args:
        wave: Input wav data as numpy.ndarray
        axis (int): Axis along which the spectrogram is computed; the default 
            is over the last axis (i.e. axis=-1).
        blackman_coeff :Constant coefficient for generalized Blackman window.
        cepstral_lifter :Constant that controls scaling of MFCCs
        dither :Dithering constant (0.0 means no dither)
        energy_floor :Floor on energy (absolute, not relative)
            in MFCC computation
        frame_length_ms :Frame length in milliseconds
        frame_shift_ms :Frame shift in millisecond
        high_freq :High cutoff frequency for mel bins
            (if < 0, offset from Nyquist)
        htk_compat :If true, put energy or C0 last and use a factor of sqrt(2) on C0.
            Warning: not sufficient to get HTK compatible features
            (need to change other parameters).
        low_freq :Low cutoff frequency for mel bins
        num_ceps :Number of cepstra in MFCC computation (including C0)
        num_mel_bins :Number of triangular mel_frequency bins
        preemphasis_coefficient :Coefficient for use in signal preemphasis
        raw_energy :If true, compute energy before preemphasis and windowing
        remove_dc_offset :Subtract mean from waveform on each frame
        round_to_power_of_two :If true, round window size to power of two.
        sample_frequency (float):Waveform data sample frequency
        snip_edges :If True, end effects will be handled by outputting
            only frames that completely fit in the file,
            and the number of frames depends on the frame-length.
        subtract_mean :Subtract mean of each feature file [CMS];
            not recommended to do it this way.
        use_energy (bool):Use energy (not C0) in MFCC computation
        utt2spk (str): Utterance to speaker-id map 
            (if doing VTLN and you have warps per speaker)
        vtln_high (float): 
            High inflection point in piecewise linear VTLN warping function
             (if negative, offset from high-mel-freq
        vtln_low (float): Low inflection point in piecewise linear VTLN warping
            function
        vtln_map (str): Map from utterance or speaker-id to vtln warp factor
            (rspecifier)
        vtln_warp (float): Vtln warp factor 
            (only applicable if vtln-map not specified)
        window_type :Type of window
            ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann")
            
    Returns:
        The extracted mfcc as numpy.ndarray
    """
    cmd = ['compute-mfcc-feats',
           f'--blackman-coeff={blackman_coeff}',
           f'--cepstral-lifter={cepstral_lifter}',
           f'--dither={dither}',
           f'--energy-floor={energy_floor}',
           f'--frame-length={frame_length_ms}',
           f'--frame-shift={frame_shift_ms}',
           f'--high-freq={high_freq}',
           f'--htk-compat={str(htk_compat).lower()}',
           f'--low-freq={low_freq}',
           f'--num-ceps={num_ceps}',
           f'--num-mel-bins={num_mel_bins}',
           f'--preemphasis-coefficient={preemphasis_coefficient}',
           f'--raw-energy={str(raw_energy).lower()}',
           f'--remove-dc-offset={str(remove_dc_offset).lower()}',
           f'--round-to-power-of-two={str(round_to_power_of_two).lower()}',
           f'--sample-frequency={int(sample_frequency)}',
           f'--snip-edges={str(snip_edges).lower()}',
           f'--subtract-mean={str(subtract_mean).lower()}',
           f'--use-energy={str(use_energy).lower()}',
           f'--utt2spk={utt2spk}',
           f'--vtln-high={vtln_high}',
           f'--vtln-low={vtln_low}',
           f'--vtln-map={vtln_map}',
           f'--vtln-warp={vtln_warp}',
           f'--window-type={window_type}',
           'ark:-', 'ark:-']
    if wave.ndim >= 2:
        raise ValueError('Input matrix or vector(1 < wave.ndim < 2)')
    if wave.ndim == 1:
        if axis not in (0, -1):
            raise ValueError(
                f'The input axis {axis} '
                'is invalid for the array shape: {wave.shape}')
    else:
        wave = wave[axis]
    _check_length(len(wave),
                  frame_length_ms, frame_shift_ms, sample_frequency,
                  snip_edges)
    return _tpl_command(wave, cmd, sample_frequency)


def fbank_feats(wave: numpy.ndarray,
                axis: int=-1,
                blackman_coeff: float=0.42,
                dither: float=1.,
                energy_floor: float=0.,
                frame_length_ms: float=25.,
                frame_shift_ms: float=10.,
                high_freq: float=0.,
                htk_compat: bool=False,
                low_freq: float=20.,
                num_mel_bins: int=23,
                preemphasis_coefficient: float=0.97,
                raw_energy: bool=True,
                remove_dc_offset: bool=True,
                round_to_power_of_two: bool=True,
                sample_frequency: float=16000.,
                snip_edges: bool=True,
                subtract_mean: bool=False,
                use_energy: bool=False,
                use_log_fbank: bool=True,
                use_power: bool=True,
                utt2spk: str='',
                vtln_high: float=-500.,
                vtln_low: float=100.,
                vtln_map: str='',
                vtln_warp: float=1.,
                window_type: str='povey') -> numpy.ndarray:
    """Compute Mel-filter bank feature

    Args:
        wave: Input wav data as numpy.ndarray
        axis (int): Axis along which the spectrogram is computed; the default 
            is over the last axis (i.e. axis=-1).
        blackman_coeff (flaot): 
            Constant coefficient for generalized Blackman window. 
        dither (float): Dithering constant (0.0 means no dither)
        energy_floor (float): Floor on energy (absolute, not relative) 
            in FBANK computation
        frame_length_ms (float): Frame length in milliseconds
        frame_shift_ms (float): Frame shift in milliseconds 
        high_freq (float): High cutoff frequency for mel bins 
            (if < 0, offset from Nyquist) 
        htk_compat (bool): If true, put energy last.  
            Warning: not sufficient to get HTK compatible features
            (need to change other parameters). 
        low_freq (float): Low cutoff frequency for mel bins
        num_mel_bins (int): Number of triangular mel_frequency bins
        preemphasis_coefficient (float): Coefficient for use 
            in signal preemphasis 
        raw_energy (bool): If true, compute energy before 
            preemphasis and windowing
        remove_dc_offset (bool): Subtract mean from waveform on each frame 
        round_to_power_of_two (bool): 
            If true, round window size to power of two. 
        sample_frequency (float): Waveform data sample frequency
        snip_edges (bool): If true, end effects will be handled 
            by outputting only frames that completely fit in the file, 
            and the number of frames depends on the frame_length. 
            If false, the number of frames depends only on the frame_shift, 
            and we reflect the data at the ends.
        subtract_mean (bool): Subtract mean of each feature file [CMS]; 
            not recommended to do it this way. 
        use_energy (bool): Add an extra dimension with energy to 
            the FBANK output. 
        use_log_fbank (bool): If true, produce log_filterbank,
            else produce linear.
        use_power (bool): If true, use power, else use magnitude.
        utt2spk (str): Utterance to speaker-id map 
            (if doing VTLN and you have warps per speaker)
        vtln_high (float): 
            High inflection point in piecewise linear VTLN warping function
             (if negative, offset from high-mel-freq
        vtln_low (float): Low inflection point in piecewise linear VTLN warping
            function
        vtln_map (str): Map from utterance or speaker-id to vtln warp factor
            (rspecifier)
        vtln_warp (float): Vtln warp factor 
            (only applicable if vtln-map not specified)
        window_type : Type of window
            ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann")
    """
    cmd = ['compute-fbank-feats',
           f'--blackman-coeff={blackman_coeff}',
           f'--dither={dither}',
           f'--energy-floor={energy_floor}',
           f'--frame-length={frame_length_ms}',
           f'--frame-shift={frame_shift_ms}',
           f'--high-freq={high_freq}',
           f'--htk-compat={str(htk_compat).lower()}',
           f'--low-freq={low_freq}',
           f'--num-mel-bins={num_mel_bins}',
           f'--preemphasis-coefficient={preemphasis_coefficient}',
           f'--raw-energy={str(raw_energy).lower()}',
           f'--remove-dc-offset={str(remove_dc_offset).lower()}',
           f'--round-to-power-of-two={str(round_to_power_of_two).lower()}',
           f'--sample-frequency={sample_frequency}',
           f'--snip-edges={str(snip_edges).lower()}',
           f'--subtract-mean={str(subtract_mean).lower()}',
           f'--use-energy={str(use_energy).lower()}',
           f'--use-log-fbank={str(use_log_fbank).lower()}',
           f'--use-power={str(use_power).lower()}',
           f'--utt2spk={utt2spk}',
           f'--vtln-high={vtln_high}',
           f'--vtln-low={vtln_low}',
           f'--vtln-map={vtln_map}',
           f'--vtln-warp={vtln_warp}',
           f'--window-type={window_type}',
           'ark:-', 'ark:-']
    if wave.ndim >= 2:
        raise ValueError('Input matrix or vector(1 < wave.ndim < 2)')
    if wave.ndim == 1:
        if axis not in (0, -1):
            raise ValueError(
                f'The input axis {axis} '
                'is invalid for the array shape: {wave.shape}')
    else:
        wave = wave[axis]
    _check_length(len(wave),
                  frame_length_ms, frame_shift_ms, sample_frequency,
                  snip_edges)
    return _tpl_command(wave, cmd, sample_frequency)


def spectrogram_feats(wave: numpy.ndarray,
                      axis: int=-1,
                      blackman_coeff: float=0.42,
                      dither: float=1.0,
                      energy_floor: float=0.0,
                      frame_length_ms: float=25,
                      frame_shift_ms: float=10.,
                      preemphasis_coefficient: float=0.97,
                      raw_energy: bool=True,
                      remove_dc_offset: bool=True,
                      round_to_power_of_two: bool=True,
                      sample_frequency: float=16000.,
                      snip_edges: bool=True,
                      subtract_mean: bool=False,
                      window_type: str='povey') -> numpy.ndarray:
    """Invoke 'compute-spectrogram-feats' and
    return its retrun values as numpy.ndarray
    
    Args:
        wave (numpy.ndarray):
        axis (int): Axis along which the spectrogram is computed; the default 
            is over the last axis (i.e. axis=-1).
        blackman_coeff (flaot): Constant coefficient 
            for generalized Blackman window. 
        dither (float): Dithering constant (0.0 means no dither)
        energy_floor (float): Floor on energy 
            (absolute, not relative) in Spectrogram computation
        frame_length_ms (float): Frame length in milliseconds
        frame_shift_ms (float): Frame shift in milliseconds
        preemphasis_coefficient (float): 
            Coefficient for use in signal preemphasis
        raw_energy (bool): If true, compute energy before preemphasis 
            and windowing
        remove_dc_offset (bool): Subtract mean from waveform on each frame
        round_to_power_of_two (bool): 
            If true, round window size to power of two. 
        sample_frequency (float): Waveform data sample frequency
            (must match the waveform file, if specified there)
        snip_edges (True): If true, end effects will be handled 
            by outputting only frames that completely fit in the file, 
            and the number of frames depends on the frame-length. 
            If false, the number of frames depends only on the frame-shift, 
            and we reflect the data at the ends.
        subtract_mean (False): 
            Subtract mean of each feature file [CMS]; 
            not recommended to do it this way.
        window_type (str): Type of window 
            ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann")
            
    Example:
        >>> wave = (numpy.sin(numpy.arange(1000)) * 1000).astype(numpy.int16)
        >>> spec = spectrogram_feats(wave)
    """
    cmd = ['compute-spectrogram-feats',
           f'--blackman-coeff={blackman_coeff}',
           f'--dither={dither}',
           f'--energy-floor={energy_floor}',
           f'--frame-length={frame_length_ms}',
           f'--frame-shift={frame_shift_ms}',
           f'--preemphasis-coefficient={preemphasis_coefficient}',
           f'--raw-energy={str(raw_energy).lower()}',
           f'--remove-dc-offset={str(remove_dc_offset).lower()}',
           f'--round-to-power-of-two={str(round_to_power_of_two).lower()}',
           f'--sample-frequency={sample_frequency}',
           f'--snip-edges={str(snip_edges).lower()}',
           f'--subtract-mean={str(subtract_mean).lower()}',
           f'--window-type={window_type}',
           'ark:-', 'ark:-']
    if wave.ndim >= 2:
        raise ValueError('Input matrix or vector(1 <= wave.ndim <= 2)')
    if wave.ndim == 1:
        if axis not in (0, -1):
            raise ValueError(
                f'The input axis {axis} '
                'is invalid for the array shape: {wave.shape}')
    else:
        wave = wave[axis]
    _check_length(len(wave),
                  frame_length_ms, frame_shift_ms, sample_frequency,
                  snip_edges)
    return _tpl_command(wave, cmd, sample_frequency)
