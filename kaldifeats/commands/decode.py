import errno
import shutil
from io import BytesIO
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Union

import kaldiio
import numpy

for _cmd in ['latgen-faster-mapped']:
    if shutil.which(_cmd) is None:
        raise RuntimeError(
            f'Command not found: {_cmd}. This tool depends on kaldi')

DUMMY_KEY = 'kaldifeats_commands_decode'


def general_kaldi_commands(cmds,
                           array,
                           verbose: int=-1) -> bytes:
    if shutil.which(cmds[0]) is None:
        raise RuntimeError(f'Command not found: {cmds[0]}')
    with Popen(cmds, stdin=PIPE, stdout=PIPE,
               stderr=None if verbose > -1 else PIPE,
               bufsize=-1) as p:
        try:
            kaldiio.save_ark(p.stdin, {DUMMY_KEY: array})
        except BrokenPipeError:
            pass  # communicate() must ignore broken pipe errors.
        except OSError as e:
            if e.errno == errno.EINVAL and p.poll() is not None:
                # Issue #19612: On Windows, stdin.write() fails with EINVAL
                # if the process already exited before the write
                pass
            else:
                raise
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            if stderr is not None:
                ms = stderr.decode()
            else:
                ms = f'Fail: {" ".join(cmds)}'
            raise RuntimeError(ms)
        return stdout


def latgen_faster_mapped(
        array: numpy.ndarray,
        transition_model: str,
        fst: str,
        acoustic_scale: float=0.1,
        allow_partial: bool=False,
        beam: float=16,
        beam_delta: float=0.5,
        delta: float=0.000976562,
        determinize_lattice: bool=True,
        hash_ratio: float=2,
        lattice_beam: float=10,
        max_active: int=2147483647,
        max_mem: int=50000000,
        min_active: int=200,
        minimize: bool=False,
        phone_determinize: bool=True,
        prune_interval: int=25,
        word_determinize: bool=True,
        word_symbol_table: str=None,
        align: bool=False,
        as_text: bool=False,
        verbose: int=-1,
        ) -> Union[bytes, numpy.ndarray]:
    """
    Generate lattices, reading log-likelihoods as matrices, using multiple 
    decoding threads
     (model is needed only for the integer mappings in its transition-model)
     
    This functions returns raw string representing lat file of Kaldi
     
    Example:
        >>> import kaldiio
        >>> gen = kaldiio.load_ark('log_likelihood.ark')
        >>> d = {}
        >>> for uttid, array in gen:
        ...     lat_rawbytes = latgen_faster_mapped(
        ...         array, 
        ...         transition_model='exp/gmm/final.mdl', 
        ...         fst='graph/HCLG.fst')
        ...     d[uttid] = lat_rawbytes 
        >>> import gzip
        >>> with gzip.open('lat.ark.gz', 'wb') as f:
        ...     kaldiio.save_ark(f, d, as_bytes=True)

    Args:
        array (numpy.ndarray): Log likelihood
        transition_model (str): Input transition model
        fst (fst): Input fst file
        acoustic_scale (float): Scaling factor for acoustic likelihoods
        allow_partial (bool): If true, produce output even if end state was 
            not reached.
        beam (float): Decoding beam.  Larger->slower, more accurate.
        beam_delta  (float): Increment used in decoding
         --- this parameter is obscure and relates to a speedup in the way
         the max-active constraint is applied.  Larger is more accurate.
        delta (float): Tolerance used in determinization
        determinize_lattice (bool):If true, determinize the lattice
            (lattice-determinization, keeping only best pdf-sequence 
            for each word-sequence).
        hash_ratio (float): Setting used in decoder to control hash behavior
        lattice_beam (float): Lattice generation beam.  Larger->slower, 
            and deeper lattices
        max_active (int): Decoder max active states.  
            Larger->slower; more accurate
        max_mem (int): Maximum approximate memory usage in determinization 
            (real usage might be many times this).
        min_active (int): Decoder minimum #active states.
        minimize (bool): If true, push and minimize after determinization.
        phone_determinize (bool): If true, do an initial pass of 
            determinization on both phones and words
            (see also --word-determinize)
        prune_interval (int): Interval (in frames) at which to prune tokens
        word_determinize (bool): 
            If true, do a second pass of determinization on words only 
            (see also --phone-determinize)
        word_symbol_table (str): Symbol table for words [for debug output]
        align (bool): Return the alignment as ndarray instead of lattice
        as_text (bool): If True output as text format
        verbose (int): Verbose level (higher->more logging)
    Returns:
        Raw bytes string
    """
    if word_symbol_table is None:
        word_symbol_table = ''
    cmds = [
        'latgen-faster-mapped',
        f'--acoustic-scale={acoustic_scale}',
        f'--allow-partial={str(allow_partial).lower()}',
        f'--beam={beam}',
        f'--beam-delta={beam_delta}',
        f'--delta={delta}',
        f'--determinize_lattice={str(determinize_lattice).lower()}',
        f'--lattice-beam={lattice_beam}',
        f'--max-active={max_active}',
        f'--max-mem={max_mem}',
        f'--min-active={min_active}',
        f'--minimize={str(minimize).lower()}',
        f'--hash_ratio={hash_ratio}',
        f'--phone-determinize={str(phone_determinize).lower()}',
        f'--prune-interval={prune_interval}',
        f'--word_determinize={str(word_determinize).lower()}',
        f'--word-symbol-table={word_symbol_table}',
        f'--verbose={max(verbose, 0)}',
        f'{transition_model}',
        f'{fst}', 'ark:-']

    # Only one of lat and ali can be generate onece
    # because stdout is only one...
    # If you don't prefer, hack this code to use tempfile and write in it.
    if align:
        cmds += [f'ark:/dev/null',  f'ark:-']
    else:
        cmds.append('ark,t:-' if as_text else 'ark:-')

    for f in (transition_model, fst) + \
            (word_symbol_table,) if word_symbol_table != '' else ():
        assert isinstance(f, str), f
        if not Path(f).exists():
            raise FileNotFoundError(f'{f}: No such file or directory')
    assert isinstance(array, numpy.ndarray), \
        f'Expect ndarray, but got {type(array)}'

    assert array.ndim == 2, f'2 != {array.ndim}'
    out = general_kaldi_commands(cmds, array, verbose)

    if align:
        with BytesIO(out) as f:
            # Return as numpy.ndarray
            return next(kaldiio.load_ark(f))[1]
    else:
        # Return as bytes
        return out[len(DUMMY_KEY) + 1:]


def decode_faster_mapped(
    array: numpy.ndarray,
    transition_model: str,
    fst: str,
    acoustic_scale: float=0.1,
    allow_partial: bool=True,
    beam: float=16.,
    beam_delta: float=0.5,
    hash_ratio: float=2.,
    max_active: int=2147483647,
    min_active: int=20,
    word_symbol_table: str=None,
    verbose: int=-1
    ) -> numpy.ndarray:
    """
    
    Args:
        array (numpy.ndarray): Log likelihood
        transition_model (str): Input transition model
        fst (fst): Input fst file
        acoustic_scale: Scaling factor for acoustic likelihoods
        allow_partial: Produce output even when final state was not reached
        beam: Decoding beam.  Larger->slower, more accurate.
        beam_delta: Increment used in decoder [obscure setting]
        hash_ratio: Setting used in decoder to control hash behavior
        max_active: Decoder max active states.  Larger->slower; more accurate
        min_active: Decoder min active states 
            (don't prune if #active less than this). 
        word_symbol_table: Symbol table for words [for debug output] 
        verbose (int): Verbose level (higher->more logging)
    """
    if word_symbol_table is None:
        word_symbol_table = ''

    cmds = [
        'decode-faster-mapped',
        f'--acoustic-scale={acoustic_scale}',
        f'--allow-partial={str(allow_partial).lower()}',
        f'--beam={beam}',
        f'--beam-delta={beam_delta}',
        f'--max-active={max_active}',
        f'--min-active={min_active}',
        f'--hash_ratio={hash_ratio}',
        f'--word-symbol-table={word_symbol_table}',
        f'--verbose={max(verbose, 0)}',
        f'{transition_model}',
        f'{fst}', 'ark:-', 'ark:-']

    for f in (transition_model, fst) + \
            (word_symbol_table,) if word_symbol_table != '' else ():
        assert isinstance(f, str), f
        if not Path(f).exists():
            raise FileNotFoundError(f'{f}: No such file or directory')
    assert isinstance(array, numpy.ndarray), \
        f'Expect ndarray, but got {type(array)}'

    assert array.ndim == 2, f'2 != {array.ndim}'

    with Popen(cmds, stdin=PIPE, stdout=PIPE,
               stderr=None if verbose > -1 else PIPE,
               bufsize=-1) as p:
        try:
            kaldiio.save_ark(p.stdin, {DUMMY_KEY: array})
        except BrokenPipeError:
            pass  # communicate() must ignore broken pipe errors.
        except OSError as e:
            if e.errno == errno.EINVAL and p.poll() is not None:
                # Issue #19612: On Windows, stdin.write() fails with EINVAL
                # if the process already exited before the write
                pass
            else:
                raise
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            if stderr is not None:
                ms = stderr.decode()
            else:
                ms = f'Fail: {" ".join(cmds)}'
            raise RuntimeError(ms)

        with BytesIO(stdout) as f:
            return next(kaldiio.load_ark(f))[1]
