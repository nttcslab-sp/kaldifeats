import collections.abc
import errno
import shutil
from io import BytesIO
from typing import Sequence, Union
from pathlib import Path

import kaldiio
import numpy
from subprocess import Popen, PIPE

for _cmd in ['compile-train-graphs', 'align-compiled-mapped']:
    if shutil.which(_cmd) is None:
        raise RuntimeError(
            f'Command not found: {_cmd}. This tool depends on kaldi')

DUMMY_KEY = 'kaldifeats_commands_align'


def align(
        array: numpy.ndarray,
        transcription: Union[Sequence[int], numpy.ndarray],
        tree: str,
        transition_model: str,
        fst: str,
        batch_size: int=250,
        read_disambig_syms: str=None,
        reorder: bool=True,
        rm_eps: bool=False,
        transition_scale=1.0,
        acoustic_scale: float=1,
        self_loop_scale=0.1,
        beam: float=200,
        binary: bool=True,
        careful: bool=False,
        retry_beam: float=0,
        verbose: int=-1) -> numpy.ndarray:
    """
    compile-train-graph + align-compiled-mapped
    Args:
        array (numpy.ndarray):
        transcription (Sequence[int]): A sequnce of integer representing
            the transcription of a utterance.
            It can be get by this command,
            % utils/sym2int.pl --map-oov $(cat $lang/oov.int)
              -f 2- $lang/words.txt $text |"
        tree (str): tree-in
        transition_model (str): transition-model-in("final.mdl")
        fst (str): lexicon-fst-in("$lang/L.fst")
        batch_size (int): Number of FSTs to compile at a time
            (more -> faster but uses more memory.  E.g. 500
        read_disambig_syms (str): File containing list of disambiguation
            symbols in phone symbol table, $lang/phones/disambig.int
        reorder (bool): Reorder transition ids for greater decoding
            efficiency.
        rm_eps (bool) : Remove [most] epsilons before minimization
            (only applicable if disambig symbols present)
        transition_scale (float):
        acoustic_scale (float): Scaling factor for acoustic likelihoods
        self_loop_scale (float):
        beam (float): Decoding beam used in alignment
        binary (bool): Write output in binary mode
        careful (bool): If true, do 'careful' alignment,
            which is better at detecting alignment failure
            (involves loop to start of decoding graph).
        retry_beam (float): Decoding beam for second try at alignment
        verbose (int): Verbose level (higher->more logging)
    """
    for cmd in ['compile-train-graphs', 'align-compiled-mapped']:
        if shutil.which(cmd) is None:
            raise RuntimeError(
                f'Command not found: {cmd}')
    if isinstance(transcription, numpy.ndarray) and \
            len(transcription.shape) == 1 and transcription.dtype.kind == 'i':
        transcription = [int(v) for v in transcription]
    if not isinstance(transcription, collections.abc.Sequence) or\
            any(not isinstance(v, int) for v in transcription):
        raise TypeError(f'{type(transcription)} is not a sequence of integer')

    if read_disambig_syms is None:
        read_disambig_syms = ''
    for f in [tree, transition_model, fst] +\
            [read_disambig_syms] if read_disambig_syms != '' else []:
        if not Path(f).exists():
            raise FileNotFoundError(f'{f}: No such file or directory')

    if array.ndim != 2:
        raise ValueError(
            'The input posterior must be 2-dimension array: '
            f'{array.ndim} != 2')

    if len(array) < len(transcription):
        raise ValueError(
            f'The frame length of the input likehood array '
            f'is shorter than the length of the transcription. '
            f'{len(array)} < {len(transcription)}')

    # Don't support self-loop-scale and transition-scale
    # because I don't understand how to input...
    compile_train_graph = (
        f'echo {DUMMY_KEY} {" ".join(map(str, transcription))} | '
        f'compile-train-graphs '
        f'--batch-size={batch_size} '
        f'--read-disambig-syms={read_disambig_syms} '
        f'--reorder={str(reorder).lower()} '
        f'--rm-eps={str(rm_eps).lower()} '
        f'--verbose={max(verbose, 0)} '
        f'{tree} {transition_model} {fst} ark:- ark:- |')

    cmds = ['align-compiled-mapped',
            f'--transition-scale={transition_scale}',
            f'--self-loop-scale={self_loop_scale}',
            f'--acoustic-scale={acoustic_scale}',
            f'--beam={beam}',
            f'--binary={str(binary).lower()}',
            f'--careful={str(careful).lower()}',
            f'--retry-beam={retry_beam}',
            f'--verbose={max(verbose, 0)}',
            f'{transition_model}',
            f'ark:{compile_train_graph}', 'ark:-', 'ark:-']

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
        fout = BytesIO(stdout)
        return next(kaldiio.load_ark(fout))[1]
