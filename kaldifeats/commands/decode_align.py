import numpy

from kaldifeats.commands.align import align
from kaldifeats.commands.decode import decode_faster_mapped


def decode_align(
    array: numpy.ndarray,
    transition_model: str,
    fst: str,
    lexicon_fst: str,
    tree: str,
    acoustic_scale: float=0.1,
    allow_partial: bool=True,
    beam: float=16.,
    beam_delta: float=0.5,
    hash_ratio: float=2.,
    max_active: int=2147483647,
    min_active: int=20,
    word_symbol_table: str=None,
    batch_size: int=250,
    read_disambig_syms: str=None,
    reorder: bool=True,
    rm_eps: bool=False,
    transition_scale=1.0,
    self_loop_scale=0.1,
    ali_beam: float=200.,
    binary: bool=True,
    careful: bool=False,
    retry_beam: float=0,
    verbose: int=-1,
        ) -> numpy.ndarray:
    """
    
    Args:
        array (numpy.ndarray): Log likelihood
        transition_model (str): Input transition model
        fst (fst): $graph/HCLG.fst
        lexicon_fst (fst): lexicon-fst-in("$lang/L.fst")
        tree (str): tree-in
        acoustic_scale: Scaling factor for acoustic likelihoods
        allow_partial: Produce output even when final state was not reached
        beam: Decoding beam.  Larger->slower, more accurate.
        beam_delta: Increment used in decoder [obscure setting]
        hash_ratio: Setting used in decoder to control hash behavior
        max_active: Decoder max active states.  Larger->slower; more accurate
        min_active: Decoder min active states 
            (don't prune if #active less than this). 
        word_symbol_table: Symbol table for words [for debug output] 
        batch_size (int): Number of FSTs to compile at a time
            (more -> faster but uses more memory.  E.g. 500
        read_disambig_syms (str): File containing list of disambiguation
            symbols in phone symbol table, $lang/phones/disambig.int
        reorder (bool): Reorder transition ids for greater decoding
            efficiency.
        rm_eps (bool) : Remove [most] epsilons before minimization
            (only applicable if disambig symbols present)
        transition_scale (float):    
        self_loop_scale (float):
        ali_beam (float): Decoding beam used in alignment
        binary (bool): Write output in binary mode
        careful (bool): If true, do 'careful' alignment,
            which is better at detecting alignment failure
            (involves loop to start of decoding graph).
        retry_beam (float): Decoding beam for second try at alignment       
        verbose (int): Verbose level (higher->more logging)
    """
    words = decode_faster_mapped(
        array=array,
        transition_model=transition_model,
        fst=fst,
        acoustic_scale=acoustic_scale,
        allow_partial=allow_partial,
        beam=beam,
        beam_delta=beam_delta,
        hash_ratio=hash_ratio,
        max_active=max_active,
        min_active=min_active,
        word_symbol_table=word_symbol_table,
        verbose=verbose
        )

    return align(
        array=array,
        transcription=words,
        tree=tree,
        transition_model=transition_model,
        fst=lexicon_fst,
        batch_size=batch_size,
        read_disambig_syms=read_disambig_syms,
        reorder=reorder,
        rm_eps=rm_eps,
        transition_scale=transition_scale,
        acoustic_scale=acoustic_scale,
        self_loop_scale=self_loop_scale,
        beam=ali_beam,
        binary=binary,
        careful=careful,
        retry_beam=retry_beam,
        verbose=verbose)

