import numpy as np
import pytest
import scipy.stats

import kaldifeats.commands as C
import kaldifeats.signal as S


@pytest.mark.parametrize('length', [1000,
                                    4503,
                                    6290,
                                    ])
def test_fbank(length):
    x = (np.random.randn(length,) * 1000).astype(np.int16)

    keywords = {'window_type': 'povey',
                'dither': 0.0,
                'snip_edges': False,
                'preemphasis_coefficient': 0.97,
                'round_to_power_of_two': True,
                }
    spec, energy = S.spectrogram(x[:],
                                 frame_length=400,
                                 frame_shift=160,
                                 mode='kaldi_power',
                                 detrend='constant',
                                 return_raw_energy=True,
                                 return_energy=False,
                                 **keywords)
    spec[..., 0] = energy
    spec = np.log(np.maximum(spec, np.finfo(spec.dtype).eps))
    spec2 = C.spectrogram_feats(x[:],
                                frame_length_ms=25,
                                frame_shift_ms=10,
                                remove_dc_offset=True,
                                raw_energy=True,
                                sample_frequency=16000,
                                **keywords)

    diff = spec - spec2
    print(f'Diff stats: {scipy.stats.describe(diff, axis=None)}')
    np.testing.assert_allclose(spec, spec2, rtol=1e-01)
