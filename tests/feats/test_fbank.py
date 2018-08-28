import numpy as np
import pytest
import scipy.stats

import kaldifeats.commands as C
import kaldifeats.feats as F


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
                'use_energy': False,
                'raw_energy': False,
                'num_mel_bins': 60,
                'sample_frequency': 16000,
                'round_to_power_of_two': True,
                }
    fbank = F.fbank_feats(x[:],
                          frame_length=400,
                          frame_shift=160,
                          mode='kaldi_power',
                          detrend='constant',
                          **keywords)
    fbank2 = C.fbank_feats(x[:],
                           frame_length_ms=25,
                           frame_shift_ms=10,
                           remove_dc_offset=True,
                           **keywords)

    diff = fbank - fbank2
    print(f'Diff stats: {scipy.stats.describe(diff, axis=None)}')
    np.testing.assert_allclose(fbank, fbank2, rtol=1e-01)

