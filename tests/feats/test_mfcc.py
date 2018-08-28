import numpy as np
import pytest
import scipy.stats

from kaldifeats import feats as F, commands as C


@pytest.mark.parametrize('length', [1000,
                                    4503,
                                    6290,
                                    ])
def test_mfcc(length):
    x = (np.random.randn(length,) * 1000).astype(np.int16)

    keywords = {'window_type': 'povey',
                'dither': 0.0,
                'snip_edges': False,
                'preemphasis_coefficient': 1.00,
                'use_energy': False,
                'raw_energy': False,
                'num_mel_bins': 60,
                'sample_frequency': 16000,
                'num_ceps': 40,
                'cepstral_lifter': 22.,
                'round_to_power_of_two': True,
                }
    mfcc = F.mfcc_feats(x[:],
                        frame_length=400,
                        frame_shift=160,
                        mode='kaldi_power',
                        detrend='constant',
                        **keywords)
    mfcc2 = C.mfcc_feats(x[:],
                         frame_length_ms=25,
                         frame_shift_ms=10,
                         remove_dc_offset=True,
                         **keywords)

    diff = mfcc - mfcc2
    print(f'Diff stats: {scipy.stats.describe(diff, axis=None)}')
    np.testing.assert_allclose(mfcc, mfcc2, rtol=1e-01)
