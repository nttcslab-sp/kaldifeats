![python-version](https://img.shields.io/badge/python-3.6-blue.png)

# KamoKaldi
- kamokaldi.commands:

A wrapper in python modules to invoke commad line tools of `Kaldi`, ensuring 100% reproducibility of kaldi features, of course.

- kamokaldi.signal:

Pure python implementations to perform `stft`.
Actually, instead of Kaldi, it depends on scipy for some useful signal funciton, e.g. `scipy.signal.get_window`,
but not using the highest function, `scipy.signal.stft`, `scipy.signal.spectrogram`.

- kamokaldi.feats:

Pure python modules for audio feature extractor, e.g. `log-fbank-feats`, `mfcc`, `add-deltas`.

## Instal
```
export PIP_EXTRA_INDEX_URL=http://kishin-gitlab.cslab.kecl.ntt.co.jp:8001/simple/
export PIP_TRUSTED_HOST=kishin-gitlab.cslab.kecl.ntt.co.jp
pip install kamokaldi
```

## Requirements
```
python3.6
scipy
numpy
kaldiio # At kishin pypi server
```

## How to use
See http://kishin-gitlab.cslab.kecl.ntt.co.jp/kamo/kamo_notes/blob/master/kamokaldi

## TODO
- Implement istft(just to use scipy.signal.istft with sligtly changes)

## Comparison of scipy.signal.stft & Kaldi
||scipy.signal.stft| kaldi| kamokaldi.signal|
|---|---|---|---|
|Add "constant"/"zero"/etc boundary&Padded|✓|x|✓|
|Add reflected values for boundary|x|✓|✓|
|Snip edges(no boundary&no padded)|✓|✓|✓|
|Dithering|x|✓|✓|
|Remove dc offset(equivalent to "constant" detrend)|✓|✓|✓|
|Povey(a window function)|x|✓|✓|
|Pre-emphasis filter|x|✓|✓|
|Add energy(sum of squared window)|x|✓|✓|
|Round up window length to power of 2|x|✓|✓|

The execution order is the same as the rows order, so

1. Dealing edges(Add boundary&padded@scipy, Add reflected values@kaldi)
1. Dithering
1. Remove dc offset(Detrend)
1. Multiply window matrix
1. Apply Pre-emphasis filter
1. FFT

```
# povey window is like hamming but goes to zero at edges
hamming: 0.54 - 0.46*cos(a * i_fl)
povey: pow(0.5 - 0.5*cos(a * i_fl), 0.85)
```

# Ref

- https://github.com/scipy/scipy/blob/master/scipy/signal/spectral.py
- https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc
- https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-spectrogram.cc

# Related projects
- https://github.com/librosa/librosa
  - Depends on scipy and [audioread](https://github.com/beetbox/audioread)
  - Not compatible with HTK? (https://github.com/georgid/mfcc-htk-an-librosa/blob/master/htk%20and%20librosa%20MFCC%20extract%20comparison.ipynb)

- https://github.com/MTG/essentia
  - C++ implementation and python binding
  - Completely reproduce HTK feats(https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_mfcc_the_htk_way.py)
  - Supports loader for audio file depending libavcodec/libavformat/libavutil/libavresample(FFmpeg/LibAv)
  - Not easy to build on centos because some depended library cannot be installed by yum, e.g. ffmpeg-devel
  - Not supporting Python3 (https://github.com/MTG/essentia/issues/138)

- https://github.com/jameslyons/python_speech_features
  - Simple implementation
