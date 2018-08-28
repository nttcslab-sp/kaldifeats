# kaldifeats
- kaldifeats.commands:

A wrapper in python modules to invoke commad line tools of `Kaldi`, ensuring 100% reproducibility of kaldi features, of course.

- kaldifeats.signal:

Pure python implementations to perform `stft`.
Actually, instead of Kaldi, it depends on scipy for some useful signal funciton, e.g. `scipy.signal.get_window`,
but not using the highest function, `scipy.signal.stft`, `scipy.signal.spectrogram`.

- kaldifeats.feats:

Pure python modules for audio feature extractor, e.g. `log-fbank-feats`, `mfcc`, `add-deltas`.

## Requirements
```
python3.6
scipy
numpy
kaldiio
```

## TODO
- Implement istft(just to use scipy.signal.istft with sligtly changes)


## Ref

- https://github.com/scipy/scipy/blob/master/scipy/signal/spectral.py
- https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc
- https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-spectrogram.cc

## Related projects
- https://github.com/librosa/librosa
  - Depends on scipy and [audioread](https://github.com/beetbox/audioread)

- https://github.com/MTG/essentia
  - C++ implementation and python binding
  - Supports loader for audio file depending libavcodec/libavformat/libavutil/libavresample(FFmpeg/LibAv)
  - Not supporting Python3 (https://github.com/MTG/essentia/issues/138)

- https://github.com/jameslyons/python_speech_features
  - Simple implementation
