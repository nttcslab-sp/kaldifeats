#!/usr/bin/env python
import os

from setuptools import setup

setup(name='kaldifeats',
      version='1.11.0',
      description='Create kaldi compatible audio features',
      author='Naoyuki Kamo',
      author_email='kamo_naoyuki_t7@lab.ntt.co.jp',
      packages=['kaldifeats',
                'kaldifeats.feats',
                'kaldifeats.signal',
                'kaldifeats.utils',
                'kaldifeats.commands'
                ],
      long_description=open(os.path.join(os.path.dirname(__file__),
                            'README.md'), 'r', encoding='utf-8').read(),
      install_requires=['kaldiio>=2.0',
                        'scipy'],
      setup_requires=['numpy', 'pytest-runner'],
      tests_require=['pytest-cov', 'pytest-html', 'pytest'])
