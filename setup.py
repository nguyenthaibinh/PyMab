#!/usr/bin/env python

from distutils.core import setup

setup(name='PyMAB',
      version='1.0',
      description='Python library for Multi-Armed Bandits algorithms',
      author='Binh Nguyen',
      author_email='nguyenthaibinh@gmail.com',
      url='http://github.com/nguyenthaibinh/pymab',
      packages=['pymab', 'pymab.algorithms.thompson', 'pymab.algorithms.greedy', 'pymab.algorithms.softmax', 'pymab.arms'],
    )