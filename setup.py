#!/usr/bin/env python

from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Python library for Multi-Armed Bandits algorithms',
      author='Binh Nguyen',
      author_email='nguyenthaibinh@gmail.com',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['pymab', 'pymab.algorithms.thompson.bernulli_thompson'],
    )