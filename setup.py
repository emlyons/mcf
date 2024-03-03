#!/usr/bin/env python

# This file is used by Pip to install the Tethys package, e.g.: python3 -m pip install .

from distutils.core import setup

setup(name='MCF',
      version='0.0.1',
      description='motion compensated filtering through target tracking',
      author='Ethan Lyons',
      packages=['mcf'],
      )
