#! /usr/bin/env python
#
# Copyright (c) 2018 Princeton Neuroscience Institute
# https://pni.princeton.edu/

import os, sys
from setuptools import setup, find_packages
path = os.path.abspath(os.path.dirname(__file__))

## Metadata
DISTNAME = 'sisyphus'
MAINTAINER = 'Sam Zorowitz'
MAINTAINER_EMAIL = 'szorowi1@gmail.com'
DESCRIPTION = 'Code associated with paper'
URL = 'https://pni.princeton.edu/'
LICENSE = 'MIT'
DOWNLOAD_URL = 'http://github.com/szorowi1/sisyphus'

with open(os.path.join(path, 'README.rst'), encoding='utf-8') as readme_file:
    README = readme_file.read()

with open(os.path.join(path, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]
    
VERSION = '0.1'

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=README,
      packages=find_packages(exclude=['docs', 'tests']),
      install_requires=requirements,
      license=LICENSE
)
