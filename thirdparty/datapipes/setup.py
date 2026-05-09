##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by David Michael Fürst <david_michael.fuerst@dfki.de>, October 2022
##############################################################################
from setuptools import setup, find_packages
from codecs import open
from os import path
from datapipes.versions import api_version

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='datapipes',
    version=api_version,
    description='A library for creating and streaming sharded datasets.',
    license='INTERNAL',
    packages=find_packages(exclude=['examples', 'docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
)
