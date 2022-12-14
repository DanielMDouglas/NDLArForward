#!/usr/bin/env python

import setuptools

VER = "0.0.1"

reqs = ["numpy",
        "larcv",
        "plotly",
        "pyaml",
        "h5py",
        'LarpixParser @ git+https://github.com/YifanC/larpix_readout_parser@develop',
        "torch",
        "MinkowskiEngine"]

setuptools.setup(
    name="NDLArForward",
    version=VER,
    author="Daniel D. and others",
    author_email="dougl215@slac.stanford.edu",
    description="Forward NN for NDLAr sim (testing!)",
    url="https://github.com/YifanC/NDLArForward",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
