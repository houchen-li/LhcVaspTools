#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='LhcVaspTools',
      version='1.0',
      description='The VASP Tools to Handle \"vaspout.h5\"',
      author='Houchen Li',
      author_email='houchen_li@hotmail.com',
      # url='https://',
      package_dir={
          'LhcVaspTools': 'src',
      },
      packages=['LhcVaspTools'],
      scripts=[
          'bin/extractEFermi.py',
          'bin/deterCrossBands.py',
          'bin/genBands.py',
          'bin/plotBands.py',
          'bin/genEnergyCutCrsSec.py',
          'bin/plotEnergyCutCrsSec.py',
          'bin/genElecDnstyCrsSec.py',
          'bin/plotElecDnstyCrsSec.py',
          'bin/genBandsWithOam.py',
          'bin/plotBandsWithOam.py',
          'bin/genElecDnstyCrsSecWithOam.py',
          'bin/plotElecDnstyCrsSecWithOam.py',
      ],
      python_requires='>=3.9',
      install_requires=[
          'numpy',
          'matplotlib',
          'h5py',
      ],
      )
