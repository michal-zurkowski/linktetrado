from setuptools import find_packages, setup

import os

with open('README.md') as f:
    long_description = f.read()

setup(name = "linktetrado",
      version = "1.0.0",
      packages = ['linktetrado'],
      package_dir = {'': 'src'},
      author = "Michal Zurkowski",
      author_email = "michal.zurkowski@cs.put.poznan.pl",
      description = "Detect multimeric motifs.",
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = "https://github.com/michal-zurkowski/linktetrado",
      project_urls = {
          'Bug Tracker': 'https://github.com/michal-zurkowski/linktetrado/issues'
      },
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      entry_points = {'console_scripts': ['linktetrado=linktetrado.main:main']},
      install_requires = [
          'numpy', 'rnapolis', 'orjson', 'eltetrado'
      ]
)
