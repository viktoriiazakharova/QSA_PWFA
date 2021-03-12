# Copyright 2021, Viktoriia Zakharova, Igor Andriyash
# Authors: Viktoriia Zakharova, Igor Andriyash
# License: GPL3

import sys
from setuptools import setup, find_packages
import qsa_pwfa

# Obtain the long description from README.md
# If possible, use pypandoc to convert the README from Markdown
# to reStructuredText, as this is the only supported format on PyPI
try:
    import pypandoc
    long_description = pypandoc.convert( './README.md', 'rst')
except (ImportError, RuntimeError):
    long_description = open('./README.md').read()
# Get the package requirements from the requirements.txt file
with open('requirements.txt') as f:
    install_requires = [ line.strip('\n') for line in f.readlines() ]

setup(
    name='QSA_PWFA',
    python_requires='>=3.6',
    version=qsa_pwfa.__version__,
    description='Tool for quasi-static modeling plasma wakefield acceleration',
    long_description=long_description,
    maintainer='Igor Andriyash',
    maintainer_email='igor.andriyash@gmail.com',
    license='GPL3',
    packages=find_packages('.'),
    package_data={"": ['*']},
    tests_require=[],
    cmdclass={},
    install_requires=install_requires,
    include_package_data=True,
    platforms='any',
    url='https://github.com/viktoriiazakharova/QSA_PWFA.git',
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: >=3.6',
        ],
    zip_safe=False,
    )
