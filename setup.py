# TODO: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

from setuptools import setup, find_packages

setup(name='repair50',
    python_requires='>=3.6',
    version='0.2',
    description='Repair50',
    url='http://github.com/rbowden91/repair50',
    author='Rob Bowden',
    author_email='rbowden91@gmail.com',
    license='GPLv3',
    packages=find_packages('src'),
    package_dir={'':'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=[
      'pycparser',
      'centipyde',
      #'tensorflow-rocm',
    ],
    entry_points={
        'console_scripts': [
            'repair50_shim = repair50.scripts.server_shim:main',
            'repair50_preprocess = repair50.scripts.preprocess:main',
            'repair50_train = repair50.scripts.train:main',
            'repair50_multi_shim = repair50.scripts.multi_server_shim:main'
        ]
    },
    zip_safe=False)
