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

setup(name='robo50',
    python_requires='>=3.6',
    version='0.2',
    description='Robo50',
    url='http://github.com/rbowden91/robo50',
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
            'robo50_shim = robo50.scripts.server_shim:main',
            'robo50_preprocess = robo50.scripts.preprocess:main',
            'robo50_train = robo50.scripts.train:main',
            'robo50_multi_shim = robo50.scripts.multi_server_shim:main'
        ]
    },
    zip_safe=False)
