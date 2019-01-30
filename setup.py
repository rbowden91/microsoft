# TODO: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
from setuptools import setup

setup(name='repair50',
      python_requires='>=3.6',
      version='0.1',
      description='Repair50',
      url='http://github.com/rbowden91/repair50',
      author='Rob Bowden',
      author_email='rbowden91@gmail.com',
      license='GPLv3',
      packages=['src'],
      install_requires=[
        'pycparser',
        'centipyde'
      ],
      zip_safe=False)
