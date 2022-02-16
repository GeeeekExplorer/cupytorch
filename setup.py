import subprocess
from setuptools import setup, find_packages

rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.0.0" + rev

setup(name='cupytorch',
      version=version,
      description='cupytorch',
      author='Xingkai Yu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['numpy>=1.22.0'],
      packages=find_packages(),
)