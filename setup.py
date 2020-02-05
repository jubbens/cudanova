import os
import sys
from setuptools import setup, find_packages

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
    
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
    
setup(
    name='cudanova',
    version=get_version("src/__init__.py"),
    author='Jordan Ubbens',
    author_email='jubbens@gmail.com',
    description='GPU accelerated implementation of permutational multivariate analysis of variance (PERMANOVA).',
    packages=find_packages(where="src"),
    license='GPLv3',
    package_dir={"":"src"}
    install_requires=[
        'numpy',
        'psutil',
        'tqdm',
        'tensorflow >=1.12.1, < 2.0'
    ],
    url="https://github.com/jobdiogenes/cudanova",
    long_descripton=read("README.md")
    keywords="PERMANOVA, GPU, CUDA",
)
