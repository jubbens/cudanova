import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()    
    
setup(
    name='cudanova',
    version='0.1.0',
    author='Jordan Ubbens',
    author_email='jubbens@gmail.com',
    description='GPU accelerated implementation of permutational multivariate analysis of variance (PERMANOVA).',
    packages=find_packages(where="src"),
    license='GPLv3',
    package_dir={"":"src"},
    install_requires=[
        'numpy',
        'psutil',
        'tqdm',
        'tensorflow >=1.12.1, < 2.0'
    ],
    url="https://github.com/jobdiogenes/cudanova",
    long_descripton=read("README.md"),
    keywords="PERMANOVA, GPU, CUDA"
)
