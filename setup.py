from setuptools import setup, find_packages

setup(
    name='cudanova',
    version='0.1.0',
    author='Jordan Ubbens',
    author_email='jubbens@gmail.com',
    description='GPU accelerated implementation of permutational multivariate analysis of variance (PERMANOVA).',
    packages=find_packages(),
    license='GPLv3',
    install_requires=[
        'numpy',
        'psutil',
        'tqdm',
        'tensorflow >=1.12.1, < 2.0'
    ],
    url="https://github.com/jubbens/cudanova",
    keywords="PERMANOVA, GPU, CUDA"
)
