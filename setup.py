from distutils.core import setup

setup(
    name='cudanova',
    version='0.1.0',
    author='Jordan Ubbens',
    author_email='jubbens@gmail.com',
    description='GPU accelerated implementation of permutational multivariate analysis of variance (PERMANOVA).',
    packages=['cudanova'],
    license='GPLv3',
    install_requires=[
        'numpy',
        'psutil',
        'tqdm',
        'tensorflow <= 1.9'
    ],
)