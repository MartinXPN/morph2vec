from setuptools import find_packages, setup

# Where the magic happens:
setup(
    name='morph2vec',
    version='1.0.0',
    description='Python package for learning vector representations of words with morphemes',
    author='Martin Mirakyan',
    author_email='mirakyanmartin@gmail.com',
    python_requires='>=3.6.0',
    url='https://github.com/MartinXPN/morph2vec',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'tqdm>=4.31.1',
        'fire>=0.1.3',
        'nltk>=3.4',
        'scipy>=1.2.1',
        'numpy>=1.16.1',
        'sentence2tags @ git+https://www.github.com/MartinXPN/sentence2tags@master',
        'word2morph @ git+https://www.github.com/MartinXPN/word2morph@master',
        'fastText @ git+https://www.github.com/MartinXPN/fastText@master',
    ],
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Full list of Trove classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
