import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='massspecgym',
    packages=find_packages(include=[
        'massspecgym',
        'massspecgym.utils',
    ]),
    version='0.0.1',  # TODO: Update version automatically
    description='MassSpecGym: Benchmark For the Discovery of New Molecules From Mass Spectra',
    author='MassSpecGym developers',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="",  # TODO: Add URL to documentation
    install_requires=[  # TODO: specify versions, requirements.txt 
        'torch',
        'pytorch-lightning',
        'torchmetrics',
        'numpy',
        'rdkit',
        'matchms',
        'wandb'
    ],
)
