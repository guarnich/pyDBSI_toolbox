from setuptools import setup, find_packages

setup(
    name='dbsi_fusion',
    version='1.0.0',
    description='Complete DBSI Toolbox: Fusion of Scientific Rigor and HPC Speed',
    author='DBSI Team',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'numba>=0.56.0',
        'nibabel>=3.2.0',
        'tqdm>=4.60.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0'
    ],
)