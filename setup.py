"""
DBSI Toolbox - Setup Configuration

Installation:
    pip install -e .
    
Or:
    python setup.py install
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dbsi-toolbox",
    version="2.0.0",
    author="DBSI Toolbox Contributors",
    author_email="",
    description="Diffusion Basis Spectrum Imaging (DBSI) - Two-Step Implementation with Numba Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/dbsi-toolbox",
    packages=find_packages(),
    py_modules=["model"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.55.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "dbsi-fit=scripts.run_dbsi:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "diffusion MRI",
        "DBSI",
        "neuroimaging",
        "white matter",
        "multiple sclerosis",
        "inflammation",
        "demyelination",
    ],
)