"""
DBSI Toolbox - Setup Configuration (v3 - Hybrid Two-Stage Architecture)

Installation:
    pip install -e .

Or:
    python setup.py install
"""

import re
from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _version():
    """Read __version__ from dbsi_toolbox/__init__.py — the ONE source of truth.

    It used to be hardcoded here as well, and in v1.3.5 the two drifted: the
    package source said 1.3.5 while `pip install -e .` registered 1.3.4, so the
    installed distribution metadata reported a version the code did not have.
    (An earlier divergence had left 3.0.0 registered for the same tree.) In an
    editable install this is only mislabelling — `dbsi_toolbox.__version__` is
    read from the source tree, so the notebook version guards still see the real
    version — but a non-editable install would ship the wrong number, and the
    run report's provenance is only worth what the version string is worth.

    Parsed with a regex rather than imported: importing the package at build
    time would require numpy/numba to be installed first.
    """
    src = Path(__file__).parent.joinpath("dbsi_toolbox", "__init__.py").read_text(
        encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', src, re.M)
    if not m:
        raise RuntimeError(
            "cannot find __version__ in dbsi_toolbox/__init__.py — the single "
            "source of truth for the package version has moved or been renamed")
    return m.group(1)


setup(
    name="dbsi-toolbox",
    version=_version(),
    author="DBSI Toolbox Contributors",
    author_email="",
    description="Diffusion Basis Spectrum Imaging (DBSI) - Hybrid Two-Stage Architecture with Numba Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guarnich/pyDBSI_toolbox",
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
        "matplotlib>=3.3.0",
        "ipykernel>=6.0.0",
    ],
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
