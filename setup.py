"""
RAGDA Package Setup - Pure Cython/C Implementation

Build with:
    pip install .

Or for development:
    pip install -e .

Or build extension only:
    python setup.py build_ext --inplace
"""

import os
import sys
import platform
from setuptools import setup, find_packages, Extension
import numpy as np

# Cython is REQUIRED - no fallback
from Cython.Build import cythonize

# Get long description from README
with open("Readme.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Platform-specific compiler flags for maximum optimization
if platform.system() == "Windows":
    extra_compile_args = [
        "/O2",           # Maximum optimization
        "/GL",           # Whole program optimization
        "/fp:fast",      # Fast floating point
    ]
    extra_link_args = ["/LTCG"]  # Link-time code generation
else:
    extra_compile_args = [
        "-O3",                    # Maximum optimization
        "-ffast-math",            # Fast floating point
        "-march=native",          # Optimize for current CPU
        "-funroll-loops",         # Unroll loops
        "-fno-strict-aliasing",   # Allow aliasing optimizations
    ]
    extra_link_args = []

# Cython compiler directives for maximum performance
cython_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "overflowcheck": False,
    "embedsignature": False,
    "binding": False,
    "linetrace": False,
    "profile": False,
}

# Build extensions
ext_modules = cythonize(
    [
        Extension(
            "ragda.core",
            sources=["ragda/core.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ("CYTHON_WITHOUT_ASSERTIONS", "1"),
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        Extension(
            "ragda.highdim_core",
            sources=["ragda/highdim_core.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ("CYTHON_WITHOUT_ASSERTIONS", "1"),
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    compiler_directives=cython_directives,
    annotate=False,  # Set True to generate HTML annotation
)

setup(
    name="ragda",
    version="2.1.0",
    author="RAGDA Team",
    author_email="",
    description="RAGDA - High-Performance Derivative-Free Optimizer (Pure Cython/C)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdventuresInDataScience/RAGDA",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.0",
        "loky>=3.0.0",
    ],
    extras_require={
        "full": [
            "pandas>=1.3.0",
            "scipy>=1.7.0",
            "matplotlib>=3.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="optimization derivative-free hyperparameter-tuning machine-learning cython",
    zip_safe=False,
)
