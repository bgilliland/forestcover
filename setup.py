# setup.py
from pathlib import Path
from setuptools import find_packages, setup

NAME = 'ml-template'
DESCRIPTION = "Example ML model directory package."
URL = "https://github.com/bgilliland/ml-template"
EMAIL = "bagilliland5328@gmail.com"
AUTHOR = "BlakeGilliland"
REQUIRES_PYTHON = ">=3.9.0"

long_description = DESCRIPTION
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / '_model'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"_model": ["VERSION"]},
    # install_requires=list_reqs(),  # Keep this if needed
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
