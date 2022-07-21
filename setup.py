import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "balanced_loss", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_DEV_REQUIREMENTS = ["black==21.7b0", "flake8==3.9.2", "isort==5.9.2"]

extras = {"tests": _DEV_REQUIREMENTS, "dev": _DEV_REQUIREMENTS}


setuptools.setup(
    name="balanced-loss",
    version=get_version(),
    author="",
    license="MIT",
    description="Easy to use class-balanced cross-entropy and focal loss implementation for Pytorch.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fcakyon/balanced-loss",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require=extras,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, deep-learning, ml, pytorch, vision, loss, image-classification, video-classification",
)
