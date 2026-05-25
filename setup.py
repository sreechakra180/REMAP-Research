"""Setup script for REMAP-Net research package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the long description from README
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="remap-net",
    version="0.1.0",
    author="REMAP-Net Authors",
    description=(
        "REMAP-Net: Recursive Episodic Memory-Augmented Prediction Network "
        "for few-shot continual learning"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/remap-net/remap-net",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "notebooks"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
            "pre-commit>=3.5.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "deep-learning",
        "few-shot-learning",
        "continual-learning",
        "meta-learning",
        "episodic-memory",
        "pytorch",
    ],
    entry_points={
        "console_scripts": [
            "remap-train=remap_net.cli.train:main",
            "remap-eval=remap_net.cli.evaluate:main",
        ],
    },
)
