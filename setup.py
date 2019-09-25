import setuptools
from gravipy import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="GraviPy",
    version=__version__,
    description="Tensor Calculus Package for General Relativity",
    long_description=long_description,
    url="",
    author="Wojciech Czaja and contributors",
    author_email="wojciech.czaja@gmail.com",
    maintainer="Wojciech Czaja",
    maintainer_email="wojciech.czaja@gmail.com",
    license="BSD",
    packages=["gravipy"],
    include_package_data=True,
    install_requires=["sympy >= 1.4"],
    platforms="any",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
