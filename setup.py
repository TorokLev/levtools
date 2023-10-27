#!/usr/bin/env python
import setuptools

import codecs
import os.path

def get_version_from_init(rel_path):

    def read_by_codec(rel_path):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, rel_path), 'r') as fp:
            return fp.read()

    for line in read_by_codec(rel_path).splitlines():
        if line.startswith('VERSION'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def fread(filename):
    with open(filename, 'r') as f:
        return f.read()


setuptools.setup(
    name="levtools",
    version=get_version_from_init("levtools/__init__.py"),
    author="Levente Torok",
    author_email="toroklev@gmail.com",
    description="",
    long_description=fread('README.md'),
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: " + fread("LICENSE.txt"),
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=fread('requirements.txt').splitlines(),
    include_package_data=True,
    setup_requires=[],
    tests_require=['pytest'],
    package_data={'': ['requirements.txt', "README.md", "LICENSE.txt", "setup.py"]}
)
