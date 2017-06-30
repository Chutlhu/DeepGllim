# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Deep Gllim',
    version='0.0.1',
    description='Deep Gllim package',
    long_description=readme,
    author='Rémi Juge',
    author_email='remi.juge@supelec.fr',
    url='https://github.com/rjuge/',
    license=license,
    install_requires=['numpy','scipy','Theano','keras'],
    packages=find_packages())

