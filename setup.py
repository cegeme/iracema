from setuptools import setup, find_packages

setup(
    name='iracema',
    version='0.0.1',
    description='Audio Content Analysis for Research on Musical Expressiveness and Individuality',
    author='Tairone Magalhães <taironemagalhaes@gmail.com>',
    install_requires=[
        'numpy>=1.18.4',
        'scipy>=1.4.1',
        'sounddevice>=0.3.12',
        'audioread>=2.1.8',
        'matplotlib==3.2.1'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries'
    ],
    packages=find_packages())
