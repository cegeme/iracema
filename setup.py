from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='iracema',
    version='0.1.3',
    url='http://github.com/cegeme/iracema',
    author='Tairone Magalhães',
    author_email='taironemagalhaes@gmail.com',
    description='Audio Content Analysis for Research on Musical Expressiveness and Individuality',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        'numpy>=1.18.4',
        'scipy>=1.4.1',
        'sounddevice>=0.3.12',
        'audioread>=2.1.8',
        'matplotlib==3.2.1',
        'resampy==0.2.2'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
    ],
    packages=find_packages(),
    )
