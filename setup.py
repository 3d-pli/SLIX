from setuptools import setup

setup(
    name='SLIX',
    version='1.1',
    packages=['SLIX'],
    url='www.github.com/3d-pli/slix',
    license='MIT',
    author='Jan André Reuter',
    author_email='j.reuter@fz-juelich.de',
    description='SLIX allows an automated evaluation of SLI measurements and generates different parameter maps.',
    scripts=['bin/SLIXParameterGenerator', 'bin/SLIXLineplotParameterGenerator']
)
