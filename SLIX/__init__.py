"""
Scattered Light Imaging Toolbox (SLIX) – an open-source Python package that allows a fully automated evaluation of SLI
measurements and the generation of different parameter maps
"""
__version__ = '2.4.2'
__all__ = ['toolbox', 'io', 'visualization', 'preparation',
           'attributemanager', 'classification', 'parameters']

from . import toolbox, io, visualization, preparation, attributemanager, classification
