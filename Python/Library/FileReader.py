import os
import h5py
import numpy
from PIL import Image

class FileReader():
    def readFromFile(self, file):
        self.__file = file

    @property
    def image(self):
        try:
            if self.__file.endswith('.h5'):
                with h5py.File(self.__file, 'r') as f:
                    # Transmittance
                    try:
                        self.__image = f['Image'][:]
                    except KeyError:
                        self.__image = f['Image_thumbnail_4'][:]
            elif self.__file.endswith('.tif') or self.__file.endswith('.tiff'):
                with Image.open(self.__file) as f:
                    self.__image = numpy.array(f)
            else:
                self.__image = None
            return self.__image
        except OSError:
            return None

    @property
    def attributes(self):
        try:
            with h5py.File(self.__file, 'r') as f:
                # Transmittance
                try:
                    self.__attributes = f['Image'].attrs
                except KeyError:
                    self.__attributes = f['Image_thumbnail_4'].attrs

                finalAttributes = []
                for attr in zip(self.__attributes.keys(), self.__attributes.values()):
                    finalAttributes.append(attr)
            return finalAttributes
        except:
            return None
