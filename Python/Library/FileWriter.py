import numpy
from PIL import Image
import os
import cv2

def writeThumbnail(image, input_file, output_folder, slice_name, histogram = True, withParameters = None):
    # Convert to RGB
    if(len(image.shape) < 3):
        rgb_image = numpy.empty((image.shape[0], image.shape[1], 3))
        rgb_image[:, :, 0] = image
        rgb_image[:, :, 1] = image
        rgb_image[:, :, 2] = image
        rgb_image = numpy.transpose(rgb_image, (1, 0, 2))
    else:
        rgb_image = image

    if image.size > 1e8:
        factor = numpy.sqrt(1e8 / image.size)
        image = cv2.resize(image, (int(image.shape[1] * factor), int(image.shape[0] * factor)), interpolation=cv2.INTER_CUBIC)
    rgb_image = (255 * (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())).astype('uint8')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+'/thumbnails'):
        os.makedirs(output_folder+'/thumbnails')

    Image.fromarray(rgb_image).save(output_folder+'/thumbnails/'+slice_name+'_Inclination.tiff')

    if not os.path.exists(output_folder+'/histograms'):
        os.makedirs(output_folder+'/histograms')

def writeToHDF5File(image, input_file, output_folder):
    pass