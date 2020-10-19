#!/usr/bin/env python
from SLIX import toolbox, visualization
from matplotlib import pyplot as plt
from PIL import Image
import numpy

import urllib.request

# Download example image for visualization
url = 'https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub' \
      '/Human_Brain/optic_tracts_crossing_sections/SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_' \
      'registered.nii'
urllib.request.urlretrieve(url, './SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii')

# Read image and create roi set for toolbox evaluation
image = toolbox.read_image('./SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii')
roiset = toolbox.create_roiset(image, 1)

# Create direction array for unit vector visualization
direction = toolbox.crossing_direction_image(roiset)

# Reconstruct direction map to image
direction_image = direction.reshape((image.shape[0], image.shape[1], 3))

# Create unit vectors
UnitX, UnitY = visualization.unit_vectors(direction_image)

# Show unit vectors with image in one python plot
plt.imshow(numpy.mean(image, axis=-1), cmap='gray')
visualization.visualize_unit_vectors(UnitX, UnitY, thinout=10)
plt.axis('off')
plt.show()

# ---- #
# Example 2

# Create prominence array based on previously read roi set
prominence = toolbox.prominence_image(roiset)
# Reshape resulting array to image for visualization
prominence_image = toolbox.reshape_array_to_image(prominence, image.shape[0], 1)
# Visualize parameter map with base parameters
visualization.visualize_parameter_map(prominence_image[:, :])
plt.axis('off')
plt.show()
# Some areas seem to bright. Let's change the values a bit
visualization.visualize_parameter_map(prominence_image[:, :], vmax=5)
plt.axis('off')
plt.show()



