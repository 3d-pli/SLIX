from SLIX import toolbox
import tifffile
import numpy

if __name__ == "__main__":
    image = toolbox.read_image('/home/jreuter/AktuelleArbeit/90_Stack.tif')
    print(image.shape)

    peaks = toolbox.peaks(image)
    tifffile.imwrite('/home/jreuter/AktuelleArbeit/peak_positions.tiff', numpy.swapaxes(peaks, -1, 0))

    #peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks, kind_of_normalization=1).astype('float32')
    #tifffile.imwrite('/home/jreuter/AktuelleArbeit/prominence.tiff', numpy.swapaxes(peak_prominence_full, -1, 0))

    #peak_prominence = numpy.sum(peak_prominence_full, axis=-1) / numpy.maximum(1, numpy.count_nonzero(peaks, axis=-1))
    #tifffile.imwrite('/home/jreuter/AktuelleArbeit/mean_peak_prominence.tiff', peak_prominence.astype('float32'))

    #del peak_prominence_full
    #del peak_prominence

    peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks).astype('float32')
    peaks[peak_prominence_full < 0.08] = False
    peak_prominence_full[peak_prominence_full < 0.08] = 0
    #tifffile.imwrite('/home/jreuter/AktuelleArbeit/prominence_2.tiff', numpy.swapaxes(peak_prominence_full, -1, 0))
    #tifffile.imwrite('/home/jreuter/AktuelleArbeit/peak_positions_2.tiff', numpy.swapaxes(peaks, -1, 0))

    direction = toolbox.direction(peaks)
    for dim in range(direction.shape[-1]):
        tifffile.imwrite('/home/jreuter/AktuelleArbeit/direction_'+str(dim)+'.tiff', direction[:, :, dim])


