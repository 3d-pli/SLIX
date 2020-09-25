from SLIX import toolbox
import tifffile
import numpy

if __name__ == "__main__":
    image = toolbox.read_image('/home/jreuter/AktuelleArbeit/90_Stack.tif')
    print(image.shape)
    #num_peaks = toolbox.num_peaks(image)

    peaks = toolbox.peaks(image)
    peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks, kind_of_normalization=1).astype('float32')
    tifffile.imwrite('/home/jreuter/AktuelleArbeit/prominence_full.tiff', peak_prominence_full)
    peak_prominence = numpy.sum(peak_prominence_full, axis=-1) / numpy.maximum(1, numpy.count_nonzero(peaks, axis=-1))
    tifffile.imwrite('/home/jreuter/AktuelleArbeit/peak_prominence.tiff', peak_prominence)

