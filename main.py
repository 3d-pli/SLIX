from SLIX import toolbox
import tifffile

if __name__ == "__main__":
    image = toolbox.read_image('/home/jreuter/AktuelleArbeit/90_Stack.tif')
    print(image.shape)
    #num_peaks = toolbox.num_peaks(image)

    tifffile.imwrite('/home/jreuter/AktuelleArbeit/prominence.tiff', toolbox.peak_prominence(image, kind_of_normalization=1).astype('float32'))
