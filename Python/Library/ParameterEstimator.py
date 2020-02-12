import sys
import os
import glob
import numpy
import h5py
import cv2
import gc
import getpass
from PIL import Image
from tqdm import tqdm
#from ArgumentParser import ArgumentParser
from matplotlib import pyplot as plt

import scipy
import scipy.ndimage

class ParameterEstimator():
    def __init__(self, transmittance, retardation, mask = None, h5attr = None, Bins = 128, N = 1, GrayWhiteSeperator = None, Debug = False, DebugImage = False):
        self.__NUMBER_OF_BINS = Bins
        self.__N = N
        self.__DEBUG = Debug
        self.__DEBUG_IMAGE = DebugImage

        if self.__DEBUG_IMAGE:
            self.__DEBUG = True

        self.__transmittance = transmittance
        self.__retardation = retardation

        if not mask is None:
            if numpy.count_nonzero(mask) > 0:
                self.__transmittance = numpy.where(mask > 0, self.__transmittance, 1)
                self.__retardation = numpy.where(mask > 0, self.__retardation, 0)

        self.__h5attr = h5attr

        self.__fitgm = None
        self.__fitwm = None
        self.__im = None
        self.__ic = None
        self.__imcor = None
        self.__iccor = None
        self.__fitgmcor = None
        self.__fitwmcor = None

        self.__parameterMap_fitgm = None
        self.__parameterMap_fitwm = None
        self.__parameterMap_fitim = None
        self.__parameterMap_fitic = None

        self.__saturated_gray = None
        self.__saturated_white = None
        self.__gray_substance_mask = None
        self.__white_substance_mask = None
        self.inclination_uncorrected = None
        
        self.__gray_white_seperator = GrayWhiteSeperator

    @property
    def fitgm(self):
        return self.__fitgm

    @fitgm.setter
    def fitgm(self, value):
        self.__fitgm = value
        return self.__fitgm

    @property
    def fitwm(self):
        return self.__fitwm

    @fitwm.setter
    def fitwm(self, value):
        self.__fitwm = value
        return self.__fitwm

    @property
    def im(self):
        return self.__im

    @im.setter
    def im(self, value):
        self.__im = value
        return self.__im

    @property
    def ic(self):
        return self.__ic

    @ic.setter
    def ic(self, value):
        self.__ic = value
        return self.__ic

    @property
    def corrected_fitgm(self):
        return self.__fitgmcor

    @property
    def corrected_fitwm(self):
        return self.__fitwmcor

    @property
    def corrected_im(self):
        return self.__imcor

    @property
    def corrected_ic(self):
        return self.__iccor

    @property
    def saturated_pixels_white(self):
        return self.__saturated_white

    @property
    def saturated_pixels_gray(self):
        return self.__saturated_gray

    def ret_wm(self):
        return numpy.where(self.__white_substance_mask == 1, self.__retardation, 0)
               
    def ret_gm(self):
        return numpy.where(self.__gray_substance_mask == 1, self.__retardation, 0)

    def tra_wm(self):
        return numpy.where(self.__white_substance_mask == 1, self.__corrected_transmittance, 0)

    def tra_gm(self):
        return numpy.where(self.__gray_substance_mask == 1, self.__corrected_transmittance, 0)

    def white_mask(self):
        return self.__white_substance_mask

    def gray_mask(self):
        return self.__gray_substance_mask

    def crossing_mask(self):
        return self.__crossing_substance_mask

    def __getRetardationPlateau(self):
        ret_hist, ret_bins = numpy.histogram(self.__retardation, bins=self.__NUMBER_OF_BINS, range=(0, 1))
        # Cut backgrund
        ret_hist = ret_hist[1:]
        ret_bins = ret_bins[1:]

        if self.__DEBUG_IMAGE:
            plt.plot(ret_bins[:-1], ret_hist)
            plt.show()
        # Get peak of the histogram
        peak = ret_hist.argmax()

        # Search for position in histogram with smallest inside angle after the first peak
        ret_hist = ret_hist.astype('float32')
        ret_hist /= ret_hist.max()

        # Create region of interest around the peak area to determine datapoint
        peakWidth = numpy.argwhere(ret_hist[peak:] < ret_hist[peak]/2).min()
        ret_hist_roi = ret_hist[:min(peak+20*peakWidth, len(ret_hist))]
        ret_bins_roi = ret_bins[:min(peak+20*peakWidth, len(ret_bins))]

        # Array to save the angle between two points
        alpha = numpy.zeros(len(ret_hist_roi) - self.__N + 1)

        # Invalid values might be encountered when calucating the angle between points but they are not relevant for the calcuation
        # Therefore those warnings are ignored here to keep the output clean.
        with numpy.errstate(all='ignore'):
            for i in range(peak, len(ret_hist_roi) - self.__N):
                y1 = ret_hist_roi[i] - ret_hist_roi[i + self.__N]
                y2 = ret_hist_roi[i] - ret_hist_roi[i - self.__N]
                x1 = ret_bins_roi[i] - ret_bins_roi[i + self.__N]
                x2 = ret_bins_roi[i] - ret_bins_roi[i - self.__N]
                alpha[i] = numpy.arccos((y1 * y2 + x1 * x2) / max(1e-15, (numpy.sqrt(x1**2 + y1**2) * numpy.sqrt(x2**2 + y2**2))))
        alpha = numpy.where(numpy.isnan(alpha), 1e10, alpha)
        
        plateau = peak+peakWidth+alpha[peak+peakWidth:-self.__N//2].argmin()

        if self.__DEBUG_IMAGE:
            plt.plot(alpha)
            plt.show()

        if self.__DEBUG:
            print('Plateau is at bin:')
            print(plateau)
            print('Retardation value at plateau is:')
            print(ret_bins[plateau])

        return ret_bins[plateau]

    def __getRegionGrowingMask(self):
        # Generating mask for transmittance_avg_val based on highest connected retardation
        mask_pixels = 0
        difference = 0.01
        while mask_pixels < 512:
            region_growing_mask = numpy.where(self.__retardation > self.__retardation.max() - difference, 1, 0).astype(numpy.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(region_growing_mask)
            maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_pixels = numpy.count_nonzero(labels == maxLabel)
            difference += 0.01

        if self.__DEBUG_IMAGE:
            plt.imshow(labels)
            plt.show()
            region_growing_mask_image = numpy.where(labels == maxLabel, 1, 0)
            plt.imshow(region_growing_mask_image, cmap="gray")
            plt.show()
        
        if self.__DEBUG:
            print("Max retardation is: ", self.__retardation.max())
            print("Difference is:", difference)

        return region_growing_mask.astype(numpy.bool)

    def __getTransmittancePlateau(self):
        # Calculate transmittance values below the maximal peek in histogram
        # Create transmittance histogram
        tra_hist, tra_bins = numpy.histogram(self.__corrected_transmittance, bins=self.__NUMBER_OF_BINS, range=(0, 1))
        if self.__DEBUG_IMAGE:
            plt.plot(tra_bins[:-1], tra_hist)
            plt.show()
        # Get peak of the histogram on the right side
        peak = tra_hist[self.__NUMBER_OF_BINS//2-1:].argmax() + self.__NUMBER_OF_BINS//2-1

        # Search for position in histogram with smallest inside angle before the highest peak
        tra_hist = tra_hist.astype('float32')
        tra_hist /= tra_hist.max()

        # Create region of interest around the peak area to determine datapoint
        peakWidth = peak - numpy.argwhere(tra_hist[:peak] < tra_hist[peak]/2).max()
        tra_hist_roi = tra_hist[max(0, peak-5*peakWidth):peak]
        tra_bins_roi = tra_bins[max(0, peak-5*peakWidth):peak]
        
        alpha = numpy.zeros(len(tra_hist_roi) - self.__N + 1)
        # Invalid values might be encountered when calucating the angle between points but they are not relevant for the calcuation
        # Therefore those warnings are ignored here to keep the output clean.
        with numpy.errstate(all='ignore'):
            for i in range(len(tra_hist_roi) - self.__N):
                y2 = tra_hist_roi[i] - tra_hist_roi[i + self.__N]
                y1 = tra_hist_roi[i] - tra_hist_roi[i - self.__N]
                x2 = tra_bins_roi[i] - tra_bins_roi[i + self.__N]
                x1 = tra_bins_roi[i] - tra_bins_roi[i - self.__N]
                alpha[i] = numpy.arccos((y1 * y2 + x1 * x2) / max(1e-15, (numpy.sqrt(x1**2 + y1**2) * numpy.sqrt(x2**2 + y2**2))))
        alpha = numpy.where(numpy.isnan(alpha), 1e10, alpha)

        min_peak = peak - alpha[self.__N:len(alpha) - self.__N - peakWidth][::-1].argmin() - self.__N - peakWidth
        
        if self.__DEBUG_IMAGE:
            plt.plot(alpha)
            plt.show()

        if self.__DEBUG:
            print('Minimal transmittance bin after maximal peak is at:')
            print(min_peak)
            print('Transmittance value at minimal bin is:')
            print(tra_bins[min_peak])

        return tra_bins[min_peak]

    def createGrayWhiteMask(self):
        # Create retardation histogram
        retardationPlateauEstimation = self.__getRetardationPlateau()

        # Mark all pixels greater than the peak of the histogram 
        mask_wm = numpy.where(self.__retardation > retardationPlateauEstimation, 1, 0)
        mask_gm = numpy.where(self.__retardation <= retardationPlateauEstimation, 1, 0)
        if self.__DEBUG_IMAGE:
            plt.imshow(mask_wm, cmap='gray')
            plt.show()
            plt.imshow(mask_gm, cmap='gray')
            plt.show()

        region_growing_mask = self.__getRegionGrowingMask()
        
        # Calculate average transmittance value in region_growing_mask
        self.__im = numpy.average(self.__transmittance[region_growing_mask])
        if self.__DEBUG:
            print('Average transmittance value for selected region is:')
            print(self.__im)
        
        self.__corrected_transmittance = numpy.where((self.__transmittance < self.__im) & (self.__transmittance > 0), self.__im, self.__transmittance)

        transmittancePlateauEstimation = self.__getTransmittancePlateau()
        
        if self.__gray_white_seperator is None:
            hist, bins = numpy.histogram(self.__corrected_transmittance, bins=self.__NUMBER_OF_BINS, range=(0, 1))
            hist = hist[1:-1]
            bins = bins[1:-1]
            
            im_pos = numpy.argwhere(self.__im < bins).min()
            plateau_pos = numpy.argwhere(transmittancePlateauEstimation < bins).min()
            if plateau_pos != im_pos:
                peak_pos = hist[im_pos:plateau_pos].argmax() + im_pos
                if peak_pos != im_pos:
                    min_pos = hist[im_pos:peak_pos].argmin() + im_pos

                    # Create region of interest around the peak area to determine datapoint
                    hist_roi = hist[min_pos:peak_pos]
                    bins_roi = bins[min_pos:peak_pos]
                    
                    alpha = numpy.zeros(len(hist_roi) - self.__N + 1)
                    if len(alpha) > 2 * self.__N:
                        # Invalid values might be encountered when calucating the angle between points but they are not relevant for the calcuation
                        # Therefore those warnings are ignored here to keep the output clean.
                        with numpy.errstate(all='ignore'):
                            for i in range(len(hist_roi) - self.__N):
                                y2 = hist_roi[i] - hist_roi[i + self.__N]
                                y1 = hist_roi[i] - hist_roi[i - self.__N]
                                x2 = bins_roi[i] - bins_roi[i + self.__N]
                                x1 = bins_roi[i] - bins_roi[i - self.__N]
                                alpha[i] = numpy.arccos((y1 * y2 + x1 * x2) / max(1e-15, (numpy.sqrt(x1**2 + y1**2) * numpy.sqrt(x2**2 + y2**2))))
                        alpha = numpy.where(numpy.isnan(alpha), 1e10, alpha)

                        min_peak = 1 + peak_pos - alpha[self.__N:len(alpha) - self.__N][::-1].argmin() - self.__N
                        self.__gray_white_seperator = max(bins[min_peak], self.__im)
                        if self.__DEBUG:
                            print("Gray/White seperator in transmittance is at {}".format(self.__gray_white_seperator))
                    else:
                        self.__gray_white_seperator = self.__im
                else:
                    self.__gray_white_seperator = self.__im
            else:
                self.__gray_white_seperator = self.__im
            
        transmittance_roi_mask = numpy.where(self.__corrected_transmittance >= self.__gray_white_seperator, numpy.where(self.__corrected_transmittance <= transmittancePlateauEstimation, 1, 0), 0).astype(numpy.bool)
        if self.__DEBUG_IMAGE:
            plt.imshow(transmittance_roi_mask, cmap='gray')
            plt.show()

        # Create gray and white matter masks
        self.__gray_substance_mask = numpy.where((transmittance_roi_mask == 1) & (mask_gm == 1), 1, 0).astype(numpy.bool)
        if self.__DEBUG_IMAGE:
            plt.imshow(self.__gray_substance_mask, cmap='gray')
            plt.show()
        self.__white_substance_mask = numpy.where((transmittance_roi_mask == 0) | (mask_wm == 1), numpy.where((self.__corrected_transmittance < transmittancePlateauEstimation) & (self.__corrected_transmittance > 0), 1, 0), 0).astype(numpy.bool)
        self.__crossing_substance_mask = numpy.where(self.__white_substance_mask, numpy.where((self.__transmittance < self.__im) & ((self.__retardation > 0.05) & (self.__retardation < 0.2)), 1, 0), 0).astype(numpy.bool)
        if self.__DEBUG_IMAGE:
            plt.imshow(self.__white_substance_mask, cmap='gray')
            plt.show()

        del mask_gm
        del mask_wm
        del transmittance_roi_mask
        del region_growing_mask
        gc.collect()

    def generateParameters(self):
        self.createGrayWhiteMask()

        # Calculate masked images and their histograms for final parameter estimation
        full_mask = (self.__white_substance_mask | self.__gray_substance_mask)
        hist, bins = numpy.histogram(self.ret_wm(), bins=512)
        self.__fitwm = bins[numpy.argwhere(hist > numpy.count_nonzero(full_mask) * 1e-5)].max()
        self.__fitgm = numpy.histogram(self.ret_gm(), bins=512)[1][-1]
        """hist, bins = numpy.histogram(self.tra_wm(), bins=512, range=(0, 1))
        plt.imshow(self.tra_wm())
        plt.show()
        plt.plot(bins[:-1], hist)
        plt.show()
        firstIndex = 1
        while hist[firstIndex] > hist[firstIndex+1] and firstIndex < len(hist):
            firstIndex += 1
        firstIndex = firstIndex % len(hist)
        self.__im = bins[hist[firstIndex:400].argmax()+firstIndex]"""
        hist, bins = numpy.histogram(self.tra_gm(), bins=512, range=(0, 0.999))
        self.__ic = bins[numpy.argwhere(hist > numpy.count_nonzero(full_mask) * 1e-5)].max()
        if self.__ic == 0:
            self.__ic = bins[numpy.argwhere(hist > 0)].max()

        self.__parameterMap_fitgm = numpy.full(self.__corrected_transmittance.shape, self.__fitgm)
        self.__parameterMap_fitwm = numpy.full(self.__corrected_transmittance.shape, self.__fitwm)
        self.__parameterMap_fitim = numpy.full(self.__corrected_transmittance.shape, self.__im)
        self.__parameterMap_fitic = numpy.full(self.__corrected_transmittance.shape, self.__ic)

        if self.__DEBUG:
            print('Generated values')
            print('fitwm', self.__fitwm)
            print('fitgm', self.__fitgm)
            print('im', self.__im)
            print('ic', self.__ic)

        if self.__DEBUG_IMAGE:
            plt.imshow(self.ret_wm(), cmap='gray')
            plt.show()
            hist, bins = numpy.histogram(self.ret_wm(), bins=self.__NUMBER_OF_BINS)
            plt.plot(bins[1:-1], hist[1:])
            plt.show()
            plt.imshow(self.ret_gm(), cmap='gray')
            plt.show()
            hist, bins = numpy.histogram(self.ret_gm(), bins=self.__NUMBER_OF_BINS)
            plt.plot(bins[1:-1], hist[1:])
            plt.show()
            plt.imshow(self.tra_wm(), cmap='gray')
            plt.show()
            hist, bins = numpy.histogram(self.tra_wm(), bins=self.__NUMBER_OF_BINS)
            plt.plot(bins[1:-1], hist[1:])
            plt.show()
            plt.imshow(self.tra_gm(), cmap='gray')
            plt.show()
            hist, bins = numpy.histogram(self.tra_gm(), bins=self.__NUMBER_OF_BINS)
            plt.plot(bins[1:-1], hist[1:])
            plt.show()

    def createAndWriteAreaImages(self, input_file, output_folder):
        if self.__im is None or self.__ic is None or self.__fitgm is None or self.__fitwm is None:
            self.generateParameters()

        # Convert transmittance and retardation to RGB image
        transmittance_rgb = numpy.empty((self.__transmittance.shape[0], self.__transmittance.shape[1], 3))
        transmittance_rgb[:, :, 0] = self.__transmittance
        transmittance_rgb[:, :, 1] = self.__transmittance
        transmittance_rgb[:, :, 2] = self.__transmittance

        retardation_rgb = numpy.empty((self.__retardation.shape[0], self.__retardation.shape[1], 3))
        retardation_rgb[:, :, 0] = self.__retardation
        retardation_rgb[:, :, 1] = self.__retardation
        retardation_rgb[:, :, 2] = self.__retardation

        retardation_rgb = numpy.where(retardation_rgb >= self.__fitgm, numpy.where(retardation_rgb < self.__fitwm, [0, 1, 0] * retardation_rgb, retardation_rgb), retardation_rgb)
        transmittance_rgb = numpy.where(transmittance_rgb > self.__ic, transmittance_rgb * [1, 1, 0], numpy.where(transmittance_rgb < self.__im, [0.5, 0, 0] * transmittance_rgb + [0.5, 0, 0], transmittance_rgb))

        # Get slice name of output_folder
        slice_name = input_file[input_file.rfind('/')+1:-3]
        print(slice_name)
        slice_name_ret = slice_name.replace('median10', '').replace('NTransmittance', 'Transmittance').replace('Transmittance', 'Retardation')
        # Create missing directories
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(output_folder+'/histograms'):
            os.makedirs(output_folder+'/histograms')
        if not os.path.exists(output_folder+'/thumbnails'):
            os.makedirs(output_folder+'/thumbnails')

        # Scale images if they are too big
        if self.__transmittance.size > 1e8:
            factor = numpy.sqrt(1e8 / self.__transmittance.size)
            retardation_rgb = cv2.resize(retardation_rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            transmittance_rgb = cv2.resize(transmittance_rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        
        # Convert to uint8 for PIL
        retardation_rgb = (255 * (retardation_rgb - retardation_rgb.min()) / (retardation_rgb.max() - retardation_rgb.min())).astype('uint8')
        transmittance_rgb = (255 * (transmittance_rgb - transmittance_rgb.min()) / (transmittance_rgb.max() - transmittance_rgb.min())).astype('uint8')
        # Swap axes
        retardation_rgb = numpy.transpose(retardation_rgb, (1, 0, 2))
        transmittance_rgb = numpy.transpose(transmittance_rgb, (1, 0, 2))

        Image.fromarray(retardation_rgb).save(output_folder+'/thumbnails/'+slice_name_ret+'.tiff')
        Image.fromarray(transmittance_rgb).save(output_folder+'/thumbnails/'+slice_name+'.tiff')

        del retardation_rgb
        del transmittance_rgb

        # Write histograms with markers
        hist, bins = numpy.histogram(self.__transmittance, bins=self.__NUMBER_OF_BINS, range=(0, 1))
        plt.plot(bins[:-2], hist[:-1])
        plt.axvline(self.__ic, color='k', linestyle='dashed', linewidth=1)
        plt.text(self.__ic, int(hist[:-1].max() * 1.1), 'Ic = ' +f"{self.__ic:.4f}", horizontalalignment='center')
        if self.__iccor and not self.__iccor == self.__ic:
            plt.axvline(self.__iccor, color='red', linestyle='dashed', linewidth=1)
            plt.text(self.__iccor, int(hist[:-1].max() * 1.15), 'Corr Ic = ' +f"{self.__iccor:.4f}", horizontalalignment='center')
        plt.axvline(self.__im, color='k', linestyle='dashed', linewidth=1)
        plt.text(self.__im, int(hist[:-1].max() * 1.1), 'Im = ' +f"{self.__im:.4f}", horizontalalignment='center')
        if self.__imcor and not self.__imcor == self.__im:
            plt.axvline(self.__imcor, color='red', linestyle='dashed', linewidth=1)
            plt.text(self.__imcor, int(hist[:-1].max() * 1.15), 'Corr Im = ' +f"{self.__imcor:.4f}", horizontalalignment='center')
        plt.axvline(self.__gray_white_seperator, color='k', linestyle='dotted', linewidth=1)
        plt.savefig(output_folder+'/histograms/'+slice_name+'.tiff')
        plt.close()

        hist, bins = numpy.histogram(self.__retardation, bins=self.__NUMBER_OF_BINS, range=(0, 1))
        plt.plot(bins[1:-1], hist[1:])
        plt.axvline(self.__fitgm, color='k', linestyle='dashed', linewidth=1)
        plt.text(self.__fitgm, int(hist[1:].max() * 1.1), 'fitgm = ' +f"{self.__fitgm:.4f}", horizontalalignment='center')
        if self.__fitgmcor and not self.__fitgmcor == self.__fitgm:
            plt.axvline(self.__fitgmcor, color='red', linestyle='dashed', linewidth=1)
            plt.text(self.__fitgmcor, int(hist[1:].max() * 1.15), 'Corr fitgm = ' +f"{self.__fitgmcor:.4f}", horizontalalignment='center')
        plt.axvline(self.__fitwm, color='k', linestyle='dashed', linewidth=1)
        plt.text(self.__fitwm, int(hist[1:].max() * 1.1), 'fitwm = ' +f"{self.__fitwm:.4f}", horizontalalignment='center')
        if self.__fitwmcor and not self.__fitwmcor == self.__fitwm:
            plt.axvline(self.__fitwmcor, color='red', linestyle='dashed', linewidth=1)
            plt.text(self.__fitwmcor, int(hist[1:].max() * 1.15), 'Corr fitwm = ' +f"{self.__fitwmcor:.4f}", horizontalalignment='center')
        plt.savefig(output_folder+'/histograms/'+slice_name_ret+'.tiff')
        plt.close()

        gc.collect()

    def __createInclinationNoWeighting(self):
        delta_max = numpy.arcsin(self.__retardation.max())
        inclination = numpy.arccos(numpy.sqrt(numpy.maximum(0, numpy.minimum(1, numpy.arcsin(self.__retardation)/delta_max))))
        
        del delta_max
        gc.collect()
        
        return inclination

    def __createInclination(self):
        # Simple version with possible errors
        # divisor = numpy.log(ic / (im))
        # divident = numpy.log(ic / (transmittance)) * numpy.arcsin(fitwm + (fitgm - fitwm) * (transmittance - im) / (ic - im))
        # delta_max = divident / (divisor)
        # inclination = numpy.arccos(numpy.sqrt(numpy.arcsin(retardation) / (delta_max + 1e-15)))
        delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwm + (self.__fitgm - self.__fitwm) * (self.__corrected_transmittance - self.__im) / numpy.maximum(1e-15, self.__ic - self.__im))))
        inclination = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(self.__retardation) * numpy.log(self.__ic / max(1e-15, self.__im)) / numpy.maximum(1e-15, numpy.log(self.__ic / numpy.maximum(1e-15, self.__corrected_transmittance)) * delta_max))))
        
        del delta_max
        gc.collect()

        return inclination

    def __createInclinationNoFitwmFitgm(self):
        fitgm = self.__fitwm
        delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwm + (fitgm - self.__fitwm) * (self.__corrected_transmittance - self.__im) / numpy.maximum(1e-15, self.__ic - self.__im))))
        inclination = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(self.__retardation) * numpy.log(self.__ic / max(1e-15, self.__im)) / numpy.maximum(1e-15, numpy.log(self.__ic / numpy.maximum(1e-15, self.__corrected_transmittance)) * delta_max))))
        
        del delta_max
        gc.collect()

        return inclination

    def __correctSaturationInpaintAlgorithm(self, inclination, saturation_params=[5e-3, 5e-3, 1e-3]):
        full_mask = (self.__white_substance_mask | self.__gray_substance_mask)
        inc = self.__createInclination()
        saturated_pixels = (((inc <= 0) | (inc >= numpy.pi / 2.0)) & full_mask).astype('uint8')
        self.__corrected_transmittance[saturated_pixels > 0] = 0
        self.__corrected_transmittance = cv2.inpaint(self.__corrected_transmittance, saturated_pixels, 3, cv2.INPAINT_NS)
        
        del full_mask
        del inc
        del saturated_pixels
        gc.collect()
        
        return self.__correctSatuarionGrayWhiteSinglePixel(self.__correctSaturationGrayWhite(self.__createInclination(), saturation_params[:2]), saturation_params[2:])

    def __correctSaturationGrayWhite(self, inclination, saturation_params=[5e-3, 5e-3]):
        inclination_wm = inclination
        inclination_gm = inclination
        self.__fitgmcor = self.__fitgm
        self.__fitwmcor = self.__fitwm
        self.__imcor = self.__im
        self.__iccor = self.__ic

        # Berechne gesättigte Pixel für weiße und graue Substanz
        saturated_pixels_white_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask)).astype('uint8')
        _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_white_mask)
        if len(stats) > 1:
            maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
            saturated_pixels = numpy.count_nonzero(labels == maxLabel)
            desired_saturation = numpy.count_nonzero(self.__white_substance_mask) * saturation_params[0]
            print("White:", saturated_pixels, desired_saturation)
            while saturated_pixels > desired_saturation:
                self.__fitwmcor += 0.01
                if self.__fitwmcor >= 1:
                    self.__imcor += 0.01
                    if self.__imcor >= self.__ic:
                        self.__imcor = self.__ic
                        break
                    self.__fitwmcor = 1
                    
                delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwmcor + (self.__fitgm - self.__fitwmcor) * (self.tra_wm() - self.__imcor) / numpy.maximum(1e-15, self.__ic - self.__imcor))))
                inclination_wm = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(self.ret_wm()) * numpy.log(self.__ic / max(1e-15, self.__imcor)) / numpy.maximum(1e-15, numpy.log(self.__ic / numpy.maximum(1e-15, self.tra_wm())) * delta_max))))
                saturated_pixels_white_mask = (((inclination_wm <= 0) | (inclination_wm >= numpy.pi / 2.0)) & (self.__white_substance_mask)).astype('uint8')
                _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_white_mask)
                maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
                saturated_pixels = numpy.count_nonzero(labels == maxLabel)
                print("White:", saturated_pixels, desired_saturation)

        saturated_pixels_gray_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask)).astype('uint8')
        _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_gray_mask)
        if len(stats) > 1:
            maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
            saturated_pixels = numpy.count_nonzero(labels == maxLabel)
            desired_saturation = numpy.count_nonzero(self.__gray_substance_mask) * saturation_params[1]
            print("Gray:", saturated_pixels, desired_saturation)
            while saturated_pixels > desired_saturation:
                self.__iccor += 0.01 
                if self.__iccor >= 1:
                    self.__fitgmcor += 0.01
                    if self.__fitgmcor >= 0.5 or self.__fitgmcor > self.__fitwm:
                        self.__fitgmcor = numpy.min(0.5, self.__fitwm)
                        break
                    self.__iccor = 1

                delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwm + (self.__fitgmcor - self.__fitwm) * (self.tra_gm() - self.__im) / numpy.maximum(1e-15, self.__iccor - self.__im))))
                inclination_gm = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(self.ret_gm()) * numpy.log(self.__iccor / max(1e-15, self.__im)) / numpy.maximum(1e-15, numpy.log(self.__iccor / numpy.maximum(1e-15, self.tra_gm())) * delta_max))))
                saturated_pixels_gray_mask = (((inclination_gm <= 0) | (inclination_gm >= numpy.pi / 2.0)) & (self.__gray_substance_mask)).astype('uint8')
                _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_gray_mask)
                maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
                saturated_pixels = numpy.count_nonzero(labels == maxLabel)
                print("Gray:", saturated_pixels, desired_saturation)

        self.__parameterMap_fitgm = numpy.where(self.__gray_substance_mask == 1, self.__fitgmcor, self.__parameterMap_fitgm)
        self.__parameterMap_fitwm = numpy.where(self.__white_substance_mask == 1, self.__fitwmcor, self.__parameterMap_fitwm)
        self.__parameterMap_fitim = numpy.where(self.__white_substance_mask == 1, self.__imcor, self.__parameterMap_fitim)
        self.__parameterMap_fitic = numpy.where(self.__gray_substance_mask == 1, self.__iccor, self.__parameterMap_fitic)

        inclination = numpy.where(self.__white_substance_mask == 1, inclination_wm, inclination)
        inclination = numpy.where(self.__gray_substance_mask == 1, inclination_gm, inclination)

        self.__saturated_gray = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask))
        self.__saturated_gray = self.__saturated_gray, self.__saturated_gray / numpy.count_nonzero(self.__gray_substance_mask)
        self.__saturated_white = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask))
        self.__saturated_white = self.__saturated_white, self.__saturated_white / numpy.count_nonzero(self.__white_substance_mask)
        
        del saturated_pixels_gray_mask
        del saturated_pixels_white_mask
        del saturated_pixels
        del labels
        del stats
        del inclination_gm
        del inclination_wm
        gc.collect()

        return inclination

    def __correctSatuarionGrayWhiteSinglePixel(self, inclination, saturation_params=[1e-3]):
        inclination_wm = inclination
        inclination_gm = inclination
        fitgmcor = self.__fitgm
        fitwmcor = self.__fitwm
        imcor = self.__im
        iccor = self.__ic

        # Berechne gesättigte Pixel für weiße und graue Substanz
        saturated_pixels_white_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask)).astype('uint8')
        _, labels = cv2.connectedComponents(saturated_pixels_white_mask)
        saturated_pixels = numpy.count_nonzero(labels > 0)
        desired_saturation = numpy.count_nonzero(saturated_pixels_white_mask) * saturation_params[0]
        while saturated_pixels > desired_saturation:
            saturated_pixels_white_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask)).astype('uint8')
            _, labels = cv2.connectedComponents(saturated_pixels_white_mask)
            saturated_pixels =  numpy.count_nonzero(labels > 0)
            
            fitwmcor += 0.01
            if fitwmcor >= 1:
                fitwmcor = 1
                imcor += 0.01
                if imcor >= self.__ic:
                    imcor = self.__ic
                    break
                fitwmcor = 1
            
            ret_sat = numpy.where(saturated_pixels_white_mask, self.__retardation, 0)
            tra_sat = numpy.where(saturated_pixels_white_mask, self.__transmittance, 0)
            delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, fitwmcor + (self.__fitgm - fitwmcor) * (tra_sat - imcor) / numpy.maximum(1e-15, self.__ic - imcor))))
            inclination_wm = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(ret_sat) * numpy.log(self.__ic / max(1e-15, imcor)) / numpy.maximum(1e-15, numpy.log(self.__ic / numpy.maximum(1e-15, tra_sat)) * delta_max))))
            inclination = numpy.where(saturated_pixels_white_mask, inclination_wm, inclination)
            self.__parameterMap_fitwm = numpy.where(saturated_pixels_white_mask, fitwmcor, self.__parameterMap_fitwm)
            self.__parameterMap_fitim = numpy.where(saturated_pixels_white_mask, imcor, self.__parameterMap_fitim)

        saturated_pixels_gray_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask)).astype('uint8')
        _, labels = cv2.connectedComponents(saturated_pixels_gray_mask)
        saturated_pixels = numpy.count_nonzero(labels > 0)
        desired_saturation = numpy.count_nonzero(saturated_pixels_gray_mask) * saturation_params[0]
        while saturated_pixels > desired_saturation:
            saturated_pixels_gray_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask)).astype('uint8')
            _, labels = cv2.connectedComponents(saturated_pixels_gray_mask)
            saturated_pixels = numpy.count_nonzero(labels > 0)

            iccor += 0.01 
            if iccor >= 1:
                iccor = 1
                fitgmcor += 0.01
                if fitgmcor >= 0.5:
                    fitgmcor = 0.5
                    break
                iccor = 1

            ret_sat = numpy.where(saturated_pixels_gray_mask, self.__retardation, 0)
            tra_sat = numpy.where(saturated_pixels_gray_mask, self.__transmittance, 0)
            delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwm + (fitgmcor - self.__fitwm) * (tra_sat - self.__im) / numpy.maximum(1e-15, iccor - self.__im))))
            inclination_gm = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(ret_sat) * numpy.log(iccor / max(1e-15, self.__im)) / numpy.maximum(1e-15, numpy.log(iccor / numpy.maximum(1e-15, tra_sat)) * delta_max))))
            inclination = numpy.where(saturated_pixels_gray_mask, inclination_gm, inclination)
            self.__parameterMap_fitgm = numpy.where(saturated_pixels_gray_mask, fitgmcor, self.__parameterMap_fitgm)
            self.__parameterMap_fitic = numpy.where(saturated_pixels_gray_mask, iccor, self.__parameterMap_fitic)

        self.__saturated_gray = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask))
        self.__saturated_gray = self.__saturated_gray, self.__saturated_gray / numpy.count_nonzero(self.__gray_substance_mask)
        self.__saturated_white = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask))
        self.__saturated_white = self.__saturated_white, self.__saturated_white / numpy.count_nonzero(self.__white_substance_mask)
        
        del saturated_pixels_gray_mask
        del saturated_pixels_white_mask
        del saturated_pixels
        del labels
        del inclination_gm
        del inclination_wm
        del ret_sat
        del tra_sat
        gc.collect()
        
        return inclination

    def __correctSaturationFull(self, inclination, saturation_params=[5e-3]):
        self.__fitgmcor = self.__fitgm
        self.__fitwmcor = self.__fitwm
        self.__imcor = self.__im
        self.__iccor = self.__ic

        full_mask = (self.__white_substance_mask | self.__gray_substance_mask)

        # Berechne gesättigte Pixel für weiße und graue Substanz
        saturated_pixels_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & full_mask).astype('uint8')
        _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_mask)
        maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
        saturated_pixels = numpy.count_nonzero(labels == maxLabel)
        desired_saturation = numpy.count_nonzero(full_mask) * saturation_params[0]
        print("Saturated Pixels:", saturated_pixels, desired_saturation)
        while saturated_pixels > desired_saturation:
            self.__fitwmcor += 0.01
            if self.__fitwmcor >= 1:
                self.__fitwmcor = 1
                self.__iccor += 0.01
                if self.__iccor >= 1:
                    self.__iccor = 1
                    self.__fitgmcor += 0.01
                    self.__imcor += 0.01
                    if self.__fitgmcor >= self.__fitwmcor or self.__imcor >= self.__iccor:
                        break
                self.__fitwmcor = 1
                
            delta_max = numpy.arcsin(numpy.maximum(0, numpy.minimum(1, self.__fitwmcor + (self.__fitgmcor - self.__fitwmcor) * (self.__corrected_transmittance - self.__imcor) / numpy.maximum(1e-15, self.__iccor - self.__imcor))))
            inclination = numpy.arccos(numpy.minimum(1, numpy.sqrt(numpy.arcsin(self.__retardation) * numpy.log(self.__iccor / max(1e-15, self.__imcor)) / numpy.maximum(1e-15, numpy.log(self.__iccor / numpy.maximum(1e-15, self.__corrected_transmittance)) * delta_max))))
            saturated_pixels_mask = (((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & full_mask).astype('uint8')
            _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_pixels_mask)
            maxLabel = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
            saturated_pixels = numpy.count_nonzero(labels == maxLabel)
            print("Saturated Pixels:", saturated_pixels, desired_saturation)

        self.__parameterMap_fitgm = numpy.full(self.__corrected_transmittance.shape, self.__fitgmcor)
        self.__parameterMap_fitwm = numpy.full(self.__corrected_transmittance.shape, self.__fitwmcor)
        self.__parameterMap_fitim = numpy.full(self.__corrected_transmittance.shape, self.__imcor)
        self.__parameterMap_fitic = numpy.full(self.__corrected_transmittance.shape, self.__iccor)

        self.__saturated_gray = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__gray_substance_mask))
        self.__saturated_gray = self.__saturated_gray, self.__saturated_gray / numpy.count_nonzero(self.__gray_substance_mask)
        self.__saturated_white = numpy.count_nonzero(((inclination <= 0) | (inclination >= numpy.pi / 2.0)) & (self.__white_substance_mask))
        self.__saturated_white = self.__saturated_white, self.__saturated_white / numpy.count_nonzero(self.__white_substance_mask)
        
        del full_mask
        del saturated_pixels
        del delta_max
        del stats
        del labels
        gc.collect()
        
        return inclination

    def __writeInclination(self, input_file, output_folder):
        self.inclination = numpy.rad2deg(self.inclination)
        if not self.inclination_uncorrected is None:
            self.inclination_uncorrected = numpy.rad2deg(self.inclination_uncorrected)

        # Get slice name of output_folder
        slice_name = input_file[input_file.rfind('/')+1:-3]
        slice_name = slice_name.replace('median10', '').replace('NTransmittance', 'Transmittance').replace('Transmittance', 'Inclination')
        # Create missing directories
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(output_folder+'/histograms'):
            os.makedirs(output_folder+'/histograms')
        if not os.path.exists(output_folder+'/thumbnails'):
            os.makedirs(output_folder+'/thumbnails')

        # Write inclination to hdf5 file
        hdf5_file = h5py.File(output_folder+'/'+slice_name.replace('_Inclination', '')+'.h5', mode='w')
        image_dataset = hdf5_file.create_dataset('inclination', self.inclination.shape, numpy.float32, data=self.inclination)
        for param in self.__h5attr:
            image_dataset.attrs[param[0]] = param[1]
        image_dataset.attrs['created_by'] = getpass.getuser()
        image_dataset.attrs['sortware'] = sys.argv[0]
        image_dataset.attrs['software_parameters'] = ' '.join(sys.argv[1:])
        image_dataset.attrs['image_modality'] = 'Inclination'
        image_dataset.attrs['filename'] = slice_name
        image_dataset.attrs['Max_NTm_GM'] = numpy.array(self.__ic, dtype=numpy.float32)
        image_dataset.attrs['Min_NTm_WM'] = numpy.array(self.__im, dtype=numpy.float32)
        image_dataset.attrs['Max_Ret_GM'] = numpy.array(self.__fitgm, dtype=numpy.float32)
        image_dataset.attrs['Max_Ret_WM'] = numpy.array(self.__fitwm, dtype=numpy.float32)
        image_dataset.attrs['Max_NTm_GM_corrected'] = numpy.array(self.__iccor, dtype=numpy.float32)
        image_dataset.attrs['Min_NTm_WM_corrected'] = numpy.array(self.__imcor, dtype=numpy.float32)
        image_dataset.attrs['Max_Ret_GM_corrected'] = numpy.array(self.__fitgmcor, dtype=numpy.float32)
        image_dataset.attrs['Max_Ret_WM_corrected'] = numpy.array(self.__fitwmcor, dtype=numpy.float32)

        # Write uncorrected inclination to hdf5 file
        if not self.inclination_uncorrected is None:
            image_dataset = hdf5_file.create_dataset('inclination_uncorrected', self.inclination_uncorrected.shape, numpy.float32, data=self.inclination_uncorrected)
            for param in self.__h5attr:
                image_dataset.attrs[param[0]] = param[1]
            image_dataset.attrs['created_by'] = getpass.getuser()
            image_dataset.attrs['sortware'] = sys.argv[0]
            image_dataset.attrs['software_parameters'] = ' '.join(sys.argv[1:])
            image_dataset.attrs['image_modality'] = 'Inclination'
            image_dataset.attrs['filename'] = slice_name
            image_dataset.attrs['Max_NTm_GM'] = numpy.array(self.__ic, dtype=numpy.float32)
            image_dataset.attrs['Min_NTm_WM'] = numpy.array(self.__im, dtype=numpy.float32)
            image_dataset.attrs['Max_Ret_GM'] = numpy.array(self.__fitgm, dtype=numpy.float32)
            image_dataset.attrs['Max_Ret_WM'] = numpy.array(self.__fitwm, dtype=numpy.float32)

        image_dataset = hdf5_file.create_dataset('transmittance_uncorrected', self.__transmittance.shape, numpy.float32, data=self.__transmittance)
        for param in self.__h5attr:
            image_dataset.attrs[param[0]] = param[1]
        hdf5_file.flush()

        image_dataset = hdf5_file.create_dataset('transmittance', self.__corrected_transmittance.shape, numpy.float32, data=self.__corrected_transmittance)
        for param in self.__h5attr:
            image_dataset.attrs[param[0]] = param[1]
        hdf5_file.flush()

        if not self.__gray_substance_mask is None:
            hdf5_file.create_dataset('masks/gray_mask', self.__gray_substance_mask.shape, numpy.uint8, data=self.__gray_substance_mask)
            hdf5_file.flush()
        if not self.__white_substance_mask is None:
            hdf5_file.create_dataset('masks/white_mask', self.__white_substance_mask.shape, numpy.uint8, data=self.__white_substance_mask)
            hdf5_file.flush()
        if not self.__parameterMap_fitgm is None:
            hdf5_file.create_dataset('parametermap/fitgm', self.__parameterMap_fitgm.shape, numpy.float32, data=self.__parameterMap_fitgm)
            hdf5_file.flush()
        if not self.__parameterMap_fitwm is None:
            hdf5_file.create_dataset('parametermap/fitwm', self.__parameterMap_fitwm.shape, numpy.float32, data=self.__parameterMap_fitwm)
            hdf5_file.flush()
        if not self.__parameterMap_fitim is None:
            hdf5_file.create_dataset('parametermap/im', self.__parameterMap_fitim.shape, numpy.float32, data=self.__parameterMap_fitim)
            hdf5_file.flush()
        if not self.__parameterMap_fitic is None:
            hdf5_file.create_dataset('parametermap/ic', self.__parameterMap_fitic.shape, numpy.float32, data=self.__parameterMap_fitic)
            hdf5_file.flush()

        # Create histogram of both inclinations in one image
        hist, bins = numpy.histogram(self.inclination, bins=self.__NUMBER_OF_BINS, range=(0, 90))
        plt.plot(bins[1:-2], hist[1:-1], label='Inclination')
        if not self.inclination_uncorrected is None:
            hist, bins = numpy.histogram(self.inclination_uncorrected, bins=self.__NUMBER_OF_BINS, range=(0, 90))
            plt.plot(bins[1:-2], hist[1:-1], label='Inclination uncorrected')
        plt.legend()
        plt.savefig(output_folder+'/histograms/'+slice_name+'.tiff')
        plt.close()

        # Write normal inclination
        self.inclination = numpy.where((self.inclination >= 90) & (self.__white_substance_mask | self.__gray_substance_mask), 200, self.inclination)
        self.inclination = numpy.where((self.inclination == 0) & (self.__white_substance_mask | self.__gray_substance_mask), 250, self.inclination)
        
        inclination_rgb = numpy.empty((self.inclination.shape[0], self.inclination.shape[1], 3))
        inclination_rgb[:, :, 0] = self.inclination
        inclination_rgb[:, :, 1] = self.inclination
        inclination_rgb[:, :, 2] = self.inclination
        inclination_rgb = numpy.transpose(inclination_rgb, (1, 0, 2))
        # Marking low saturation
        inclination_rgb = numpy.where(numpy.isclose(inclination_rgb, 200), [90, 0, 0], inclination_rgb)
        inclination_rgb = numpy.where(numpy.isclose(inclination_rgb, 250), [0, 90, 0], inclination_rgb)

        #if self.__transmittance.size > 1e8:
        #    factor = numpy.sqrt(1e8 / self.__transmittance.size)
        #    retardation_rgb = cv2.resize(retardation_rgb, (int(retardation_rgb.shape[1] * factor), int(retardation_rgb.shape[0] * factor)))

        if self.inclination.size > 1e8:
            factor = numpy.sqrt(1e8 / self.inclination.size)
            inclination_rgb = cv2.resize(inclination_rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        inclination_rgb = (255 * (inclination_rgb - inclination_rgb.min()) / (inclination_rgb.max() - inclination_rgb.min())).astype('uint8')
        Image.fromarray(inclination_rgb).save(output_folder+'/thumbnails/'+slice_name+'.tiff')
        del inclination_rgb

        # Write uncorrected inclination
        if not self.inclination_uncorrected is None:
            self.inclination_uncorrected = numpy.where((self.inclination_uncorrected >= 90) & (self.__white_substance_mask | self.__gray_substance_mask), 200, self.inclination_uncorrected)
            self.inclination_uncorrected = numpy.where((self.inclination_uncorrected == 0) & (self.__white_substance_mask | self.__gray_substance_mask), 250, self.inclination_uncorrected)

            inclination_rgb = numpy.empty((self.inclination_uncorrected.shape[0], self.inclination_uncorrected.shape[1], 3))
            inclination_rgb[:, :, 0] = self.inclination_uncorrected
            inclination_rgb[:, :, 1] = self.inclination_uncorrected
            inclination_rgb[:, :, 2] = self.inclination_uncorrected
            inclination_rgb = numpy.transpose(inclination_rgb, (1, 0, 2))
            # Marking low saturation
            inclination_rgb = numpy.where(numpy.isclose(inclination_rgb, 200), [90, 0, 0], inclination_rgb)
            inclination_rgb = numpy.where(numpy.isclose(inclination_rgb, 250), [0, 90, 0], inclination_rgb)

            if self.inclination_uncorrected.size > 1e8:
                factor = numpy.sqrt(1e8 / self.inclination_uncorrected.size)
                inclination_rgb = cv2.resize(inclination_rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            inclination_rgb = (255 * (inclination_rgb - inclination_rgb.min()) / (inclination_rgb.max() - inclination_rgb.min())).astype('uint8')
            Image.fromarray(inclination_rgb).save(output_folder+'/thumbnails/'+slice_name.replace('Inclination', 'Inclination_uncorrected')+'.tiff')
            del inclination_rgb

        gc.collect()

    def createAndWriteInclinationImage(self, input_file, output_folder, withFitGm = True, withTransmittanceWeighting = True, saturationCorrection = 0, saturation_params = [], zeroCorrection = True):
        if self.__gray_substance_mask is None or self.__white_substance_mask is None:
            self.createGrayWhiteMask()
        
        if not withTransmittanceWeighting:
            self.inclination = self.__createInclinationNoWeighting()
        elif not withFitGm:
            self.inclination = self.__createInclinationNoFitwmFitgm()
        else:
            self.inclination = self.__createInclination()

            if saturationCorrection > 0:
                self.inclination_uncorrected = self.inclination
                if saturationCorrection == 1:
                    self.inclination = self.__correctSaturationGrayWhite(self.inclination_uncorrected, saturation_params)
                elif saturationCorrection == 2:
                    self.inclination = self.__correctSaturationFull(self.inclination_uncorrected, saturation_params)
                elif saturationCorrection == 3:
                    self.inclination = self.__correctSaturationInpaintAlgorithm(self.inclination_uncorrected, saturation_params)
        
        if zeroCorrection:
            self.inclination = numpy.where(self.__corrected_transmittance > self.__ic, numpy.pi / 2.0, self.inclination)
            if not self.inclination_uncorrected is None:
                self.inclination_uncorrected = numpy.where(self.__corrected_transmittance > self.__ic, numpy.pi / 2.0, self.inclination_uncorrected)

        self.__writeInclination(input_file, output_folder)
