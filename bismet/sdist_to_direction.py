# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:23:42 2018

@author: rwang
"""

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.stats import mode
import argparse

def calc_extent(gt,rows,cols):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    xmin = gt[0]
    xmax = xmin  +(cols*gt[1])+(rows*gt[2])
    ymax = gt[3]
    ymin = gt[3] +(cols*gt[4])+(rows*gt[5])
    assert ymin < ymax
    return (xmin,ymin,xmax,ymax)



def write_tiff(outfname,ds,arr_out,no_data=None,datatype=None):
    """ other useful datatype would be gdal.GDT_Float64"""
    if datatype is None:
        if arr_out.dtype == np.dtype('int_'):
            datatype=gdal.GDT_UInt32
        elif arr_out.dtype == np.dtype('float_'):
            print("Writing as Float32")
            datatype=gdal.GDT_Float32
        elif arr_out.dtype == np.dtype('float32'):
            print("Writing as Float32")
            datatype=gdal.GDT_Float32
        else:
            raise ValueError("Unanticipated data type for write_tiff")
    
    driver = gdal.GetDriverByName("GTiff")
    ny,nx = arr_out.shape
    print("writing tiff %s with ny=%s and nx=%s elements" % (outfname,ny,nx))

    outdata = driver.Create(outfname, nx, ny, 1, datatype)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    #nd = -1e-34 if no_data is None else no_data
    #outdata.GetRasterBand(1).SetNoDataValue(nd)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    
def convert_sdist_to_dir(input, output):
    fname = input
    angle_tiff = output
    dataset = gdal.Open(fname, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray() #[4000:8000,4000:8000]
    print("Original array size: %s %s" % array.shape)
    
    #output_folder = 'D:/dem_v4/montezuma_sl/montezuma_west/from_eli/small/'
    #array = -1.0 *array
    print("Starting averaging")
    #array = ndimage.gaussian_filter(array,sigma=(5,5),order=0)
    #array = ndimage.gaussian_filter(array,sigma=(5,5),order=0)
    #array = ndimage.gaussian_filter(array,sigma=(5,5),order=0)
    print("Ending averaging")
    graddist0 = np.gradient(array,axis=0)
    graddist1 = np.gradient(array,axis=1)
    mag = np.hypot(graddist0,graddist1)
    
    print("Min max gradient mag")
    print(np.nanmin(mag))
    print(np.nanmax(mag))
    angle = np.arctan(graddist1/graddist0)/math.pi*180.
    window = 7
    napply = 15
    print(("window={} napply={}".format(window,napply)))
    footprint=np.ones((window,window))
    footprint[window//2,window//2] = 0
    
    usesmooth = np.isnan(mag) | (mag<1.1)
    for i in range(napply):
        #smoothangle = ndimage.median_filter(angle,footprint=footprint)
        #angle=np.where(usesmooth,smoothangle,angle)
        angle = ndimage.median_filter(angle,footprint=footprint)
    print("Angle min/max")
    print(np.nanmin(angle))
    print(np.nanmax(angle))
    
    print("write")
    write_tiff(angle_tiff, dataset, angle)
    print("done")
     
    fig, axes = plt.subplots(1, figsize=(8, 8))
    
    on_image = array < 0
    xx,yy=np.meshgrid(np.linspace(0,array.shape[1],array.shape[1]),np.linspace(0,array.shape[0],array.shape[0]))
    
    ax = [axes] #axes.ravel()
    
    ax[0].imshow(angle, cmap=plt.cm.Spectral,interpolation='nearest')
    ax[0].contour(array,[0.5],colors='w')
    ax[0].axis('off')
    ax[0].set_title('skeleton', fontsize=20)
    
    
    g0 = np.cos(math.pi*angle/180.)
    g1 = np.sin(math.pi*angle/180.)
    
    #ax[0].quiver(xx[on_image][::stride],yy[on_image][::stride],graddist0[on_image][::stride],graddist1[on_image][::stride],scale_units="inches",scale = 2,color = "black")
    
    stride=47
    ax[0].quiver(xx[on_image][::stride],yy[on_image][::stride],g0[on_image][::stride],g1[on_image][::stride],scale_units="inches",scale = 2,color = "white")
    
    fig.tight_layout()
    plt.show()
    
def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert signed distance to direction.")
    parser.add_argument('--input', help="Input sdist file")
    parser.add_argument('--output', help="Output direction file")
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    convert_sdist_to_dir(args.input, args.output)

if __name__ == '__main__':
    main()
