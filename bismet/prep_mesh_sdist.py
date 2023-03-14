from osgeo import gdal
from osgeo import ogr
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage import io
import scipy
from scipy import ndimage
from skimage.segmentation import  *
from skimage.morphology import binary_dilation,binary_erosion,binary_opening,disk
import skfmm
import distmesh as dm
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from schimpy.schism_mesh import SchismMesh,read_mesh,write_mesh

from distmesh._distance_functions import dsegment

#todo: get rid of hardwired assumptions of 1m (e.g. in rect spline)

def prep_mesh_sdist(infile,bnd_shapefile,outdir=".",res=None,
                    adapt_res=False,fixed_points=None,
                    h0=None,cache_dir=None,lab1=0,lab2=0,seed=None):
    """ Create mesh and signed distance required for fdaPDE
    
    This function takes an image of lidar, an image representing 
    desired resolution and boundary shapefile and prepares a mesh.
    There is a plotting option, which produces plots in the opening 3 stages
    as well as the final mesh. These may be helpful to make sure 
    the process is on the right track. Last three args are deprecated.
    
    Parameters
    ----------
    
    infile : str
    Name of tiff file containing image of domain, typically at 1m resolution,
    where no-data value represents water. Ideally the interface between water and 
    land should be smooth as it will be used to identify the directionality of the
    geometry.
    
    res : float or str
    If a string, gives the name of a file describing the desired resolution of mesh. 
    If a float, gives a global resolution. This may be modified with adapt_res.
    if None: will only produce the signed distance and not make the mesh
    
    bnd_shapefile : str
    Name of shapefile containing boundary polygons (or bloack polygons). The edge closest to the domain interior
    will represent the boundary. The polygon should mask all the wetted (no-data) parts just
    outside the boundary for a distance equal to half the channel width. This buffer is required 
    in order for the sdist output not to be affected by boundary effects.
    
    outdir : str
    Directory where outputs will be stored. Default is .
    
    adapt_res : bool
    If True, resolution will be coarsened on interior. This is not parameterized yet,
    so there is only the option to toggle it on. Boundaries will honor res.
 
    h0 : float
    Initial resolution for distmesh. Recommend something a little smaller than 
    the smallest value in res (e.g. 2.8 if res is 3.0).
    
    fixed_points : array
    A csv file with single header line containing an n x 2 array of points
    that must be included. Sometimes used in distmesh 
    for things like corners. Default is None.
    
    seed : int
    To set a seed for the random number generator, default is None
    
    """ 
    
    
    import os
    import gdal
    
    if cache_dir is None: cache_dir = outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(infile):
        raise ValueError("File {} does not exist".format(infile))
    resval = None
    if not res is None:
        try:
            resval=float(res)
        except:    
            if not os.path.exists(res):
                raise ValueError("File {} does not exist".format(res)) 
                
    if not os.path.exists(cache_dir):
        raise ValueError("Cache directory {} does not exist".format(cache_dir))
    
    if seed is None:
        np.random.seed(None)
        print('No random seed selected, output mesh will not be reproducible')
    else:
        np.random.seed(seed)
        print('random seed set to: ',seed)
    
    do_label_1 = True
    do_segment = True
    do_signed_distance = True # Don't change until fixed
    do_mesh = not res is None
    
    # Read the LiDAR image and calculate extent
    file = infile
    ds = gdal.Open(file)            # gdal dataset object
    gt = ds.GetGeoTransform()       # get corrdinate of raster origin + xy resolution 
    band = ds.GetRasterBand(1)      # get raster values
    im = band.ReadAsArray().astype("d") # convert to array
    [rows, cols] = im.shape         # lidar image nrow - ncol
    print(("Image shape: {}".format((im.shape,))))
    bounds = calc_extent(gt,rows,cols)
    xmin,ymin,xmax,ymax = bounds    # pixel coordinate of corners
    print(("Image extent: {}".format((bounds,))))
    imagemax = im.max()
    imagemin = im.min()
    
    # pull and impose no data values
    print(("Image min: {} max: {}".format(imagemin,imagemax)))    
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    print(("No data value for input image: {}".format(ndv)))
    assert ds.GetRasterBand(1).GetNoDataValue() < -1e6
    
    # assign nodata to area under the block polygons
    dsmask = create_masked_raster(ds,bnd_shapefile,"rastermasked.tif")
    bmask = dsmask.GetRasterBand(1) 
    bim = bmask.ReadAsArray().astype("d")    
    assert bmask.GetNoDataValue() < -1e6
    bnanpart = bim < -1e6    
    
    nanpart = im < -1e6 # 10000.
    num_nan = np.count_nonzero(nanpart)
    num_notnan = im.shape[0]*im.shape[1] - num_nan
    print(("Number pixels nan: {} and not nan: {}".format(num_nan,num_notnan)))    
    im[nanpart] = imagemin-0.25

  
    if do_label_1:
        print("label 1")
        
        imt = np.where(bnanpart,1.,0.) # make the nan part (water)= 1, the rest (land) = 0
        labeled, nr_objects = ndimage.label(imt) # assign values to each regions (object)
        print(("Number of objects: {}".format(nr_objects)))

        vizlabeled = np.minimum(labeled,7)
        # save and plot
        np.save(os.path.join(outdir,"cache_labeled_masked.npy"),labeled)
        # plt.figure()
        # plt.imshow(vizlabeled,cmap=plt.get_cmap("Set3"))
        # plt.colorbar()
        # plt.show()
    else:
        labcachefile = os.path.join(cache_dir,"cache_labeled_masked.npy")
        labeled = np.load(labcachefile)
        nr_objects = labeled.max()
    if do_segment:
        print("segment")
        maxndx, maxsz = max_component_size(labeled,nr_objects) # find size of largest region  
        print(("Original image maxndx={} maxsz={}".format(maxndx,maxsz)))    
        labelndx1 = lab1 if lab1 > 0 else maxndx
        # First possibility is that this is part of the largest connected components
        # Second possibility is that it got masked by the boundary
        # polygons
        init = np.isclose(labeled,labelndx1) | ((labeled ==0) & nanpart)
        
        # smooths and disconnects very small openings
        im2 = binary_opening(init,disk(2))
        # buffers the domain by about 4 to 6 pixels
        im2 = binary_dilation(im2,disk(3))
        im2 &= ((labeled == labelndx1) | (labeled == 0) )
        labeled=None   
        
        im4label = np.where(im2,1.,0.)  # changed this (switch)
        labeled2, nr_objects2 = ndimage.label(im4label)
        print("Number of components second pass: {}".format(nr_objects2))
        label_outfile = os.path.join(outdir,"labeled_components.tif")
        write_tiff(label_outfile,ds,labeled2)
        np.save(os.path.join(outdir,"cache_labeled2.npy"),labeled2)       
        vizlabeled2 = np.minimum(labeled2,7)
        # plt.figure()
        # plt.imshow(vizlabeled2,cmap=plt.get_cmap("Set3"))
        # plt.colorbar()
        # plt.show()

    else:
        lab2cachefile = os.path.join(cache_dir,"cache_labeled2.npy")    
        labeled2=np.load(lab2cachefile)  
        nr_objects2 = labeled2.max()        

    maxndx2, maxsz2 = max_component_size(labeled2,nr_objects2)    
    print(("Second round image maxndx={} maxsz={}".format(maxndx2,maxsz2)))
    if do_signed_distance:
        print("signed distance")
        labelndx2=lab2 if lab2 >0 else maxndx2
        domain = np.where(np.isclose(labeled2,labelndx2),-1.,1.)
        domain = ndimage.gaussian_filter(domain,sigma=(3,3),order=0) #-1 for water, 1 for land
        labeled2=None
        sdist = skfmm.distance(domain)  # fast marching method. Distance from boundary, -distance inside the mesh region, +distance outward 
        write_tiff(os.path.join(outdir,"sdist_%s.tif" % outdir),ds,sdist,no_data=None)
        plt.figure()
        plt.imshow(sdist) #,cmap=plt.get_cmap("Set3"))
        plt.colorbar()
        plt.show()
    else: 
        raise NotImplementedError("code path is broken, must calculate sdist")
        sdistcachefile = os.path.join(cache_dir,"cache_sdist.npy")
        sdist = np.load("cache_sdist.npy")
     
        
    if do_mesh:
        print("mesh")
        print(domain.shape[0]) 
        print(ymax - ymin)
        print(domain.shape[0])
        print(xmax - xmin)
         
        # make evently spaced mesh points
        yspace1d = np.linspace(0.5,domain.shape[0]-0.5,domain.shape[0])
        xspace1d = np.linspace(0.5,domain.shape[1]-0.5,domain.shape[1])
        
        # bivariate spline approximation over a rectangular mesh
        fdspline = RectBivariateSpline(xspace1d,yspace1d,sdist[::-1,:].T,kx=1,ky=1)
        paths,pointsets = paths_from_shapefile(bnd_shapefile,xmin,ymin)

        def poly_fd(p,pths=paths,pts=pointsets):
            """Signed distance function for polygons

            """
            from matplotlib import path
            
            dists = [(-1)**pt.contains_points(p) * dsegment(p, pv).min(1)  for pt,pv in zip(pths,pts)]
            d = dists[0]
            for d1 in dists:
                d = np.minimum(d,d1)
            return d
                        
        
        
        def fd(p):
            '''
            make fd the signed function. fd is the signed distance from each node location p to the closest boundary

            Parameters
            ----------
            p : Node positions
    
            '''
            d0 = poly_fd(p)
            d1 = fdspline(p[:,0],p[:,1],grid=False)   
            return np.maximum(-d0, d1)        
            
        if resval is None:
            imres = io.imread(res)
            # no place in domain should be so small,
            # but this is needed to get rid of errors 
            # due to bad background values
            imres = np.maximum(imres,0.1)            
            assert im.shape == imres.shape
        else:
            imres=np.full_like(im,resval)
        if adapt_res:
            imres += np.minimum(1.,np.abs(sdist)/(10.*imres)) * imres

        hdspline = RectBivariateSpline(xspace1d,yspace1d,imres[::-1,:].T,kx=1,ky=1)
        def hd(p):
            return hdspline(p[:,0],p[:,1],grid=False)
        # Make the bounding box           
        bbox=(np.min(xspace1d),np.min(yspace1d),np.max(xspace1d),np.max(yspace1d))
        #bbox=(-500,-500,17000,17000)
        #bbox= (0,0,domain.shape[1],domain.shape[0])
        #bbox = (xmin+0.5,ymin+0.5,xmax-0.5,ymax-0.5)
        print("bbox: %s %s %s %s" % bbox)
        if h0 is None: h0 = imres.min()
        #p,d=dm.distmesh2d(fd, dm.huniform, 4.0, bbox=bbox,fig='gcf')
        base = np.array([xmin,ymin])        
        if not fixed_points is None: # if there are fixed point added. Those will be set as node positions
            fixed_points = fixed_points - base
        #import datetime;
        #print ('before calling distmesh2d:', datetime.datetime.now() )
        
        # Make the mesh
        plt.figure()
        p,d=dm.distmesh2d(fd, hd, h0, bbox=bbox, pfix=fixed_points, fig='gcf', dptol=.06, ttol=.1,
              Fscale=1.2, deltat=.2, geps_multiplier=.001, densityctrlfreq=30)
        #print ('after calling distmesh2d:', datetime.datetime.now() )
        p=p + base
        npt = p.shape[0] # number of points
        ne = d.shape[0] # number of elements
        mesh = SchismMesh()
        mesh.allocate(ne,npt)
        for ip in range(npt):
            xyz = (p[ip,0],p[ip,1],0.)        
            mesh.set_node(ip,xyz)
        for ie in range(ne):
            mesh.set_elem(ie,d[ie,:])
        write_mesh(mesh,os.path.join(outdir,"hgrid.gr3"))
        print('Number of elements = ',ne)
        plt.show()

    band=None
    gt=None
    ds=None      
        
def paths_from_shapefile(fname,xmin=0,ymin=0):
    # Extract first layer of features from shapefile using OGR
    ds = ogr.Open(fname)
    nlay = ds.GetLayerCount()
    lyr = ds.GetLayer(0)
 
    paths = []
    pointsets = []
    lyr.ResetReading()

    # Read all features in layer and store as paths
    for feat in lyr:
        geom = feat.geometry()
        codes = []
        all_x = []
        all_y = []
        for i in range(geom.GetGeometryCount()):
            # Read ring geometry and create path
            r = geom.GetGeometryRef(i)
            x = [r.GetX(j)-xmin for j in range(r.GetPointCount())]
            y = [r.GetY(j)-ymin for j in range(r.GetPointCount())]
            # skip boundary between individual rings
            codes += [mpath.Path.MOVETO] + \
                         (len(x)-1)*[mpath.Path.LINETO]
            all_x += x
            all_y += y
        pointset = np.column_stack((all_x,all_y))
        path = mpath.Path(pointset, codes)
        paths.append(path)
        pointsets.append(pointset)

    return paths,pointsets




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

 
def max_component_size(labels,num_comp):
    max_size = 0
    max_ndx = -1
    size = []
    if labels.dtype == np.dtype('int_'):
        for i in range(1,num_comp+1):
            isize = np.count_nonzero(labels == i)
            size.append(isize)
            if isize > max_size:
                max_ndx=i
                max_size = isize
    elif labels.dtype == np.dtype('float_'):
        for i in range(1,num_comp+1):
            isize = np.isclose(labels,i)
            size.append(isize)
            if isize > max_size:
                max_ndx=i
                max_size = isize
    print("Label range: {} to {}".format(labels.min(),labels.max()))
    print("Background size:")
    print(np.count_nonzero(labels == 0))
    print("Sizes:") 
    print(size[0:3])
    return max_ndx,max_size    

def write_tiff(outfname,ds,arr_out,no_data=None,datatype=None):
    """ other useful datatype would be gdal.GDT_Float64"""
    if datatype is None:
        if arr_out.dtype == np.dtype('int_'):
            datatype=gdal.GDT_UInt32
        elif arr_out.dtype == np.dtype('float_'):
            datatype=gdal.GDT_Float32
        else:
            raise ValueError("Unanticipated data type for write_tiff")
    
    driver = gdal.GetDriverByName("GTiff")
    ny,nx = arr_out.shape
    print("writing tiff")

    outdata = driver.Create(outfname, nx, ny, 1, datatype)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    #nd = -1e-34 if no_data is None else no_data
    #outdata.GetRasterBand(1).SetNoDataValue(nd)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None

def create_masked_raster(src_ds,polyfname,dst_fname):
    '''
    Masked the original lidar raster with the block polygons

    Parameters
    ----------
    src_ds : gdal dataset object
    polyfname : name of shapefiles with the block polygons
    dst_fname : output raster file name

    Returns
    -------
    target_ds : TYPE
        DESCRIPTION.

    '''
    
    print(polyfname)
    source_shape = ogr.Open(polyfname)
    source_layer = source_shape.GetLayer()  # block polygons
    if os.path.exists(dst_fname):
        os.remove(dst_fname)
    xmin,xmax,ymin,ymax = source_layer.GetExtent(0) 
    target_ds = gdal.GetDriverByName("GTiff").CreateCopy(dst_fname,src_ds) # create a copy of the original
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[2.5]) #Burns vector geometries into a raster, not sure why use burn = 2.5.
    target_ds.FlushCache() ##saves to disk!!
    return target_ds
    
def create_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Create mesh and sdist file from an image (LiDAR or otherwise) where no-data values denote water and the interface with land is sufficiently smooth. ")
    parser.add_argument('--infile', dest='infile', default = None, help='image or DEM used to infer locations where there is water. No-data value must be < -1e6.')
    parser.add_argument('--res', dest='resolution', default = None, help='scalar or image giving desired resolution. May be modified under adapt_res option. ')
    parser.add_argument('--bnd_shapefile', type = str,dest='bnd_shapefile', 
                        default = None, help='Polygon shapefile delineating boundaries and masking part of the domain outside the boundary at least as wide as the channel or water body. ')    
    parser.add_argument('--adapt_res',action = 'store_true', default=False, help='resolution given by res is coarsened for some distance into interior.')    
    parser.add_argument('--outdir', dest = 'outdir', default=False,help='directory in which output is generated')
    parser.add_argument('--h0', dest = 'h0', type=float, default=None,help='resolution for initializing distmesh. Recomment a number slightly smaller than smallest res (e.g. 2.8 if smallest is 3.0')
    parser.add_argument('--fixed', dest = 'fixedfile', type=str, default=None,help='csv file with one line header containing x,y fixed points that must be included in mesh.')    
    parser.add_argument('--seed', dest = 'seed', type=int, default=None,help='Set a fix seed for the number generator')        
    return parser
               

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    
    # First run
    infile = args.infile  #"deltalidar_middle_rvr.tif"
    res    = args.resolution
    outdir = args.outdir  # "middle_run4"
    h0     = args.h0      # 2.8
    fixedf = args.fixedfile
    adapt_res = args.adapt_res  # True
    bnd_shapefile = args.bnd_shapefile
    seed = args.seed
  
    
    if fixedf is not None:
        fixed_points = np.loadtxt(fixedf, delimiter=",",skiprows=1)
    else:
        fixed_points = None
        
    prep_mesh_sdist(infile=infile,\
            res=res,
            bnd_shapefile=bnd_shapefile,
            outdir=outdir,
            cache_dir=outdir,
            adapt_res = adapt_res,
            fixed_points = fixed_points,
            h0 = h0,
            seed = seed)    
    
if __name__=="__main__":
    main()    