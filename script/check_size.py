from osgeo import gdal
gdal.UseExceptions()
import os, glob

img_dir = '/mnt/c/SECOND_6class_v2/train/image'
files = sorted(glob.glob(os.path.join(img_dir, '*_pre_war.tif')))[:5]
for f in files:
    ds = gdal.Open(f)
    print(f'{os.path.basename(f)}: {ds.RasterXSize} x {ds.RasterYSize}, bands={ds.RasterCount}')
    ds = None
