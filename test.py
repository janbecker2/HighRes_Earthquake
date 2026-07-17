import rasterio

INFER_RASTER = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\after_aoi_clip.tif"

with rasterio.open(INFER_RASTER) as src:
    print("CRS:      ", src.crs)
    print("Bounds:   ", src.bounds)
    print("Transform:", src.transform)
    print("Width x H:", src.width, "x", src.height)