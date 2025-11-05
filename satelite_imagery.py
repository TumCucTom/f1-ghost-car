import ee
ee.Initialize()
hungaroring = ee.Geometry.Point([19.2520, 47.5823])
image = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(hungaroring).first()
url = image.getThumbURL({'bands': ['B4','B3','B2'], 'min':0, 'max':3000})
print(url)
