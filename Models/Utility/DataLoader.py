import pydicom as dim
import matplotlib.pyplot as mlt
import os


def test_dicom_reader():
    file = os.getcwd()
    file += "\\000000.dcm"
    ds = dim.dcmread(file)
    print(ds)
    pixeldata = ds.pixel_array
    mlt.imshow(pixeldata)
