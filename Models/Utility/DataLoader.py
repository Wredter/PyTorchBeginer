import pydicom as dim
import matplotlib.pyplot as plt
import numpy as np
import os


def test_dicom_reader():
    file0 = os.getcwd()
    file01 = file0 + "\\000000.dcm"
    file1 = file0 + "\\roi\\000000.dcm"
    file2 = file0 + "\\roi\\000001.dcm"
    ds1 = dim.dcmread(file1)
    ds2 = dim.dcmread(file2)
    ds0 = dim.dcmread(file01)
    print(ds0)
    print("\n")
    print(ds1)
    print("\n")
    print(ds2)
    pixeldata0 = ds0.pixel_array
    pixeldata1 = ds1.pixel_array
    pixeldata2 = ds2.pixel_array
    plt.imshow(pixeldata0)
    plt.show()
    plt.imshow(pixeldata1)
    plt.show()
    plt.imshow(pixeldata2)
    plt.show()
