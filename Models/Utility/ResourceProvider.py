import os
import csv

import pydicom as dim

from sympy import Matrix
############## ROWS ###########################
#0  patient ID
#1  breast_density
#2  left or right breast
#3  image view
#4  abnormality ID
#5  abnormality type
#6  mass shape
#7  mass margins
#8  assessment
#9 pathology
#10 subtlety
#11 image PATH
#12 cropped image PATH
#13 ROI image PATH
################################################
class ResourceProvider:

    def __init__(self, mycsv, datapath, roipath):
        self.mycsv = mycsv
        self.datapath = datapath
        self.roipath = roipath
        self.columns = dict()
        self.rows = dict()

    def read(self):
        self.read_csv_column()
        self.read_csv_rows()
        return "Columns and rows initialized"

    def read_csv_column(self):
        with open(self.mycsv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    for column_name in row:
                        self.columns[column_name] = []
                else:
                    line_count +=1
                    row_count = 0
                    for key in self.columns:
                        self.columns[key].append(row[row_count])
                        row_count += 1
        return self.columns

    def read_csv_rows(self):
        with open(self.mycsv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.rows[line_count-1] = row
                    line_count += 1
        return self.rows

    def read_dicom_file(self, row_number):
        file_path = self.datapath
        file_path += self.rows[row_number][11].replace("/", "\\")
        dicom = dim.dcmread(file_path)
        return dicom.pixel_array
