import os
import csv
import pydicom as dim
import matplotlib.pyplot as plt

from sympy import Matrix
# ############# ROWS ###########################
# 0  patient ID
# 1  breast_density
# 2  left or right breast
# 3  image view
# 4  abnormality ID
# 5  abnormality type
# 6  mass shape
# 7  mass margins
# 8  assessment
# 9 pathology
# 10 subtlety
# 11 image PATH
# 12 cropped image PATH
# 13 ROI image PATH
# ###############################################


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
        with open(self.mycsv, newline='') as csv_file:
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
        with open(self.mycsv, newline='') as csv_file:
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
        img_path = self.datapath
        roi_path = self.roipath
        img_path = img_path + self.rows[row_number][11]
        roi_path = roi_path + self.rows[row_number][12].replace("\n", "")
        dicom_img = dim.dcmread(img_path)
        dicom_roi = dim.dcmread(roi_path)
        print(dicom_img.pixel_array.__len__())
        print(dicom_roi.pixel_array.__len__())
        return dicom_img.pixel_array, dicom_roi.pixel_array

    def compare_img(self, img_path, crop_path,roi_path):
        dicom_img = dim.dcmread(img_path)
        dicom_crop = dim.dcmread(crop_path)
        dicom_roi = dim.dcmread(roi_path)
        if dicom_img.pixel_array.size == dicom_roi.pixel_array.size:
            return dicom_img, dicom_roi
        elif dicom_img.pixel_array.size == dicom_crop.pixel_array.size:
            return dicom_img, dicom_crop
        else:
            return 0, 0

    def prep_grand_truth_box(self):
        jeblo = 0
        udalo = 0
        row = 0

        first = True
        y_gora = 0
        y_dol = 0
        x_lewo = 0
        x_prawo = 0

        for omage in self.rows:
            img_path = ""
            crop_path = ""
            roi_path = ""
            img_path += self.datapath
            img_path += self.rows[omage][11]
            crop_path += self.roipath
            crop_path += self.rows[omage][12]
            roi_path += self.roipath
            roi_path += self.rows[omage][13].replace("\n", "")
            img, roi = self.compare_img(img_path, crop_path, roi_path)
            if img != 0 and roi != 0:
                gt_img = roi.pixel_array
                y_size, x_size = gt_img.shape
                for y in range(y_size):
                    for x in range(x_size):
                        if gt_img[y][x] == 0:
                            continue
                        else:
                            if first:
                                y_gora = y
                                y_dol = y
                                x_lewo = x
                                x_prawo = x
                                first = False
                            else:
                                if x < x_lewo:
                                    x_lewo = x
                                if y > y_dol:
                                    y_dol = y
                                if x > x_prawo:
                                    x_prawo = x
                pos_x = ((x_prawo + x_lewo)/2)/x_size
                pos_y = ((y_dol + y_gora)/2)/y_size
                height = (y_dol - y_gora)/y_size
                width = (x_prawo - x_lewo)/x_size
                gt_txt = roi_path.replace(".dcm", ".txt")
                f = open(gt_txt, "w")
                f.write(str(pos_x) + ",")
                f.write(str(pos_y) + ",")
                f.write(str(width) + ",")
                f.write(str(height))
                f.close()
                udalo += 1
            else:
                jeblo += 1
            row += 1
            print("Zostało jeszcze " + str(len(self.rows) - row))

        print("udało sie porównać " + udalo.__str__() + " obrazów, jebło: " + jeblo.__str__())
