import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMainWindow, QComboBox
from PyQt5.QtGui import QIcon, QImage, QPixmap, QIntValidator
from PyQt5.uic import loadUi
import cv2
import math
import numpy as np
from multiprocessing import Pool
import struct
import array
import io
import pydicom
from enum import Enum
from pascal_voc_writer import Writer
import csv

class ImageType(Enum):
    DCM = '.dcm'
    RAW = '.raw'
    RAWS = '.raws'
    PNG = '.png'
    JPG = '.jpg'
    BMP = '.bmp'

class MainWindow(QMainWindow):
    toolVersion = '1.5.7'
    releaseDate = '2019/12/10'
    inputImage  = None
    imageData = [] # Storage the data of the raw, raws or DICOM file.
    samplesPerPixel = 1 # DICOM samples per pixel
    imageWidth = 0
    imageHeight = 0
    imageSpacing = 0
    processedImage = None
    isLoadImage = False
    resultImage = []
    inputImageFileName = ''
    saveImageFileName = ''
    saveFileNameList = []
    imageType = None
    dirPath = ''
    dirName = ''
    frameIndex = 0
    numOfFrames = 0
    vocWriter = None
    labelPt1 = (0, 0) # Storage the original label point
    labelPt2 = (0, 0) # Storage the original label point
    processedLabelPt1 = (0, 0) # Storage the processed label point
    processedLabelPt2 = (0, 0) # Storage the processed label point

    # Rotate
    def Rotate(self, image, angle, center=None, scale=1.0):
        #(h, w) = image.shape[:2]
        #if center is None:
        #    (cX, cY) = (w // 2, h // 2)
        #else:
        #    (cX, cY) = center[:2]

        #M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        #cos = np.abs(M[0, 0])
        #sin = np.abs(M[0, 1])
        #nW = int((h * sin) + (w * cos))
        #nH = int((h * cos) + (w * sin))
        #M[0, 2] += (nW / 2) - cX
        #M[1, 2] += (nH / 2) - cY
        #rotated = cv2.warpAffine(image, M, (nW, nH))
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    # Sharpen
    def Sharpen(self, image):
        
        # Laplacian
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    # Contrast
    def Contrast(self, image, alpha, beta):
        # alpha value [1.0-3.0]
        # beta value [0-100]
        result = np.clip(alpha * image + beta, 0, 255)
        return result.astype(image.dtype)
   
    # Gaussian Noise
    def GaussianNoise(self, image, mean, sigma):
        noisy = image.copy()
        cv2.randn(noisy, mean, sigma)
        noisy = image + noisy
        return noisy

    # Fliplr
    def Fliplr(self, image, flipCode):
        flipped = image.copy()
        flipped = cv2.flip(image, flipCode)
        return flipped

    # Salt and Pepper
    def SaltAndPepper(self, image, s_vs_p, amount):
        row,col,ch = image.shape
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[(coords[0], coords[1], 0)] = 255
        out[(coords[0], coords[1], 1)] = 255
        out[(coords[0], coords[1], 2)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[(coords[0], coords[1], 0)] = 0
        out[(coords[0], coords[1], 1)] = 0
        out[(coords[0], coords[1], 2)] = 0
        return out

    # Pad
    def Pad(self, image, offsetX, offsetY):
        rows,cols, ch = image.shape
        pt1 = [0,0]
        pt2 = [cols,0]
        pt3 = [0,rows]
        pts1 = np.float32([pt1,pt2,pt3])
        if offsetX > 0:
            pt1[0] += offsetX
            pt3[0] += offsetX
        elif offsetX < 0 :
            pt2[0] += offsetX
        if offsetY > 0:
            pt1[1] += offsetY
            pt2[1] += offsetY
        elif offsetY < 0:
            pt3[1] += offsetY
        pts2 = np.float32([pt1,pt2,pt3])

        M = cv2.getAffineTransform(pts1,pts2)
        # M = np.float32([[1,0,offsetX],[0,1,offsetY]])
        affined = cv2.warpAffine(image,M,(cols,rows))
        return affined

    def Crop(self, image, offsetX, offsetY):
        rows,cols, ch = image.shape
        crop_img = image[offsetX:(cols-offsetX), offsetY:(rows-offsetY)]
        crop_img = cv2.resize(crop_img,(cols,rows))
        return crop_img

    # Open image file (JPG, PNG)
    def OpenImageFile(self, fileName):
        self.inputImage = cv2.imread(fileName)
        self.imageHeight, self.imageWidth = self.inputImage.shape[:2]
        self.numOfFrames = 1
        self.horizontalSlider_Frame.setMinimum(1)
        self.horizontalSlider_Frame.setMaximum(1)
        return

    # Open DICOM file
    def OpenDICOM(self, fileName):
        ds = pydicom.dcmread(fileName)
        pixelData = ds.PixelData
        self.samplesPerPixel = ds[0x0028, 0x0002].value
        self.imageWidth = ds[0x0028, 0x0011].value
        self.imageHeight = ds[0x0028, 0x0010].value
        if ('NumberOfFrames' in ds) is True:
            if ds[0x0028, 0x0008].value is '':
                self.numOfFrames = 1
            else:
                self.numOfFrames = ds[0x0028, 0x0008].value
        else:
            self.numOfFrames = 1
        self.horizontalSlider_Frame.setMinimum(1)
        self.horizontalSlider_Frame.setMaximum(int(self.numOfFrames))
        for i in range(1, (self.numOfFrames + 1)):
            frameSize = self.imageWidth * self.imageHeight * self.samplesPerPixel
            begin = int(frameSize * (i - 1))
            end = int(frameSize * i) 
            mat = np.frombuffer(pixelData[begin:end], np.uint8)
            if self.samplesPerPixel == 1:
                mat = np.stack((mat,)*3, axis=-1)
            reshapedImage = mat.reshape(self.imageHeight, self.imageWidth, 3)
            self.imageData.append(reshapedImage)
        if len(self.imageData) is not 0:
            self.isLoadImage = True
            self.btn_DICOMtoImage.setEnabled(True)
        self.inputImage = self.imageData[0]
        if self.numOfFrames is 1:
            self.saveImageFileName = self.dirName + '_' +self.inputImageFileName
        else:
            self.saveImageFileName = self.dirName + '_' +self.inputImageFileName + '_frame_1'
        return

    # Open Raw or Raws file
    def OpenRawOrRaws(self, fileName):
        with open(fileName, mode='rb') as file: # b is important -> binary
            binaryData = file.read()
        header = array.array('d', binaryData[0:192])
        self.imageWidth = int(header[16])
        self.imageHeight = int(header[17])
        self.imageSpacing = 40.0 / self.imageWidth
        frameSize = self.imageWidth * self.imageHeight
        picSize = frameSize + 192
        self.numOfFrames = int(len(binaryData) / float(picSize))
        self.horizontalSlider_Frame.setMinimum(1)
        self.horizontalSlider_Frame.setMaximum(self.numOfFrames)
        index = 0
        for i in range(1, (self.numOfFrames + 1)):
            begin = int(picSize * (i - 1))
            end = int(picSize * i) 
            mat = np.frombuffer(binaryData[begin:end], np.uint8)
            end = int(self.imageWidth * self.imageHeight + 192) 
            mat = np.frombuffer(mat[192:end], np.uint8)
            mat = np.stack((mat,)*3, axis=-1)
            if self.imageType == ImageType.RAW:
                mat = self.SolveShiftProblem(0, self.imageWidth, self.imageHeight, mat)
            reshapedImage = mat.reshape(self.imageHeight, self.imageWidth, 3)
            self.imageData.append(reshapedImage)
        if len(self.imageData) is not 0:
            self.isLoadImage = True
        self.inputImage = self.imageData[0]
        if self.numOfFrames is 1:
            self.saveImageFileName = self.dirName + '_' +self.inputImageFileName
        else:
            self.saveImageFileName = self.dirName + '_' +self.inputImageFileName + '_frame_1'
        return

    def LoadFileList(self, dirPath):
        for fname in os.listdir(dirPath):
            strlist = fname.split('.')
            fileType = strlist[len(strlist) - 1]
            if 'jpg' == fileType.lower() or 'png' == fileType.lower() or 'bmp' == fileType.lower() or 'raw' == fileType.lower() or 'raws' == fileType.lower() or 'dcm' == fileType.lower():
                self.comboBox_FileList.addItem(fname)
        return

    def OpenFileByFileName(self, fileName):
        self.btn_DICOMtoImage.setEnabled(False)
        if fileName:
            strlist = fileName.split('/')
            self.inputImageFileName = strlist[len(strlist) - 1]
            self.dirName = strlist[len(strlist) - 2]
            strlist = self.inputImageFileName.split('.')
            self.inputImageFileName = os.path.splitext(self.inputImageFileName)[0]
            self.saveImageFileName =  self.dirName + '_' + self.inputImageFileName
            print(self.inputImageFileName)
            fileType = strlist[len(strlist) - 1]
            if fileType.lower() == 'jpg':
                self.imageType = ImageType.JPG
                self.OpenImageFile(fileName)
            elif fileType.lower() == 'png':
                self.imageType = ImageType.PNG
                self.OpenImageFile(fileName)
            elif fileType.lower() == 'bmp':
                self.imageType = ImageType.BMP
                self.OpenImageFile(fileName)
            elif fileType.lower() == 'raw':
                self.imageType = ImageType.RAW
                self.OpenRawOrRaws(fileName)
            elif fileType.lower() == 'raws':
                self.imageType = ImageType.RAWS
                self.OpenRawOrRaws(fileName)
            elif fileType.lower() == 'dcm':
                self.imageType = ImageType.DCM
                self.OpenDICOM(fileName)
            else:
                self.imageType = None
                print('Can\'t open this type of file.')
            
            if self.inputImage is not None:
                self.btn_ROISave.setEnabled(True)
                self.btn_ROISquare.setEnabled(True)
            self.DisplayImageToLabel()
        return

    # Btn Open Image slot
    def Btn_OpenFile_Clicked(self):
        self.Reset()
        self.dirName = ''
        self.btn_ROISave.setEnabled(False)
        self.btn_ROISquare.setEnabled(False)
        
        self.comboBox_FileList.clear()
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,'Open Image', '','Image File (*)', options=options)
        if self.dirPath is '':
            dir = os.getcwd()
        else:
            dir = self.dirPath
        self.dirPath = QFileDialog.getExistingDirectory(self, 
                                                   'Open directory', 
                                                   dir , 
                                                   QFileDialog.ShowDirsOnly)
        if self.dirPath is not '':
            self.LoadFileList(self.dirPath)
            fileName = self.comboBox_FileList.currentText()
            if fileName is not '':
                self.OpenFileByFileName(self.dirPath + '/' + fileName)
            # cv2.imshow('input', self.inputImage)
            # cv2.waitKey(1)
        return

    # Btn ROI Save slot
    def Btn_ROISave_Clicked(self):
        if len(self.inputImage) is 0:
            print('no image')
        else:
            showCrosshair = False
            fromCenter = False
            cv2.destroyAllWindows()
            temp = self.inputImage.copy()
            if self.imageType is ImageType.RAW or self.imageType is ImageType.RAWS or self.imageType is ImageType.DCM: 
                temp =  cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            r = cv2.selectROI('Select ROI', temp, fromCenter, showCrosshair)
            cv2.waitKey(1)
            if r[3] != 0 and r[2] != 0:
                roi = temp[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                # Save ROI image
                cv2.imwrite(self.dirPath + '/' +  self.saveImageFileName + '_ROI' + '.bmp', roi)
                # Save origin image with ROI rect
                cv2.rectangle(temp,(r[0],r[1]),(r[0]+r[2], r[1]+r[3]),(0,255,0),3)
                cv2.imwrite(self.dirPath + '/' +  self.saveImageFileName + '_OriginWithROI' + '.bmp', temp)
                self.vocWriter = None
                self.vocWriter = Writer(self.dirPath + '/' +  self.inputImageFileName + '.xml', self.imageWidth, self.imageHeight)
                labelType = self.comboBox_LabelType.currentText()
                self.labelPt1 = (r[0], r[1])
                self.labelPt2 = (r[0] + r[2], r[1] + r[3])
                self.vocWriter.addObject(labelType, self.labelPt1[0], self.labelPt1[1], self.labelPt2[0], self.labelPt2[1])
                self.vocWriter.save(self.dirPath + '/' +  self.inputImageFileName + '.xml')
            cv2.destroyAllWindows()
        return

    # Btn ROI Square slot
    def Btn_ROISquare_Clicked(self):
        if len(self.inputImage) is 0:
            print('no image')
        else:
            showCrosshair = False
            fromCenter = False
            cv2.destroyAllWindows()
            temp = self.inputImage.copy()
            if self.imageType is ImageType.RAW or self.imageType is ImageType.RAWS or self.imageType is ImageType.DCM: 
                temp =  cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            r = cv2.selectROI('Select ROI', temp, fromCenter, showCrosshair)
            cv2.waitKey(1)
            rectPt1, rectPt2 = (0,0)
            if r[3] != 0 and r[2] != 0:
                if r[2] > r [3]: # width > height
                    increaseH1 = math.ceil(float(r[2] - r[3]) / 2.0)
                    increaseH2 = math.floor(float(r[2] - r[3]) / 2.0)
                    if (r[1] - increaseH1) < 0:
                        increaseH2 = increaseH2 + (increaseH1 - r[1])
                        roi = temp[0:r[1]+r[3]+increaseH2, r[0]:r[0]+r[2]]
                        rectPt1 = (r[0], 0)
                        rectPt2 = (r[0]+r[2], r[1]+r[3]+increaseH2)
                    elif (r[1] + r[3] + increaseH2) > self.imageHeight:
                        increaseH1 = increaseH1 + (increaseH2 - (self.imageHeight-(r[1]+r[3])))
                        roi = temp[r[1]-increaseH1:self.imageHeight, r[0]:r[0]+r[2]]
                        rectPt1 = (r[0],r[1]-increaseH1)
                        rectPt2 = (r[0]+r[2], self.imageHeight)
                    else:
                        roi = temp[r[1]-increaseH1:r[1]+r[3]+increaseH2, r[0]:r[0]+r[2]]
                        rectPt1 = (r[0],r[1]-increaseH1)
                        rectPt2 = (r[0]+r[2], r[1]+r[3]+increaseH2)
                elif r[3] > r[2]: # height > width
                    increaseW1 = math.ceil(float(r[3] - r[2]) / 2.0)
                    increaseW2 = math.floor(float(r[3] - r[2]) / 2.0)
                    if (r[0] - increaseW1) < 0:
                        increaseW2 = increaseW2 + (increaseW1 - r[0])
                        roi = temp[r[1]:r[1]+r[3], 0:r[0]+r[2]+increaseW2]
                        rectPt1 = (0, r[1])
                        rectPt2 = (r[0]+r[2]+increaseW2, r[1]+r[3])
                    elif (r[0] + r[2] + increaseW2) > self.imageWidth:
                        increaseW1 = increaseW1 + (increaseW2 - (self.imageWidth-(r[0]+r[2])))
                        roi = temp[r[1]:r[1]+r[3], r[0]-increaseW1:self.imageWidth]
                        rectPt1 = (r[0]-increaseW1,r[1])
                        rectPt2 = (self.imageWidth, r[1]+r[3])
                    else:
                        roi = temp[r[1]:r[1]+r[3], r[0]-increaseW1:r[0]+r[2]+increaseW2]
                        rectPt1 = (r[0]-increaseW1,r[1])
                        rectPt2 = (r[0]+r[2]+increaseW2, r[1]+r[3])
                expandRectPt1, expandRectPt2, roiSet = self.ExpandSquare(rectPt1, rectPt2, temp)
                # Save ROI image
                originROI = temp[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                temp2 = temp.copy()
                cv2.rectangle(temp2,(r[0],r[1]),(r[0]+r[2], r[1]+r[3]),(0,255,0),3)
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_OriginWithROI' + '.bmp', temp2)
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_OriginROI' + '.bmp', originROI)
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_ROI' + '.bmp', roiSet[0])
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_CROPROI1' + '.bmp', roiSet[1])
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_CROPROI2' + '.bmp', roiSet[2])
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_CROPROI3' + '.bmp', roiSet[3])
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_CROPROI4' + '.bmp', roiSet[4])
                cv2.rectangle(temp, rectPt1, rectPt2, (0,255,0),3)
                self.vocWriter = None
                self.vocWriter = Writer(self.dirPath + '/' +  self.inputImageFileName + '.xml', self.imageWidth, self.imageHeight)
                self.vocWriter.addObject('ROI', rectPt1[0], rectPt1[1], rectPt2[0], rectPt2[1])
                self.vocWriter.save(self.dirPath + '/' +  self.inputImageFileName + '.xml')
                cv2.rectangle(temp, expandRectPt1, expandRectPt2, (255,0,0),3)
                # Save origin image with ROI rect
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_OriginWithSquareROI' + '.bmp', temp)
            cv2.destroyAllWindows()
        return

    def Btn_DICOMtoImage_Clicked(self):
        index = 1
        if self.imageType is ImageType.DCM:
            for image in self.imageData:
                if self.imageType is ImageType.RAW or self.imageType is ImageType.RAWS or self.imageType is ImageType.DCM: 
                    image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.dirPath + '/' +  self.inputImageFileName + '_' + str(index) + '.bmp', image)
                index += 1
        return

    def ExpandSquare(self,pt1, pt2, temp):
        roiW = abs(pt2[0]-pt1[0])
        roiH = abs(pt2[1]-pt1[1])
        increaseH1 = math.ceil(float(round(roiH * 0.3)) / 2.0)
        increaseH2 = math.floor(float(round(roiH * 0.3)) / 2.0)
        offsetH = increaseH1
        if (pt1[1] - increaseH1) < 0:
            increaseH2 = increaseH2 + (increaseH1 - pt1[1])
            increaseH1 = pt1[1]
        elif (pt2[1] + increaseH2) > self.imageHeight:
            increaseH1 = increaseH1 + (increaseH2 - (self.imageHeight-pt2[1]))
            increaseH2 = self.imageHeight - pt2[1]
        increaseW1 = math.ceil(float(round(roiW * 0.3)) / 2.0)
        increaseW2 = math.floor(float(round(roiW * 0.3)) / 2.0)
        offsetW = increaseW1
        if (pt1[0] - increaseW1) < 0:
            increaseW2 = increaseW2 + (increaseW1 - pt1[0])
            increaseW1 = pt1[0]
        elif (pt2[0] + increaseW2) > self.imageWidth:
            increaseW1 = increaseW1 + (increaseW2 - (self.imageWidth-pt2[0]))
            increaseW2 = self.imageWidth - pt2[0]
        expandROI = temp[pt1[1]-increaseH1:pt2[1]+increaseH2, pt1[0]-increaseW1:pt2[0]+increaseW2]
        rectPt1 = (pt1[0]-increaseW1, pt1[1]-increaseH1)
        rectPt2 = (pt2[0]+increaseW2, pt2[1]+increaseH2)
        cropROI1 = temp[rectPt1[1]:rectPt2[1]-offsetH, rectPt1[0]:rectPt2[0]-offsetW]
        cropROI2 = temp[rectPt1[1]:rectPt2[1]-offsetH, rectPt1[0]+offsetW:rectPt2[0]]
        cropROI3 = temp[rectPt1[1]+offsetH:rectPt2[1], rectPt1[0]:rectPt2[0]-offsetW]
        cropROI4 = temp[rectPt1[1]+offsetH:rectPt2[1], rectPt1[0]+offsetW:rectPt2[0]]
        roiSet = [expandROI, cropROI1, cropROI2, cropROI3, cropROI4]
        return rectPt1, rectPt2, roiSet

    # Btn Save slot
    def Btn_Save_Clicked(self):
        if len(self.resultImage) is 0:
            print('no image')
        else:
            options = QFileDialog.Options()
            # fileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', self.saveFileNameList[0], 'JPG Files (*.jpg)', options=options)
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            print(directory)
            print(self.saveFileNameList[0])
            for index, fileName in enumerate(self.saveFileNameList):
                if directory:
                    self.vocWriter = None
                    print('Saving ' +  fileName + '.jpg...')
                    self.vocWriter = Writer(directory + '/' +  fileName + '.xml', self.imageWidth, self.imageHeight)
                    cv2.imwrite(directory + '/' +  fileName + '.jpg', self.resultImage[index])
                    labelType = self.comboBox_LabelType.currentText()
                    self.vocWriter.addObject(labelType, self.labelPt1[0], self.labelPt1[1], self.labelPt2[0], self.labelPt2[1])
                    self.vocWriter.save(directory + '/' +  fileName + '.xml')
        return

    # Btn Next slot
    def Btn_Next_Clicked(self):
        nextIndex = self.comboBox_FileList.currentIndex() + 1
        if nextIndex <= (self.comboBox_FileList.count() - 1):
            self.comboBox_FileList.setCurrentIndex(nextIndex)
        return

    # Btn Process slot
    def Btn_Process_Clicked(self):
        if self.inputImage is None:
            print('no image')
        else:
            print('Processing......')
            self.resultImage = []
            self.saveFileNameList = []
            self.processedImage = self.inputImage.copy()
            if self.imageType is ImageType.RAW or self.imageType is ImageType.RAWS or self.imageType is ImageType.DCM: 
                self.processedImage =  cv2.cvtColor(self.processedImage, cv2.COLOR_RGB2BGR)
            self.btn_Save.setEnabled(True)
            showCrosshair = False
            fromCenter = False
            cv2.destroyAllWindows()
            r = cv2.selectROI('Select ROI', self.processedImage, fromCenter, showCrosshair)
            cv2.destroyAllWindows()
            center = (int(r[0] + r[2] / 2.0) , int(r[1] + r[3] / 2.0))
            self.labelPt1 = (r[0], r[1])
            self.labelPt2 = (r[0] + r[2], r[1] + r[3])
            self.processedLabelPt1 = self.labelPt1
            self.processedLabelPt2 = self.labelPt2
            # roi = self.processedImage[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
            cv2.waitKey(1)
            if r[3] != 0 and r[2] != 0:
                if self.groupBox_Rotate.isChecked() is True: 
                    degreeBegin = int(self.lineEdit_DegreeBegin.text())
                    degreeEnd = int(self.lineEdit_DegreeEnd.text())
                    step = int(self.comboBox_Step.currentText())
                    d = degreeBegin
                    while d <= degreeEnd:
                        print('rotate_' + str(d))
                        self.saveFileNameList.append(self.saveImageFileName + '_rotate_' + str(d))
                        rotated = self.Rotate(self.processedImage, d, center)
                        #rotated = self.Rotate(self.processedImage, d)
                        #rotatedROI = self.Rotate(roi, d)
                        #(h, w) = rotated.shape[:2]
                        #(cX, cY) = (w // 2, h // 2)
                        #(roiH, roiW) = rotatedROI.shape[:2]
                        #(roiCX, roiCY) = (roiW // 2, roiH // 2)
                        #x = cX - roiCX
                        #y = cY - roiCY
                        #self.resultImage.append(rotated[x:(x+roiW),y:(y+roiH)])
                        self.resultImage.append(rotated)
                        d += step
                else:
                    self.saveFileNameList.append(self.saveImageFileName)
                    self.resultImage.append(self.processedImage)
                for index, image in enumerate(self.resultImage):
                    if self.groupBox_Sharpen.isChecked() is True:
                        print('sharpen')
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_sharpen'
                        self.resultImage[index] = self.Sharpen(self.resultImage[index])
                    if self.groupBox_Contrast.isChecked() is True:
                        print('contrast')
                        alpha = float(self.label_Alpha.text())
                        beta = float(self.label_Alpha.text())
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_contrast_' + str(alpha) + '_' + str(beta) 
                        self.resultImage[index] = self.Contrast(self.resultImage[index], alpha, beta)
                    if self.groupBox_GaussianNoise.isChecked() is True:
                        print('gaussian noise')
                        mean = (int(self.label_MeanB.text()), int(self.label_MeanG.text()), int(self.label_MeanR.text()))
                        sigma = (int(self.label_SigmaB.text()), int(self.label_SigmaG.text()), int(self.label_SigmaR.text()))
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_gaussian_' + str(mean) + '_' + str(sigma) 
                        self.resultImage[index] = self.GaussianNoise(self.resultImage[index], mean, sigma)
                    if self.groupBox_Fliplr.isChecked() is True:
                        print('fliplr')
                        pt1 = list(self.labelPt1)
                        pt2 = list(self.labelPt2)
                        if self.checkBox_Vertical.isChecked() is True and self.checkBox_Horizontal.isChecked() is False:
                            pt1[1] = self.imageHeight - self.labelPt2[1]
                            pt2[1] = self.imageHeight - self.labelPt1[1]
                            self.resultImage[index] = self.Fliplr(self.resultImage[index], 0)
                            self.saveFileNameList[index] = self.saveFileNameList[index] + '_fliprV' 
                        # Horizontal
                        elif self.checkBox_Horizontal.isChecked() is True and self.checkBox_Vertical.isChecked() is False:
                            pt1[0] = self.imageWidth - self.labelPt2[0]
                            pt2[0] = self.imageWidth- self.labelPt1[0]
                            self.resultImage[index] = self.Fliplr(self.resultImage[index], 1)
                            self.saveFileNameList[index] = self.saveFileNameList[index] + '_fliprH'
                        # Vertical and Horizontal
                        elif self.checkBox_Horizontal.isChecked() is True and self.checkBox_Vertical.isChecked() is True:
                            pt1[0] = self.imageWidth - self.labelPt2[0]
                            pt2[0] = self.imageWidth - self.labelPt1[0]
                            pt1[1] = self.imageHeight - self.labelPt2[1]
                            pt2[1] = self.imageHeight - self.labelPt1[1]
                            self.resultImage[index] = self.Fliplr(self.resultImage[index], -1)
                            self.saveFileNameList[index] = self.saveFileNameList[index] + '_fliprVH' 
                        self.processedLabelPt1 = tuple(pt1)
                        self.processedLabelPt2 = tuple(pt2)
                        # cv2.rectangle(self.resultImage[index], self.processedLabelPt1, self.processedLabelPt2, (0, 255, 0), 2)
                    if self.groupBox_SaltAndPepper.isChecked() is True:
                        print('salt and pepper')
                        s_vs_p = float(self.horizontalSlider_SaltAndPepper.value()) / 10.0
                        amount = float(self.label_Amount.text()) / 1000.0
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_sap_' + str(s_vs_p) + '_' + str(amount)
                        self.resultImage[index] = self.SaltAndPepper(self.resultImage[index], s_vs_p, amount)
                    if self.groupBox_Pad.isChecked() is True:
                        print('pad')
                        offsetX = int(int(self.label_PadOffset.text()) / 100 * self.imageWidth)
                        offsetY = int(int(self.label_PadOffset.text()) / 100 * self.imageHeight)
                        if self.checkBox_PadRight.isChecked() is True:
                            offsetX = offsetX
                        elif self.checkBox_PadLeft.isChecked():
                            offsetX = -offsetX
                        else:
                            offsetX = 0
                        if self.checkBox_PadBottom.isChecked() is True:
                            offsetY = offsetY
                        elif self.checkBox_PadTop.isChecked():
                            offsetY = -offsetY
                        else:
                            offsetY = 0
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_pad_' + str(offsetX) + '_' + str(offsetY)
                        self.resultImage[index] = self.Pad(self.resultImage[index], offsetX, offsetY)
                    if self.groupBox_Crop.isChecked() is True:
                        print('crop')
                        ratio = int(self.label_CropRatio.text())
                        self.saveFileNameList[index] = self.saveFileNameList[index] + '_crop_' + str(ratio)
                        ptlist = [self.processedLabelPt1[0], self.processedLabelPt1[1], self.imageWidth - self.processedLabelPt2[0], self.imageHeight - self.processedLabelPt2[1]]
                        offsetX = int(abs(self.processedLabelPt2[0] - self.processedLabelPt1[0]) * (ratio / 100.0))
                        offsetY = int(abs(self.processedLabelPt2[1] - self.processedLabelPt1[1]) * (ratio / 100.0))
                        newROIPt1X = round(((self.processedLabelPt1[0] - offsetX) / (self.imageWidth - 2 * offsetX)) * self.imageWidth)
                        if newROIPt1X < 0:
                            newROIPt1X = 0
                        newROIPt1Y = round(((self.processedLabelPt1[1] - offsetY) / (self.imageHeight - 2 * offsetY)) * self.imageHeight)
                        if newROIPt1Y < 0:
                            newROIPt1Y = 0
                        newROIPt2X = round(((self.processedLabelPt2[0] - offsetX) / (self.imageWidth - 2 * offsetX)) * self.imageWidth)
                        if newROIPt2X > self.imageWidth:
                            newROIPt2X = self.imageWidth
                        newROIPt2Y = round(((self.processedLabelPt2[1] - offsetY) / (self.imageHeight - 2 * offsetY)) * self.imageHeight)
                        if newROIPt2Y > self.imageHeight:
                            newROIPt2Y = self.imageHeight
                        self.processedLabelPt1 = (newROIPt1X, newROIPt1Y)
                        self.processedLabelPt2 = (newROIPt2X, newROIPt2Y)
                        self.resultImage[index] = self.Crop(self.resultImage[index], offsetX, offsetY)
                        # cv2.rectangle(self.resultImage[index], self.processedLabelPt1, self.processedLabelPt2, (0, 255, 0), 2)
                    cv2.imshow("result_" + str(index), self.resultImage[index])
                    cv2.waitKey(1)
         
    def Reset(self):
        self.resultImage = []
        self.saveFileNameList = []
        self.isLoadImage = False
        self.imageData = []
        self.inputImage = None
        self.saveImageFileName = ''
        self.horizontalSlider_Frame.setValue(0)
        cv2.destroyAllWindows()
        processedImage = None
        self.label_ImageDisplay.clear()
        self.btn_Save.setEnabled(False)
        return

    # Solve shift problem when file is raw
    def SolveShiftProblem(self, frameIndex, width, height, temp):
        rgbStride = (((width * 24) + 31) & ~31) >> 3
        rgbOffset = rgbStride - width * 3
        bmpIndex = 0
        memIndex = 0
        imageSize = width * height
        shiftedImg = temp.copy()
        shiftedImg.setflags(write=1)
        for h in range(height):
            for w in range(width):
                if memIndex < (height * width):
                    shiftedImg[frameIndex*imageSize + bmpIndex] = temp[memIndex]
                else:
                    shiftedImg[frameIndex*imageSize + bmpIndex] = 0
                bmpIndex += 1
                memIndex += 1
            if rgbOffset == 1:
                if h % 3 == 1:
                    memIndex += 1
            elif rgbOffset == 2:
                if h % 3 == 0 or h % 3 == 2:
                    memIndex += 1
            elif rgbOffset == 3:
                memIndex += 1
        return shiftedImg

    def DisplayImageToLabel(self):
        cv_img_rgb = self.inputImage
        if self.imageType is ImageType.JPG or self.imageType is ImageType.PNG or self.imageType is ImageType.BMP:
            cv_img_rgb = cv2.cvtColor(cv_img_rgb, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img_rgb.shape
        bytesPerLine = channel * width

        q_image = QImage(cv_img_rgb[:], cv_img_rgb.shape[1], cv_img_rgb.shape[0], cv_img_rgb.shape[1] * 3,
                                    QImage.Format_RGB888)
                
        pixmap01 = QPixmap.fromImage(q_image)
        pixmap_image = QPixmap(pixmap01)
        self.label_ImageDisplay.setPixmap(pixmap_image)
        return

    # Slider Contrast Alpha slot
    def Slider_ContrastAlpha_ValueChanged(self, value):
        self.label_Alpha.setText(str(value / 10.0))
        return

    # Slider Contrast Beta slot
    def Slider_ContrastBeta_ValueChanged(self, value):
        self.label_Beta.setText(str(value))
        return

    # Slider Frame slot
    def Slider_Frame_ValueChanged(self, value):
        if value is not 0 and self.isLoadImage is True:
            if len(self.imageData) is not 0:
                self.saveImageFileName = self.dirName + '_' + self.inputImageFileName + '_frame_'+ str(value)
                self.inputImage = self.imageData[value - 1]
                self.DisplayImageToLabel()
                # cv2.imshow('input', self.inputImage)
                # cv2.waitKey(1)
        return

    # CheckBox Pad Top slot
    def CheckBox_PadTop_Toggled(self, toggled):
        if toggled is True:
            self.checkBox_PadBottom.setChecked(False)
        return

    # CheckBox Pad Bottom slot
    def CheckBox_PadBottom_Toggled(self, toggled):
        if toggled is True:
            self.checkBox_PadTop.setChecked(False)
        return

    # CheckBox Pad Left slot
    def CheckBox_PadLeft_Toggled(self, toggled):
        if toggled is True:
            self.checkBox_PadRight.setChecked(False)
        return

    # CheckBox Pad Right slot
    def CheckBox_PadRight_Toggled(self, toggled):
        if toggled is True:
            self.checkBox_PadLeft.setChecked(False)
        return

    def ComboBox_FileList_CurrentIndexChanged(self, fileName):
        if fileName is not '' and fileName is not None:
            self.Reset()
            self.OpenFileByFileName(self.dirPath + '/' + fileName)
        return

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('mainwindow.ui', self)
        # self.setFixedSize(self.sizeHint())
        print('Tool Version: ' + self.toolVersion)
        print('Release Date: ' + self.releaseDate)
        # with open("D:/Source/NewTFSWorkspace/BR-FHUS Smart System V1.1/TestRawData/R_20150924_160118_Righttemp.raws", mode='rb') as file: # b is important -> binary
        onlyInt = QIntValidator(0, 355)
        self.lineEdit_DegreeBegin.setValidator(onlyInt)
        self.lineEdit_DegreeEnd.setValidator(onlyInt)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())