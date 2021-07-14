# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:22:11 2021

@author: user
"""

#import the libraries
import pytesseract
import pkg_resources
import cv2

#declaring the exe path for tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
print(pkg_resources.working_set.by_key['pytesseract'].version)

#print the opencv version
print(cv2.__version__)