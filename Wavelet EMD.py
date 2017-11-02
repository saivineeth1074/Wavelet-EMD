#Group 6
#Nearest Neighbour search for images using Wavelet EMD

#importing required libraries of opencv
import cv2
import matplotlib.pyplot as plt						#importing library for plotting
import pywt
import numpy as np 
import math
import os

temp2 = []
for filename in os.listdir("dataset"):
	img = cv2.imread("dataset/"+ filename,0)						#reads an input image
	histr = cv2.calcHist([img],[0],None,[256],[0,256])			#frequency of pixels in range 0 to 255
	temp2.append(np.squeeze(histr))

img = cv2.imread('test.jpg',0)						#reads an input image
histr = cv2.calcHist([img],[0],None,[256],[0,256])			#frequency of pixels in range 0 to 255
temp = np.squeeze(histr)

wavelet = pywt.ContinuousWavelet('gaus1')

final = []
for i in temp2:
	coef, freqs=pywt.cwt(temp-i,np.arange(1,129),wavelet)
	sum = 0
	for x in range(1,129):
		for y in range(0,256):
			sum = sum+(math.pow(2,x*-2)*abs(coef[x-1][y]))	#Calculating EMD in wavelet domain
	final.append(sum)
#print final

plt.xlabel('Dataset Image')
plt.ylabel('Wavelet EMD')
plt.title('Nearest Neighbour search for image')
plt.bar(np.arange(len(final)),final,facecolor='g', alpha=0.75)
plt.show()
