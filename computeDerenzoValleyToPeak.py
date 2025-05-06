#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Goal of this module:
	Use the method suggested in [1] to extract the valley-to-peak ratio of the sectors
	of a Hot Spots phantom. Currently, the user provide a json file that give a 
	description of the sector to extract from 2D image(s). If they are in 3D, the user
	is expected to provide the range of z-axis to sum.

Basic idea of the approach:
	Line profiles are drawn across the spots. They are drawn at three directions along 
	the spots to capture possible anisotropies in the spot distributions. From these 
	profiles, valley-to-peak ratio (VPR) values can be extracted. The user can decide 
	how to compute the VPR, whether by using the maximum of the peak and the minimum of 
	the valley or by doing an average around the peak and the valley to limit the 
	influence of possible fluctuations/noise.
	
Results that can be obtained with this module:
	The average VPR and standard deviation of each sector of the phantom is reported in 
	order to be compared with the Rayleigh criterion, i.e. if the VPR is below 0.735, 
	spots can be considered as "resolved". An histogram of the VPR values can be 
	plotted. The resolvability (% of spots below the Rayleigh criterion) was added to 
	complement the average VPR. Example: The 0.75 mm section has an average VPR of 
	0.710 and 70% resolvability. There is still no consensus in the community on how 
	to say whether a certain spot dimension is resolved, so the user has to decide 
	for now how to claim it; the average VPR is one choice, 90% resolvability is 
	another, etc. Beware of the resolvability % which uses line profiles than can pass
	multiple times across points.

Current state:
	Right now, we have a first version that will surely break easily. The results are 
	simply shown in the terminal and some debugging options are available. 
	
Limitations:
	- Depend entirely on the user to give the correct triangle vertex.
	- Assume that the configuration file and images are using the same units.
	- Assume that the defined triangle is filled with spots. 
	- Ignore spots outside the defined triangle.

Questions:
	- Do we need to support image offset and orientation?

TODO: (See also TODO in the code)
	Core:
		- Make sure that orientation, image spacing and image size are correctly dealt 
		  with.

	Feature:
		- Incorporate spots beyond those in a triangle.
		- Suggest a correction for the triangle vertex position.
		- Add an option to show a profile of all the spots in a line + the position of
		  the peaks/valley computed.
		
[1] Hallen, P., Schug, D. & Schulz, V. Comments on the NEMA NU 4-2008 Standard on
	Performance Measurement of Small Animal Positron Emission Tomographs. EJNMMI Phys 7, 
	12 (2020). https://doi.org/10.1186/s40658-020-0279-2
'''



#########################################################################################
# Imports:
#########################################################################################
# Basic python module
import sys
import numpy as np
import json
import argparse
import warnings
import os
# To read dicom
import pydicom
# Csv creation
import pandas as pd
# # Needed for rotation (Not used right now)
# from scipy import ndimage 
# from scipy.spatial.transform import Rotation
# For results production
import matplotlib.pylab as plt 
from matplotlib.patches import Rectangle
import nibabel as nib



#########################################################################################
# Constants
#########################################################################################
RAYLEIGH_CRITERION = 0.735

# # Warning constant
# The parabola metric was built by assuming that most of the ROI will be used. As such
# the code warn the user if the values used are lower than the following.
roiRatioForParabola = 0.95

# Global variable
bShowAxialView = False 


#########################################################################################
# Loading image(s):
#########################################################################################
def loadImages(_imPath, _zIndex, _binFormat, _nImSpacing):
	'''
	Def: Load the images stored in _imPath following the information provided by the
	     other parameters.
	@_imPath (List, Str): Path where the image(s) are saved.
	@_zIndex (2 Int): For 3D images, the z index to sum. 
	@_binFormat: Information required to extract an image from a binary file.  
	@_nImSpacing (2 Float): Physical distance between pixels in the (x, y) axes. 
	Return:
		listIm (List, 2D numpy array)
		imSpacing (List, 2 floats)
	TODO: Refactor to make more sens with image format...
	'''	
	# When path provided are folder, we assume dicom
	if len([cP for cP in _imPath if os.path.isdir(cP)]) == len(_imPath):
		if len(_zIndex) != 2:
			sys.exit("You need to provide the range (min, max) of Z index to merge.")
		# if args.amideRotMatrix != None and len(args.amideRotMatrix) != 9:
		# 	sys.exit("You need to provide the nine values of amide rotation matrix.")
		listIm, imSpacing = load3DImages(_imPath, _zIndex, None, None)
	# TODO: either check for a specific format or enable multiple format 
	elif len([cP for cP in _imPath if os.path.isfile(cP)]) == len(_imPath):
		if _binFormat != None:
			# Explicit it is a binary image
			imFormat = _binFormat
			listIm, imSpacing = load3DImages(_imPath, _zIndex, None, imFormat)
		elif _imPath[0].endswith(".hdr"):
			# CASToR images
			imFormat, imPath = extractCastorImInfo(_imPath)
			listIm, imSpacing = load3DImages(imPath, _zIndex, None, imFormat)
		elif _imPath[0].endswith(".nii.gz"):
			listIm, imSpacing = load3DImages(_imPath, _zIndex, None, None)
		elif _imPath[0].endswith(".npy"):
			# With numpy, we assume that the file hold either...
			#    a 3D numpy array as a stack of 2D images 
			#    a 2D numpy array
			# TODO: Enable also multiple numpy 3D images
			if _nImSpacing is None:
				sys.exit("The argument nSpacing is required when working with np.array.")
			imSpacing = list(_nImSpacing)
			tmp = np.load(_imPath[0])
			if tmp.ndim == 3:
				if len(_imPath) != 1:
					sys.exit("For 3D numpy images, only one image series is " \
					         "supported.")
				listIm = [tmp[i, :, :].T for i in range(tmp.shape[0])] 
			else:
				listIm = [np.load(_imPath[i]).T for i in range(len(_imPath))] 
		else:
			# We assume it is a series of images in dicom format 									
			listIm, imSpacing = loadDicom_2DImages(_imPath)
	elif len([cP for cP in _imPath if os.path.exists(cP)]) != len(_imPath):
		tmp = [cP for cP in _imPath if not os.path.exists(cP)]
		sys.exit("The path(s) " + str(tmp) + " does not exist.")
	else:
		sys.exit("All the path provided should be either folders or files.")

	return listIm, imSpacing 


def loadDicom_2DImages(_listofPath):
	'''
	Def: Load the images to analyze.
	@_listofPath (List, Str): List of the path for the images that need to be loaded.
	Return:
		listIm (List, 2D numpy array)
		imSpacing (List, 2 floats)
	TODO: Check/garantee that it is in mm.
	TODO: Check/garantee that it is stored cIm[yaxis, xaxis]
	'''	
	listIm = []
	for l, path in enumerate(_listofPath):
		tmpDicom = readDicomFile(path)
		cIm = (tmpDicom.pixel_array * tmpDicom.RescaleSlope) + tmpDicom.RescaleIntercept
		if cIm.ndim != 2:
			sys.exit("The script works only for two dimensions image.")
		if l != 0:
			if cIm.shape != listIm[0].shape:
				sys.exit("The image number " + str(l) + " does not have the same " \
				         "number of pixels has the first one.")
			if tmpDicom.PixelSpacing != imSpacing:
				sys.exit("The image number " + str(l) + " does not have the same " \
				         "pixel spacing as the first one.")
		else:
			imSpacing = tmpDicom.PixelSpacing
		listIm += [cIm[::-1, :],]
	return listIm, imSpacing


def load3DImages(_listofPath, _zIndex, _amideRotMatrix, _imFormat):
	'''
	Def: Load the images to analyze.
	@_listofPath (List, Str): List of the path for the images that need to be loaded.
	@_zIndex (Two integer): The range, defined as min and max, of Z index to merge.
	@_amideRotMatrix (List, Float)[9]: The rotation matrix, row-wise, to apply to the 
		image before doing the Z-wise merge. 
	@_imFormat (List, Float)[6]: The number of voxel in z, y and x axis and the 
		size of the voxel in (x, y). After that, it is an offset (in bytes) for the 
		image. If the value is -1, it is used to indicates that we are dealing with 
		CASToR images. The last value is used to indicates the the number of bytes used
		to store the image voxel.
	Return:
		listIm (List, 2D numpy array)
		imSpacing (List, 2 floats)
	TODO: Check/garantee that it is in mm.
	TODO: Check/garantee that it is stored cIm[yaxis, xaxis]
	'''	
	listIm = []
	for l, path in enumerate(_listofPath):
		if _imFormat != None:
			imShape = [int(_imFormat[2]), int(_imFormat[1]), \
							int(_imFormat[0])]
			if _imFormat[-2] == -1:
				# CASToR
				cIm = np.fromfile(path, dtype=np.float32, offset=0).reshape(imShape)
			else:
				# binary
				voxType = np.dtype(getattr(np, 'float' + str(_imFormat[-1])))
				cIm = np.fromfile(path, dtype=voxType, 
				                  offset=_imFormat[-1]).reshape(imShape)
			imSpacing = [_imFormat[3], _imFormat[3]]
		elif path.endswith(".nii.gz"):
			nii_img  = nib.load(path)
			cIm = nii_img.get_fdata().T
			# We do not use vozel size in the z dimension
			imSpacing = nii_img.header.get_zooms()[:-1]
		else:
			cIm = loadDicomFromDirectory(path)[::-1, ::-1, :]
		# if _amideRotMatrix != None:
		# 	amideRot = Rotation.from_matrix(np.asarray(_amideRotMatrix).reshape(3, 3))
		# 	amideEuler = amideRot.as_euler('xyz', degrees=True)

		# 	cIm = ndimage.rotate(cIm, amideEuler[2], axes=(2,1), reshape=False, order=1)
		# 	cIm = ndimage.rotate(cIm, amideEuler[0], axes=(1,0), reshape=False, order=1)
		# 	cIm = ndimage.rotate(cIm, amideEuler[1], axes=(0,2), reshape=False, order=1)
		
			tmp = [os.path.join(path, f) for f in os.listdir(path) \
			            if os.path.splitext(f)[1].lower() == '.dcm']
			tmpDicom = readDicomFile(tmp[0])
			if l != 0:
				if cIm.shape != imShape:
					sys.exit("The image number " + str(l) + " does not have the same " \
					         "number of pixels has the first one.")
				if tmpDicom.PixelSpacing != imSpacing:
					sys.exit("The image number " + str(l) + " does not have the same " \
					         "pixel spacing as the first one.")
			else:
				imSpacing = tmpDicom.PixelSpacing
				imShape = cIm.shape

		if bShowAxialView == True and l == 0:
			showAxialView(cIm, _zIndex)
			sys.exit("Exiting since axial view was shown.")

		cIm = np.sum(cIm[_zIndex[0]:(_zIndex[1] + 1)], axis=0)

		listIm += [cIm,]
	return listIm, imSpacing
	
	
def extractCastorImInfo(paths):
	'''
	Def: Extract information about the image reconstructed by Castor, such as 
	     voxelization and path of the image.
	@_listofPath (List, Str): List of the path for the images that need to be loaded.
	Return:
		imFormat (List, Float)[5] The number of voxel in z, y and x axis and the 
			size of the voxel. The last is a flag for CASToR (2.)
		imPath  (List, str) Path towards the .img file.
	'''	
	imPath = []
	imFormat = [-1, -1, -1, -1, -1]
	for path in paths:
		cHdr = open(path, 'r').readlines()
		cImPath = None
		for line in cHdr:
			if line.startswith("!name of data file := "):
				cImPath = line.split("!name of data file := ")[1].rstrip()
				cImPath = os.path.dirname(path) + "/" + cImPath
			elif line.startswith("!matrix size "):
				tmp = line.split(" ")
				dimId = int(tmp[2][1]) - 1
				# TODO: ensure that all images have same format?
				imFormat[dimId] = int(tmp[4].rstrip())
			elif line.startswith("scaling factor (mm/pixel) [3] := "):
				imFormat[3] = float(\
							line.split("scaling factor (mm/pixel) [3] := ")[1].rstrip())

		if cImPath != None:
			imPath.append(cImPath) 
		else:
			sys.exit("Path to binary data of image was not found in " + str(path))

	return imFormat, imPath


def readDicomFile(_filename):
	'''
	Def.: Read a single DICOM image
	@_filename (str): Path to a DICOM image. 
	'''
	return pydicom.read_file(_filename)


def loadDicomFromDirectory(_directory):
	'''
	Def.: Load a 3D DICOM image from a directory, and return 3D numpy image.
	@_directory (str): Directory where the DICOM images are saved. 
	'''

	filenames = [os.path.join(_directory, f) for f in os.listdir(_directory) 
				if os.path.splitext(f)[1].lower() == '.dcm']
	filenames = sorted(filenames)

	ds = readDicomFile(filenames[0])
	nx = ds.Rows
	ny = ds.Columns
	nz = ds.NumberOfSlices
	nt = ds.NumberOfTimeSlices if hasattr(ds, 'NumberOfTimeSlices') else 1

	if nz * nt != len(filenames):
		raise ValueError('Number of dcm files does not match the NumberOfSlices tag')

	im = np.zeros((nt, nz, ny, nx))
	for i, f in enumerate(sorted(filenames)):
		ds = readDicomFile(f)
		im[i // nz][i % nz] = (ds.pixel_array * ds.RescaleSlope) + ds.RescaleIntercept
	return im[0] if nt == 1 else im



#########################################################################################
# Neccessary to compute the Rayleigh Criterion:
#########################################################################################
def checkTriangleValidity(_lpConfig, _tolRel=0.25, _behavior='warn'):
	'''
	Def: Check if the triangle coordinates provided by the user make sense. Currently, 
		it does not try to correct the configuration. 
		Check:
			1) Distances between any pair of triangle coordinates should be 
				2 * (nbSpots - 1) * spotsSize
	@_lpConfig (Dict): Configuration of the lines profile to extract.
	@_tolRel (Float): Give a tolerance, relative to the sector spots size, for the check.
	@_behavior (Str): Provide instructions on how the script should react from the 
		result of the check.
	Return:
		Dict
			spotsSize: The size of the spots to study.
			nbRows: The number of rows for each spots to study.
			triangle: The position of the three spots that delimit the trinagle of each 
				spots to study.
	'''
	nbSector = len(_lpConfig['spotsSize'])
	# Tell the user which pair of coordinates of the triangle from which the problem 
	# arise.
	triangCase = ["Third vs first", "First vs second", "Second vs third"]
	
	for l in range(nbSector):
		cTriangle = _lpConfig["triangle"][l]
		cSpotsSize = _lpConfig["spotsSize"][l]
		cNbRows = _lpConfig['nbRows'][l]
		for k, cCase in enumerate([-1, 0, 1]):
			# 1)
			cLength = np.linalg.norm(cTriangle[cCase] - cTriangle[cCase + 1])
			expLength =  cSpotsSize * 2 * (cNbRows - 1)
			if np.abs(cLength - expLength) > _tolRel * cSpotsSize:
				if _behavior == 'warn':
					mess = "The length of the triangle side " + triangCase[k] \
							+ " in the " + str(l + 1) + " sector does not correspond " \
							+ "to what was predicted from the configuration file."
					warnings.warn(mess)


def genLpExtremumPos(_lpConfig):
	"""
	Def.: Generate the extrema positions (start and end) of each line profile.
	@_lpConfig (Dict): Configuration of the lines profile to extract.
	Return:
	lpExtPos (list of lists): Edges of every line profile arranged as
				[nbSector, nbAngle, nbRow, nbLpCurrSectorAndRow, ptsExt, nDim]
	spotsCenter (List of numpy arrays): Positions of spots centers.
	# cNbLpForOneAngle = int(cNbRows * (cNbRows - 1) / 2)
	"""
	# Constants:
	nbAngle = 3
	nbSector = len(_lpConfig["spotsSize"])
	
	lpExtPos = nbSector *  [None,]
	spotsCenter = nbSector *  [None,]
	for cSect in range(nbSector):		
		cSpotSize = _lpConfig["spotsSize"][cSect]
		cNbRows = _lpConfig["nbRows"][cSect]
		cTriangExtCenterPos = _lpConfig["triangle"][cSect]
		
		# Compute the center of all the spots
		spotsCenter[cSect] = np.zeros((int(cNbRows * (cNbRows + 1) / 2), 2))
		spotsCenter[cSect][0, :] = cTriangExtCenterPos[0]
		cSpot = 1
		cTriangRightExtDir = cTriangExtCenterPos[1] - cTriangExtCenterPos[0]
		cTriangRightExtDir /= np.linalg.norm(cTriangRightExtDir)
		cTriangLeftExtDir = cTriangExtCenterPos[2] - cTriangExtCenterPos[0]
		cTriangLeftExtDir /= np.linalg.norm(cTriangLeftExtDir)
		cTriangRightToLeftExtDir = cTriangExtCenterPos[2] - cTriangExtCenterPos[1]
		cTriangRightToLeftExtDir /= np.linalg.norm(cTriangRightToLeftExtDir)
		cNbSpotsInRow = 2
		for cRow in range(1, cNbRows):
			cRowRightExtPos = cTriangExtCenterPos[0] \
								+ cRow * 2.0 * cSpotSize * cTriangRightExtDir
			spotsCenter[cSect][cSpot, :] = cRowRightExtPos
			cSpot += 1
			cRowLeftExtPos = cTriangExtCenterPos[0] \
								+ cRow * 2.0 * cSpotSize * cTriangLeftExtDir
			for cSpotInRow in range(1, cNbSpotsInRow - 1):
				spotsCenter[cSect][cSpot, :] = cRowRightExtPos \
								+ cSpotInRow * 2.0 * cSpotSize * cTriangRightToLeftExtDir
				cSpot += 1
			spotsCenter[cSect][cSpot, :] = cRowLeftExtPos
			cSpot += 1
			cNbSpotsInRow += 1
		
		# Compute all the line profile extremum.
		lpExtPos[cSect] = nbAngle * [None,]	
		# Away from center. (Shortest to longest row)
		lpExtPos[cSect][0] = (cNbRows - 1) * [None,]
		cLeftSpot = 0
		for cRow in range(1, cNbRows):
			cLeftSpot += 1 # Trick of only need to skip one	
			nbLpInCurrRow = cRow
			lpExtPos[cSect][0][cRow - 1] = nbLpInCurrRow * [None,]
			for cLp in range(nbLpInCurrRow):
				firstExtPos = spotsCenter[cSect][cLeftSpot, :] \
								- 0.5 * cSpotSize * cTriangRightToLeftExtDir
				cRightSpot = cLeftSpot + 1
				secExtPos = spotsCenter[cSect][cRightSpot, :] \
								+ 0.5 * cSpotSize * cTriangRightToLeftExtDir
				lpExtPos[cSect][0][cRow - 1][cLp] = np.stack((firstExtPos, secExtPos))
				cLeftSpot = cRightSpot
		# From the right side. (Longest to shortest row)
		lpExtPos[cSect][1] = (cNbRows - 1) * [None,]
		for cRow in range(0, cNbRows - 1):
			cLeftSpot = int((cRow + 1) * (cRow + 2) / 2) - 1
			nbLpInCurrRow = cNbRows - cRow - 1
			lpExtPos[cSect][1][cRow] = nbLpInCurrRow * [None,]
			for cLp in range(nbLpInCurrRow):
				cRightSpot = cLeftSpot + cLp + cRow + 1
				firstExtPos = spotsCenter[cSect][cLeftSpot, :] \
								- 0.5 * cSpotSize * cTriangRightExtDir
				secExtPos = spotsCenter[cSect][cRightSpot, :] \
								+ 0.5 * cSpotSize * cTriangRightExtDir
				lpExtPos[cSect][1][cRow][cLp] = np.stack((firstExtPos, secExtPos))
				cLeftSpot = cRightSpot
		# From the left side. (Longest to shortest row)
		lpExtPos[cSect][2] = (cNbRows - 1) * [None,]
		for cRow in range(0, cNbRows - 1):
			cFirstSpot = int((cRow + 1) * cRow / 2)
			nbLpInCurrRow = cNbRows - cRow - 1
			lpExtPos[cSect][2][cRow] = nbLpInCurrRow * [None,]
			for cLp in range(nbLpInCurrRow):
				cSecSpot = cFirstSpot + cLp + cRow + 2
				firstExtPos = spotsCenter[cSect][cFirstSpot, :] \
								- 0.5 * cSpotSize * cTriangLeftExtDir
				secExtPos = spotsCenter[cSect][cSecSpot, :] \
								+ 0.5 * cSpotSize * cTriangLeftExtDir
				lpExtPos[cSect][2][cRow][cLp] = np.stack((firstExtPos, secExtPos))
				cFirstSpot = cSecSpot

	return lpExtPos, spotsCenter


def extractAllLineProfile(_im, _lpExtPos, _spotsSize, _imSpacing, _roiRatio, \
                          _lpSampleRate=10000):
	"""
	Def.: Extract the lines profile defined in _lpExtPos from _im. The lines profile are 
		segmented in a dictionnary of [sector, angle, row]. Each line profile is 
		segmented in three: [first peak, valley second peak].
	@_im (2D numpy array): The image to study.
	@_lpExtPos (list of lists): Edges of every line profile arranged as
				[nbSector, nbAngle, nbRow, nbLpCurrSectorAndRow, ptsExt, nDim]
	@_spotsSize (List, Float): The diameter of the spots to study.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_roiRatio (Tuple, 2 Float): Ratio of the region of interest to keep. The first is 
	    for the two spots and the second is for the background/valley.
	@_lpSampleRate (Integer): The number of samples used for the extraction of the 
		profiles from the image.
	Return: 
		List of lists, contains every segments of all line profiles, arranged as
				[nbSector, nbAngle, nbRow, nbLpCurrSectorAndRow, section, val]
	"""
	imSize = np.asarray(_im.shape[0]) * _imSpacing
	
	# Init first level of the dictionnary
	segLp = len(_lpExtPos) * [None,]
	for cSect in range(len(_lpExtPos)):
		# Init second level of the dictionnary
		segLp[cSect] = len(_lpExtPos[cSect]) * [None,]
		cSpotSize = _spotsSize[cSect]
		for cAng in range(len(_lpExtPos[cSect])):
			# Init third level of the dictionnary
			segLp[cSect][cAng] = len(_lpExtPos[cSect][cAng]) * [None,]
			for cRow in range(len(_lpExtPos[cSect][cAng])):
				# Init fourth level of the dictionnary
				segLp[cSect][cAng][cRow] = len(_lpExtPos[cSect][cAng][cRow]) * [None,]
				for cLp in range(len(_lpExtPos[cSect][cAng][cRow])):
					currLp = _lpExtPos[cSect][cAng][cRow][cLp]
					segLp[cSect][cAng][cRow][cLp] = extractLineProfile(_im, currLp, \
													cSpotSize, imSize, _roiRatio, \
													_lpSampleRate)
	return segLp
	
	
def extractLineProfile(_im, _lpExt, _cSpotSize, _imSize, _roiRatio, _lpSampleRate):
	"""
	Def.: Extract the line profile described in _lpExt from _im. The line profile is 
		divided in three array, which should represents, in order, the first spots, the 
		valley and the second spots. The nearest interpolator is used to extract the 
		line profile.
	@ (2D numpy array): The image to study.
	@_lpExt (List, List, 2 Floats): The (x, y) positions of the two extremities of the 
		line profile to extract.
	@_cSpotSize (Float): The diameter of the spot to study.
	@_imSize (1D numpy array, Float): The physical size, in x and y axes, of the image.
	@_roiRatio (Tuple, 2 Float): Ratio of the region of interest to keep. The first is 
	    for the two spots and the second is for the background/valley.
	@_lpSampleRate (Integer): The number of samples used for the extraction of the 
		profiles from the image.
	Return:
		List of 3 numpy array, Float
	"""
	# Discretize line profile sampling.
	x = np.linspace(_lpExt[0][0], _lpExt[1][0], _lpSampleRate)
	y = np.linspace(_lpExt[0][1], _lpExt[1][1], _lpSampleRate)
	linDom = np.sqrt((x - x[0])**2 + (y - y[0])**2)
	
	# Extract the values of the line profile using nearest interpolation.
	# TODO: Offer fancier interpolation?
	# TODO: Make sure that x, y axis are always stored as _im[y, x]
	x_indexSpace = np.floor(((x + (_imSize[1] / 2.0)) / _imSize[1]) * _im.shape[1]) 
	x_indexSpace = x_indexSpace.astype(np.int32)
	y_indexSpace = np.floor(((y + (_imSize[0] / 2.0)) / _imSize[0]) * _im.shape[0]) 
	y_indexSpace = y_indexSpace.astype(np.int32)
	# TODO: read and adapt orientation. 
	# lineProfile = _im[::-1, :][y_indexSpace, x_indexSpace]
	lineProfile = _im[:, :][y_indexSpace, x_indexSpace]
	
	# Find the starting/ending position of the three regions of interest. 
	# TODO: Apply a search since currently it is only an approximation (which is most 
	#       likely more than good enough with a _lpSampleRate of 10K.)
	spotFilter = (1.0 - _roiRatio[0]) / 2.0
	bckgndFilter = (1.0 - _roiRatio[1]) / 2.0
	# TODO: The following assumes that linDom is approximatly 3.0 * _cSpotSize
	firSpotStartingIndex = np.searchsorted(linDom, spotFilter * _cSpotSize)
	firSpotEndingIndex = np.searchsorted(linDom, (1.0 - spotFilter) * _cSpotSize)
	# The transition position between the first spot and the valley should be after a 
	# lenght of the spot size.
	bckgndStartingIndex = np.searchsorted(linDom, _cSpotSize + bckgndFilter * _cSpotSize)
	bckgndEndingIndex = np.searchsorted(linDom, _cSpotSize \
	                                            + (1.0 - bckgndFilter) * _cSpotSize)
	# The transition position between the valley and the second spot should be after a 
	# lenght of two times the spot size.
	secSpotStartingIndex = np.searchsorted(linDom, 2.0 * _cSpotSize \
	                                               + spotFilter * _cSpotSize)
	secSpotEndingIndex = np.searchsorted(linDom, 2.0 * _cSpotSize \
	                                            + (1.0 - spotFilter) * _cSpotSize)

	segLp = {}
	# First spot
	segLp["firstSpot"] = {"linPos": linDom[firSpotStartingIndex:firSpotEndingIndex], \
	                      "imVal": lineProfile[firSpotStartingIndex:firSpotEndingIndex]}
	# Valley inbetween the two spots
	segLp["valley"] = {"linPos": linDom[bckgndStartingIndex:bckgndEndingIndex], \
	                      "imVal": lineProfile[bckgndStartingIndex:bckgndEndingIndex]}
	lineProfile[bckgndStartingIndex:bckgndEndingIndex]
	# Second spot
	segLp["secSpot"] = {"linPos": linDom[secSpotStartingIndex:secSpotEndingIndex], \
	                      "imVal": lineProfile[secSpotStartingIndex:secSpotEndingIndex]}

	return segLp
	
	
def computeSectorValleyToPeak(_segLp, _metric, _roiRatio, _vprHistos, _imSpacing, 
	                          _savePath):
	"""
	Def.: Compute the valley to peak ratio of each sector using the metric _metric. 
		The valley to peak ratio of a sector is defined as the mean of the valley to 
		peak ratio of every line profile of that sector.
	@_segLp (List of lists): Segments of line profiles for all sections, angles and rows.
	@_metric (String): The metric used to evaluate the valley to peak ratio of a line 
		profile.
		min_max: Minimum of the valley divided by the mean of the maximum of the two 
			peaks.
	@_roiRatio (Tuple, 2 Float): Ratio of the region of interest to keep. The first is 
	    for the two spots and the second is for the background/valley.
	@_vprHistos (bool): If true, we show the histograms of VPRs.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_savePath (Str): Path where to save the figure.
	Return:
		mean (1D numpy array, Float)
		std (1D numpy array, Float)
	"""	
	# The metric is computed for each sector
	vToP = np.zeros(len(_segLp))
	vToPSquared = np.zeros(len(_segLp))
	nbLp = np.zeros(len(_segLp))
	vToPhistoAll = []
	for cSect in range(len(_segLp)):
		vToPhisto = []
		for cAng in range(len(_segLp[cSect])):
			for cRow in range(len(_segLp[cSect][cAng])):
				for cLp in range(len(_segLp[cSect][cAng][cRow])):
					cVal = metricDict[_metric](_segLp[cSect][cAng][cRow][cLp])
					vToP[cSect] += cVal
					vToPSquared[cSect] += cVal**2
					vToPhisto.append(cVal)
					nbLp[cSect] += 1

		vToPhistoAll.append(vToPhisto)

	# Plot histograms of vToP
	if _vprHistos:
		showValleyToPeakRatioHistograms(vToPhistoAll, _metric, _roiRatio, _imSpacing, \
		                                _savePath)

	# Compute resolvability of rods in each sector
	resolv = []
	for idx, vToPhisto in enumerate(vToPhistoAll):
		vToPhisto = np.asarray(vToPhisto)
		resolv.append(np.round(100 * np.sum(vToPhisto <= RAYLEIGH_CRITERION) \
											/ vToPhisto.shape[0], 1))

	mean = vToP / nbLp
	# VAR = E[X^2] - E[X]^2
	std = np.sqrt(vToPSquared / nbLp - mean**2)

	return mean, std, resolv


def lineProfilMetric_minMax(lineProfil):
	"""
	Def.: Compute the ratio of spots to valley as mean of the maximum of both spots 
	    compared to minimum of the valley.
	@_segLp (Dict, Dict, array): Segmented line profile of two spots and the valley 
	    inbetween.
	"""	
	val = lineProfil["valley"]["imVal"].min() \
	                                 / (0.5 * lineProfil["firstSpot"]["imVal"].max() \
	                                    + 0.5 * lineProfil["secSpot"]["imVal"].max())
	
	return val


def lineProfilMetric_mean(lineProfil):
	"""
	Def.: Compute the ratio of spots to valley as mean of both spots compared to the 
	    mean of the valley.
	@_segLp (Dict, Dict, array): Segmented line profile of two spots and the valley 
	    inbetween.
	"""
	val = np.mean(lineProfil["valley"]["imVal"]) \
	                              / (0.5 * np.mean(lineProfil["firstSpot"]["imVal"]) 
	                                 + 0.5 * np.mean(lineProfil["secSpot"]["imVal"]))
	
	return val


def lineProfilMetric_meanPeakMax(lineProfil):
	"""
	Def.: Compute the ratio of spots to valley as max of the mean of each spots compared 
	    to the mean of the valley.
	@_segLp (Dict, Dict, array): Segmented line profile of two spots and the valley 
	    inbetween.
	"""
	val = np.mean(lineProfil["valley"]["imVal"]) \
	                               / (max(np.mean(lineProfil["firstSpot"]["imVal"]), \
	                                       np.mean(lineProfil["secSpot"]["imVal"])))
	
	return val


def lineProfilMetric_parabola(lineProfil):
	"""
	Def.: Compute the ratio of spots to valley as the mean of the max of each spots 
	    compared to the min of the valley. The min/max of each segment is defined
	    from a quadratic fit of the segment.
	@_segLp (Dict, Dict, array): Segmented line profile of two spots and the valley 
	    inbetween.
	TODO: Force the sign of the second degree fit?
	"""
	segVal = {}
	for i, cSeg in enumerate(lineProfil):
		fit_param = np.polyfit(lineProfil[cSeg]["linPos"], lineProfil[cSeg]["imVal"], 2)
		func = np.poly1d(fit_param)
		segQuadFit = func(lineProfil[cSeg]["linPos"])
		# Parobola mix/max OUTSIDE the line profile?
		if np.all(np.diff(segQuadFit) < 0.0) or np.all(np.diff(segQuadFit) > 0.0):
			segVal[cSeg] = np.mean(segQuadFit)
		else:
			if cSeg == "valley":
				# Parobola orientation incorrect for a valley?
				if fit_param[0] < 0.0:
					segVal[cSeg] = np.mean(segQuadFit)
				else:
					# Parabola fit is ignored if it goes lower than the valley values
					# In theory, this make it more robust for spot larger than the  
					# spatial resolution
					segVal[cSeg] = max(np.min(segQuadFit), \
					                    np.min(lineProfil[cSeg]["imVal"]))
			else:
				# Parobola orientation incorrect for a spot?
				if fit_param[0] > 0.0:
					segVal[cSeg] = np.mean(segQuadFit)
				else:
					# Parabola fit is ignored if it goes higher than the spots values
					# In theory, this make it more robust for spot larger than the  
					# spatial resolution
					segVal[cSeg] = min(np.max(segQuadFit), \
					                    np.max(lineProfil[cSeg]["imVal"]))
	
	val = min(segVal["valley"] / (0.5 * segVal["firstSpot"] + 0.5 * segVal["secSpot"]), \
	          1.0)

	return val



#########################################################################################
# Line profile configurations:
#########################################################################################
def loadLineProfileConfig(_lpConfigPath):
	'''
	Def: Load the congiguration parameters for the lines profile.
	@_lpConfigPath (Str): Path of the json with the configuration parameters for the 
		analysis.
	Return:
		Dict
			spotsSize: The size of the spots to study.
			nbRows: The number of rows for each spots to study.
			triangle: The position of the three spots that delimit the trinagle of each 
				spots to study.
	'''
	with open(_lpConfigPath) as f:
		lpConfig = json.load(f)
	
	nbSector = len(lpConfig['spotsSize'])
	# Easier to work with array.
	lpConfig["triangle"] = np.asarray(lpConfig["triangle"])
	
	if len(lpConfig['nbRows']) != nbSector:
		sys.exit("The number of rows quantity provided does not match with the number " \
					"of spots size provided.")
	if lpConfig["triangle"].shape[0] != nbSector:
		sys.exit("The number of triangle coordinates provided does not match with the " \
					"number of spots size provided.")
	if lpConfig["triangle"].shape[1:] != (3, 2):
		sys.exit("The triangle coordinates should be defined as three positions in 2D.")
	
	return lpConfig


def genJsonExample(_path):
	"""
	Def.: Create an example of the json required for the configuration of the lines 
		profile.
	@_path (Str): Path where to save the json example.
	"""	
	if os.path.isfile(_path) == True:
		sys.exit(f"The file {_path} already exist. This feature would destroy " \
					+ "that file. Thus, we prefer to stop here.")
	
	config = {}
	config["spotsSize"] = [2.4, 2.0, 1.7, 1.35, 1.0, 0.75]
	config["nbRows"] = [2, 2, 3, 3, 5, 6]
	config["triangle"] = [[[-3.09, -4.62], [-7.85, -5.42], [-4.80, -9.17]],
							[[2.21, -6.28], [0.87, -10.02], [4.83, -9.22]], 
							[[5.85, -2.58], [10.29, -7.83], [12.64, -1.46]],
							[[4.62, 2.18], [10.08, 2.93], [6.49, 7.15]],
							[[0.34, 3.14], [3.12, 10.74], [-4.59, 9.51]], 
							[[-2.87, 0.2], [-6.94, 5.76], [-10.42, -0.60]]]

	json_dump = json.dumps(config, indent=4)
	
	with open(_path, 'w') as f:
		json.dump(config, f)



#########################################################################################
# Visualization features:
#########################################################################################
def showAxialView(_im, _zIndex, _savePath=None):
	"""
	Def.: Show an axial slice of _im with an overlay of the z index that will be summed.
	@_im (3D numpy array): The image to study.
	@_zIndex (2 Int): The z index that will be summed. 
	@_savePath (Str): Path where to save the figure. (Not used atm)
	"""
	# Parameters of visualization.
	color = 'orange'
	
	fig, ax = plt.subplots()
	# Trick to show the slice where there is something when the image is not at the 
	# center
	if _im[:, _im.shape[1]//2, :].sum() > _im[:, :, _im.shape[2]//2].sum():
		imSlice = _im[:, _im.shape[1]//2, :]
	else:
		imSlice = _im[:, :, _im.shape[2]//2]
	ax.imshow(imSlice, interpolation='none', cmap='Greys_r', origin='lower', \
	          vmax=np.sort(imSlice.flatten())[int(0.999 * _im.shape[0] * _im.shape[2])])
	ax.axis('off')
	
	rect = plt.Rectangle((0, _zIndex[0]), imSlice.shape[-1], 
	                     _zIndex[1] - _zIndex[0] + 1, edgecolor=color, facecolor='none')
	ax.add_patch(rect) 

	print("The --showAxialView argument only helps in selecting the argument (-z) which "
	      "defines the slice(s) in the transverse view to do the resolvability/VPR "
	      "analysis. You can see your current selection in the image. "
		  "The script will exit after the image is closed.")

	plt.tight_layout(pad=0)
	if _savePath is None:
		plt.show()	
	else:
		plt.savefig(_savePath + "zROI.png")
		plt.close()


def showTrianglePosOnImage(_im, _imSpacing, _lpConfig, _savePath):
	"""
	Def.: Show the triangles defined in _lpConfig superposed on _im.
	@_im (2D numpy array): The image to study.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_lpConfig (Dict): Configuration of the lines profile to extract.
	@_savePath (Str): Path where to save the figure.
	TODO: Possibly 1 or 1.5 pixels offset.
	"""
	# Parameters of visualization.
	color = ['r', 'g', 'b']
	
	# Put image in physical space.
	tmpX = _im.shape[0] * _imSpacing[0] / 2.0
	tmpY = _im.shape[1] * _imSpacing[1] / 2.0
	imDom = (-tmpY, tmpY, -tmpX, tmpX)
	
	fig, ax = plt.subplots()
	ax.imshow(_im, interpolation='bilinear', cmap='Greys_r', extent=imDom, origin='lower', \
				vmax=np.sort(_im.flatten())[int(0.999 * _im.shape[0] * _im.shape[1])])
	ax.axis('off')
	
	# For all sectors defined in _lpConfig
	for l in range(_lpConfig["triangle"].shape[0]):
		# For the three sides of each triangle
		for i in range(3):
			# Expected spots at the triangle vertices.
			draw_circle = plt.Circle(xy=_lpConfig["triangle"][l][i], \
						radius=_lpConfig["spotsSize"][l]/2.0, \
						edgecolor=color[i], facecolor='none')
			ax.add_artist(draw_circle)
			# Vertices of the triangle
			draw_circle = plt.Circle(xy=_lpConfig["triangle"][l][i], \
						radius=_imSpacing[0]/2.0, \
						edgecolor=color[i], facecolor=color[i])
			ax.add_artist(draw_circle)
	plt.tight_layout(pad=0)
	if _savePath is None:
		plt.show()	
	else:
		plt.savefig(_savePath + "posTriang.png")
		plt.close()
	
	
def showSpotsPosOnImage(_im, _imSpacing, _lpConfig, _spotsCenter, _savePath):
	"""
	Def.: Show the spots position predicted from the configuration file superposed on 
		_im. 
	@_im (2D numpy array): The image to study.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_lpConfig (Dict): Configuration of the lines profile to extract.
	@_spotsCenter (List of numpy arrays): Positions of spots centers.
	@_savePath (Str): Path where to save the figure.
	TODO: Possibly 1 on 1.5 pixels offset.
	"""
	# Put image in physical space.
	tmpX = _im.shape[0] * _imSpacing[0] / 2.0
	tmpY = _im.shape[1] * _imSpacing[1] / 2.0
	imDom = (-tmpY, tmpY, -tmpX, tmpX)
	
	fig, ax = plt.subplots()
	ax.imshow(_im, interpolation='none', cmap='Greys_r', extent=imDom, origin='lower', \
				vmax=np.sort(_im.flatten())[int(0.999 * _im.shape[0] * _im.shape[1])])
	ax.axis('off')
	
	# For all sectors defined in _lpConfig
	for l in range(len(_spotsCenter)):
		# For all spots of the current sector
		for cSpots in _spotsCenter[l]:
			# Expected spots at the triangle vertices.
			draw_circle = plt.Circle(xy=cSpots, radius=_lpConfig["spotsSize"][l] / 2.0, \
										edgecolor="red", facecolor='none')
			ax.add_artist(draw_circle)
			# Computed position of the spots
			draw_circle = plt.Circle(xy=cSpots, radius=_imSpacing[0] / 2.0, 
										edgecolor="red", facecolor='red')
			ax.add_artist(draw_circle)
	plt.tight_layout(pad=0)
	if _savePath is None:
		plt.show()	
	else:
		plt.savefig(_savePath + "posSpot.png")
		plt.close()
	
	
def showLinesProfileOnImage(_im, _imSpacing, _lpConfig, _spotsCenter, _lpExtPos, 
	                        _savePath):
	"""
	Def.: Show the lines profile defined in the configuration file superposed on _im.
	@_im (2D numpy array): The image to study.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_lpConfig (Dict): Configuration of the lines profile to extract.
	@_spotsCenter (List of numpy arrays): Positions of spots centers.
	@_lpExtPos (list of lists): Edges of every line profile arranged as
				[nbSector, nbAngle, nbRow, nbLpCurrSectorAndRow, ptsExt, nDim]
	@_savePath (Str): Path where to save the figure.
	TODO: Possibly 1 on 1.5 pixels offset.
	TODO: Add an option for seeing subset of line profile?
	"""
	# Parameters of visualization.
	color = ['r', 'g', 'b']	
	linewidthChoice = _lpConfig["spotsSize"]

	# Put image in physical space.
	tmpX = _im.shape[0] * _imSpacing[0] / 2.0
	tmpY = _im.shape[1] * _imSpacing[1] / 2.0
	imDom = (-tmpY, tmpY, -tmpX, tmpX)
	
	fig, ax = plt.subplots()
	ax.imshow(_im, interpolation='none', cmap='Greys_r', extent=imDom, origin='lower', \
				vmax=np.sort(_im.flatten())[int(0.999 * _im.shape[0] * _im.shape[1])])
	ax.axis('off')
	
	for l in range(len(_spotsCenter)):
		for cSpots in _spotsCenter[l]:
			draw_circle = plt.Circle(xy=cSpots, radius=_imSpacing[0]/2.0, 
										edgecolor="red", facecolor='red')
			ax.add_artist(draw_circle)	
	
	# For each section and each angle, we plot a line between adjacent points.
	for cSec in range(len(_lpExtPos)):
		# Plot for three angular directions in the triangles.
		for cAng in range(3):
				for cRowLp in _lpExtPos[cSec][cAng]:
					for cLp in cRowLp:
						lpPerpDir = (cLp[1, :] - cLp[0, :])[::-1] \
						             * np.array([-1.0, 1.0])
						lpPerpDir /= np.linalg.norm(lpPerpDir)
						oneSide = cLp + 0.5 * _imSpacing[0] * lpPerpDir
						otherSide = cLp - 0.5 * _imSpacing[0] * lpPerpDir
						plt.fill(np.append(oneSide[:, 0], otherSide[::-1, 0]), \
						         np.append(oneSide[:, 1], otherSide[::-1, 1]), \
						         color=color[cAng])
	plt.tight_layout(pad=0)
	if _savePath is None:
		plt.show()	
	else:
		plt.savefig(_savePath + "lineProfile.png")
		plt.close()


def showValleyToPeakRatioHistograms(_vToPhistoAll, _metric, _roiRatio, _imSpacing, \
	                                _savePath):
	'''
	Def: Display valley-to-peak ratios histograms for each phantom's sections.
	@_vToPhistoAll (List of lists): All VPR values, divided by phantom section.
	@_metric (String): Metric used to evaluate the valley to peak ratio of a line 
		profile.
	@_roiRatio (Tuple, 2 Float): Ratio of the region of interest to keep. The first is 
	    for the two spots and the second is for the background/valley.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_savePath (Str): Path where to save the figure.
	'''
	fig, axs = plt.subplots(3, 3)
	fig.set_size_inches(18.5, 10.5)

	# Plot phantom centered above all histograms
	imCrop = cropImage(im, _imSpacing)
	axs[0, 0].set_visible(False)
	maxVal = np.sort(imCrop.flatten())[int(0.999 * imCrop.shape[0] * imCrop.shape[1])]
	axs[0, 1].imshow(imCrop, interpolation='none', cmap='Greys_r', origin='lower', \
	                 vmax=maxVal)
	axs[0, 1].set_xticks([])
	axs[0, 1].set_yticks([])
	axs[0, 1].autoscale(False)
	axs[0, 2].set_visible(False)

	# Plot histograms. Starts with smallest sector, so we reverse the array
	_vToPhistoAll = _vToPhistoAll[::-1]
	spotsSizeReversed = lpConfig["spotsSize"][::-1]
	for idx, vToPhisto in enumerate(_vToPhistoAll):

		# Stats on vpr (mean, stdev and resolvability %)
		vToPhisto = np.asarray(vToPhisto)
		mean_vpr = np.round(np.mean(vToPhisto), 3)
		stdev_vpr = np.round(np.std(vToPhisto), 3)
		resolv = np.round(100 * np.sum(vToPhisto <= RAYLEIGH_CRITERION) 
											/ vToPhisto.shape[0], 1)

		# Plotting with results labeled on each histogram
		row = idx // 3 + 1
		col = idx % 3
		binning = np.arange(0.0, 1.0 + 0.02, 0.02)
		axs[row, col].yaxis.get_major_locator().set_params(integer=True)
		labelString = f"Average VPR    : {mean_vpr:.3f} ± {stdev_vpr:.3f}\n" \
		            + f"Resolvability  : {resolv:.1f} %\n" \
		            + f"VPR metric     : {_metric}\n" \
		            + f"ROI on peaks   : {100 * _roiRatio[0]:.0f} %\n" \
		            + f"ROI on valleys : {100 * _roiRatio[1]:.0f} %" 
		axs[row, col].hist(vToPhisto, bins=binning, label=labelString, color="#045a8d", \
		                   edgecolor='black')
		axs[row, col].set_xlim(0, 1)
		axs[row, col].axvline(x=RAYLEIGH_CRITERION, color="red", linestyle='--', 
		                      label="VPR=0.735")
		axs[row, col].set_title(f"Rods: {spotsSizeReversed[idx]:.2f} mm", 
		                        fontweight='bold')
		axs[row, col].set_xlabel("Valley-to-peak ratio")
		axs[row, col].set_ylabel("Counts")
		axs[row, col].legend(fontsize=10, prop={'family': 'monospace'})

	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.20, 
	                    hspace=0.40)
	if _savePath is None:
		plt.show()	
	else:
		plt.savefig(_savePath + ".png")
		plt.close()


def cropImage(_im, _imSpacing, _minIntensityFrac=0.02):
	'''
	Def: Crop image to keep only a region around the phantom.
	@_im (2D numpy array): The image to crop.
	@_imSpacing (List, 2 floats): Physical space, in x and y axes, between the image 
		pixels.
	@_minIntensityFrac (Float, optional): Min intensity fraction of profile to find 
		edges.
	Return (2D numpy array): Image cropped to focus on the phantom.
	'''
	# Loop on X and Y of image, sum to create a profile, find values above a 
	# fraction of the max intensity, then find edges.
	idxsEdges = []
	for i in range(2):
		profileSummed = _im.sum(axis=i)
		threshold = _minIntensityFrac * np.max(profileSummed)
		idxsAbove = np.where(profileSummed > threshold)[0]
		idxsEdges.append([idxsAbove[0], idxsAbove[-1]])

	# Check if edges seem logical compare to estimated X and Y ranges (if not, don't 
	# crop). Find largest X and Y from outside rods of triangles.
	xOutside = lpConfig['triangle'][:, 1:, 0].flatten()
	yOutside = lpConfig['triangle'][:, 1:, 1].flatten()
	offsetSpotSize = int(np.max(lpConfig['spotsSize']) / _imSpacing[0])
	xWidthInIdxs = int((xOutside.max() - xOutside.min()) / _imSpacing[0]) + offsetSpotSize
	yWidthInIdxs = int((yOutside.max() - yOutside.min()) / _imSpacing[1]) + offsetSpotSize

	if ((idxsEdges[1][1] - idxsEdges[1][0]) > xWidthInIdxs) and \
		((idxsEdges[0][1] - idxsEdges[0][0]) > yWidthInIdxs):
		imCrop = _im[idxsEdges[1][0]:idxsEdges[1][1], idxsEdges[0][0]:idxsEdges[0][1]]
	else:
		imCrop = _im[:, :]

	return imCrop



#########################################################################################
# Formating features:
#########################################################################################
def genImageName(_imId, _imPath, _stackofIt):
	"""
	Def.: Generate a name for the image being processed.
	@_imId: Position of the image being processing in _listIm.
	@_imPath: List of paths to the image to process.
	@_stackofIt: Flag that indicates that the current 3D numpy array is a stack of 2D 
	  images.
	"""
	if _stackofIt == True:
		# Special cases where a 3D numpy array contains multiple 2D images that are 
		# step from a iterative reconstruction process.
		# Example: path/potato_step_10_50.npy
		fName = _imPath[0].split("/")[-1]
		maxIte = fName.split("_")[-1].split(".npy")[0]
		cIte = (_imId + 1) * int(fName.split("step")[-1].split("_")[0])
		cImName = f"{fName}, iteration {cIte} of {maxIte}" 
	else:
		cImName = _imPath[_imId]

	return cImName


def genFigureName(_imId, _listIm, _cImName, _pathForFigures, _stackofIt):
	"""
	Def.: Generate a name for the figure related to the image being processed.
	@_imId: Position of the image being processing in _listIm.
	@_listIm: The list of image to process.
	@_cImName: The name of the image being processed.
	@_pathForFigures: Path where to save the figures.
	@_stackofIt: Flag that indicates that the current 3D numpy array is a stack of 2D 
	  images.
	"""
	if _pathForFigures is None:
		cFigSavePath = None 
	else:
		if len(_listIm) != 1:
			# Since the path of the image file is not used in naming the figures
			uniqueFigId = str(_imId) + "_"
		else:
			uniqueFigId = ""

		if os.path.isdir(_cImName):
			# If the file name is a directory, take last folder as a name
			cHistoName = os.path.basename(os.path.normpath(_cImName))
		else:
			cHistoName = os.path.basename(_cImName)

		if _stackofIt == True:
			cHistoName = cHistoName.replace(" ", "_").replace(".npy,", "")
		cFigSavePath = args.pathForFigures + "histo_" + uniqueFigId \
								+ cHistoName

	return cFigSavePath


def fmtResolved(_val, _fmt=''):
	"""
	Def.: Create a string of _val with a color that indicates that it is corresponds to
		 a spots that is resolved.
	@_val: The object to format as a string.
	@_fmtL The format to use.
	"""
	return f"\x1b[1;32;49m{_val:{_fmt}}\x1b[0m"


def fmtNotResolved(_val, _fmt=''):
	"""
	Def.: Create a string of _val with a color that indicates that it is corresponds to
		 a spots that is not resolved.
	@_val: The object to format as a string.
	@_fmtL The format to use.
	"""
	return f"\x1b[1;31;49m{_val:{_fmt}}\x1b[0m"


def simplifyImagesName(_imName):
	"""
	Def.: Remove the redondant part of the name of the images. Only consideer redondancy
	      as a prefix of the images name. 
	@_imName: List of the image names.
	"""
	redondantPart = _imName[0]
	for cImName in _imName[1:]:
		tmp = ""
		for i, carac in enumerate(redondantPart):
			if carac == cImName[i]:
				tmp += carac
			else:
				break 
		redondantPart = tmp 
	
	if tmp != "":
		for i in range(len(_imName)):
			_imName[i] = _imName[i][len(redondantPart):]

	return _imName



#########################################################################################
# Script feature:
#########################################################################################
# Trick to be able to define metric as a dictionnary.
metricDict = {"min_max": lineProfilMetric_minMax,
              "mean": lineProfilMetric_mean,
              "meanPeakMax": lineProfilMetric_meanPeakMax, 
              "parabola": lineProfilMetric_parabola}


def parserCreator():
	parser = argparse.ArgumentParser(description="Use the method suggested in Hallen "
					"et al 2020 to extract the valley-to-peak ratio of the sectors "
					"of a Hot Spots phantom. The user provide a json file that give a "
					"description of the sectors to extract and the image is assumed "
					"to be in 2D.")

	# Basic:
	parser.add_argument('-f', action='store', type=str, required=True, \
						dest='imPath', nargs='+', default=[], \
						help='List of path to the images to process. Binary, CASToR, '
							'dicom and numpy are all partially supported in one way or '
							' the other.')
	parser.add_argument('-c', action='store', type=str, required=True, \
						dest='lpConfigPath', \
						help='Path to the json with the configuration of the lines ' 
							'profile.')
	parser.add_argument('-z', action='store', nargs=2, type=int, required=False, \
						dest='zIndex', \
						help='Needed to create a 2D image from a 3D image. It ' \
								'defines the span of Z index to sum.')
	parser.add_argument('-m', action='store', required=False, default="min_max", \
					 	dest='metric', choices=metricDict.keys(), \
						help='Metric to use for peak to valley computation.')
	parser.add_argument('-r', action='store', nargs=2, type=float, required=False, \
						dest='roiRatio', default=(1.0, 1.0), \
						help='Ratio of the region of interest to keep. The first is ' \
						    'for the two spots and the second is for the ' \
						    'background/valley.')
		
	# Features:
	parser.add_argument('--genJsonExample', action='store_true', required=False,\
						dest='genJsonExample', default=False, \
						help='Generate an example of json file used to store the '
							'configuration of the lines profile that is saved at '
							'lpConfigPath. Stop the script after the file is created.')	
	parser.add_argument('--showAxialView', action='store_true', required=False,\
						dest='showAxialView', default=False, \
						help='Display an axial view of the first image provided with ' 
						     'an overlay of the z indexes that will be summed (based of '
						     'the -z option). Only work for 3D images. The script exit '
						     'after showing the view.')
	parser.add_argument('--showTriangPos', action='store_true', required=False,\
						dest='showTriangPos', default=False, \
						help='Show the triangle positions on the first image provided '
							'for each sectors defined in the json file.')
	parser.add_argument('--showSpotsPos', action='store_true', required=False,\
						dest='showSpotsPos', default=False, \
						help='Show the predicted position of each spots of each zone '
							'on the first image provided.')
	parser.add_argument('--showLinesProfile', action='store_true', required=False,\
						dest='showLinesProfile', default=False, \
						help='Show the resulting lines profiles on the first image '
							'provided.')
	parser.add_argument('--imageShown', action='store', required=False, type=int, \
						dest='imageShown', default=0, \
						help='Select which images is used for showing the triangle, '
							'spots or line profiles.')
	parser.add_argument('--showVprHistos', action='store_true', required=False,\
						dest='showVprHistos', default=False, \
						help='Show valley-to-peak ratio histograms.')
	parser.add_argument('--pathForFigures', type=str, required=False,\
						dest='pathForFigures', default=None, \
						help='The figures are now saved as file(s) at the place of '
						     'showing them.')
	# parser.add_argument('-amideRotMatrix', nargs="+", type=float, required=False,\
	# 					dest='amideRotMatrix', default=None, \
	# 					help='The rotation matrix, found with Amide, to reorient that '
	# 							'will be used to orient the 3D images.')
	parser.add_argument('--binFormat', nargs=4, type=float, required=False,\
						dest='binFormat', default=None, \
						help='Number of voxels in z, y and x dimensions followed by ' 
							'the size of the voxels in (x, y)')
	parser.add_argument('--binOffset', type=int, required=False,\
						dest='binOffset', default=32, \
						help='The offset, in bytes, to read a binary images.')
	parser.add_argument('--saveResults', type=str, required=False,\
						dest='saveResults', default=None, \
						help='Save a compact version of the results in csv format.')
	parser.add_argument('--nSpacing', action='store', nargs=2, type=float, \
						required=False, dest='nSpacing', default=None, \
						help='The spacing of the pixel in the (x, y) axis. Only ' \
						     'needed (and used) when loading a numpy array.')
	parser.add_argument('--npy3DIsStackOfIter', action='store_true', required=False, \
						dest='stackofIt', default=False, \
						help='Used for better verbose when showing results from a ' \
								'stack of 2D numpy arrays that are iterations.')
	parser.add_argument('--binVoxelFloatType', type=int, required=False,\
						dest='binVoxFType', default=32, \
						help='Set the number of bytes used for voxel float value.')
	parser.add_argument('--simplifyName', action='store_true', required=False, \
						dest='simplifyName', default=False, \
						help='When saving results, simplify the image name.')
						
	return parser.parse_args()


def argsValidator(_args):
	"""
	Def.: Basic check of some arguments.
	"""
	if _args.metric == "parabola":
		if np.any(np.array(_args.roiRatio) < roiRatioForParabola):
			print()
			mess = "At least one of the ROI ratio is lower than "\
			        "{roiRatioForParabola}. Using small values when choosing the " \
			        "parabola metric can yield unreliable fits. The user roiRatio " \
			        "values are kept, but it is recommended using values closer or " \
			        "equal to 1."
			warnings.warn(mess)

	if _args.pathForFigures is not None and _args.showVprHistos == False \
		              and _args.showTriangPos == False and _args.showSpotsPos == False \
		              and _args.showLinesProfile == False:
		print()
		mess = "The argument pathForFigures is only used when one of the show " \
				"option is enabled. Since it is not the case, it is not used."
		warnings.warn(mess)



#########################################################################################
# Main : We use this to make the module usable as a script and a method.
#########################################################################################
if __name__=='__main__':
	'''
	Def: Script version of this module. Enable direct study of images from a 
	     configuration file.
	'''	
	args = parserCreator()

	argsValidator(args)
	
	if args.genJsonExample == True:
		genJsonExample(args.lpConfigPath)
		sys.exit("The Json example file was created. See " + args.lpConfigPath)
		
	if args.showAxialView == True:
		bShowAxialView = True

	if args.binFormat != None:
		binFormat = args.binFormat + (args.binOffset, args.binVoxelFloatType)
	else:
		binFormat = None
	listIm, imSpacing = loadImages(args.imPath, args.zIndex, binFormat, \
	                               args.nSpacing)

	lpConfig = loadLineProfileConfig(args.lpConfigPath)
	
	checkTriangleValidity(lpConfig, _behavior="")
	lpExtPos, spotsCenter = genLpExtremumPos(lpConfig)
	
	if args.showTriangPos == True:
		showTrianglePosOnImage(listIm[args.imageShown], imSpacing, lpConfig, \
		                       args.pathForFigures)
	if args.showSpotsPos == True:
		showSpotsPosOnImage(listIm[args.imageShown], imSpacing, lpConfig, spotsCenter, \
		                    args.pathForFigures)
	if args.showLinesProfile == True:
		showLinesProfileOnImage(listIm[args.imageShown], imSpacing, lpConfig, 
								spotsCenter, lpExtPos, args.pathForFigures)
	
	if args.saveResults != None:
		resultArray = -np.ones(shape=(len(listIm), 3 * len(lpConfig["spotsSize"])))
		imName = len(listIm) * [""]

	for l, im in enumerate(listIm):
		cImName = genImageName(l, args.imPath, args.stackofIt)
		cFigSavePath = genFigureName(l, listIm, cImName, args.pathForFigures, \
		                             args.stackofIt)

		segLp = extractAllLineProfile(im, lpExtPos, lpConfig["spotsSize"], imSpacing, \
		                              args.roiRatio)	
		results = computeSectorValleyToPeak(segLp, args.metric, args.roiRatio, 
		                                    args.showVprHistos, imSpacing, 
		                                    cFigSavePath)

		if args.saveResults == None:
			print("\n=== Results ===")
			print(f"Image             : {cImName}")
			print(f"Z-slice(s)        : {args.zIndex}")
			print(f"VPR metric        : {args.metric}")
			print(f"ROI on peaks      : {100.0 * args.roiRatio[0]:.0f}%")
			print(f"ROI on valleys    : {100.0 * args.roiRatio[1]:.0f}%")
			print('Rod size (mm)     : ', end='')
			for i,v in enumerate(results[0]):
				if v < RAYLEIGH_CRITERION:
					print(fmtResolved(lpConfig['spotsSize'][i], '.3f') + '\t', end='')
				else:
					print(fmtNotResolved(lpConfig['spotsSize'][i], '.3f') + '\t', end='')

			print('\nAverage VPR       : ', end='')
			for avgVpr in results[0]:
				if avgVpr < RAYLEIGH_CRITERION:
					print(fmtResolved(avgVpr, '.3f') + '\t', end='')
				else:
					print(fmtNotResolved(avgVpr, '.3f') + '\t', end='')
			print('\nStdev VPR         :', '\t'.join(f'{k:.3f}' for k in results[1]))
			print('Resolvability [%] :', '\t'.join(f'{k:.1f}'.rjust(5) \
			                                             for k in results[2]))

			print(f"\n---> In {fmtResolved('green')} if phantom section has an " +
			      f"average VPR < 0.735, in {fmtNotResolved('red')} otherwise.\n")

		else:
			nbSpot = len(lpConfig["spotsSize"])
			resultArray[l, :nbSpot] = np.round(results[0], 3)
			resultArray[l, nbSpot:(2 * nbSpot)] = np.round(results[1], 3)
			resultArray[l, (2 * nbSpot):] = results[2]
			# Remove file extention from name
			imName[l] = cImName.rsplit( ".", 1)[0]

	if args.saveResults != None:
		iterables = [["Average VPR", "Stdev VPR", "Resolvability [%]"], \
		              lpConfig['spotsSize']]
		header = pd.MultiIndex.from_product(iterables, names=["Metric", "Rod size (mm)"])
		if args.simplifyName == True and len(imName) != 1:
			imName = simplifyImagesName(imName) 
		data = pd.DataFrame(resultArray, columns=header, index=imName)
		filename, extension = os.path.splitext(args.saveResults)
		if extension not in ["", ".csv"]:
			print("Forcing the extension of the results file to be a csv.")  
		data.to_csv(filename + ".csv")
	
	
