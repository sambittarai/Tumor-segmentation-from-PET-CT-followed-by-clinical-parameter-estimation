import numpy as np
import pandas as pd
import os
import torch
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
from tqdm import tqdm
import cc3d
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import nibabel as nib
import torch
import numpy as np
import scipy.ndimage

def convert_npy_to_nii(GT, prediction, pat_id, scan_date, files, args):
	FN = GT - prediction
	FN = np.where(FN == -1, 0, FN)
	path = os.path.join(args.path_FN_Data, pat_id, scan_date)
	if len(np.unique(FN)) == 2:
		SUV = nib.load(files['SUV'])
		FN_img = nib.Nifti1Image(FN, SUV.affine)
		if not os.path.exists(path):
			os.makedirs(path)
		nib.save(FN_img, os.path.join(path, "SEG.nii.gz"))

	else:
		#Don't generate data for this scan
		print(pat_id, scan_date)

def read_nii(path):
	img = sitk.ReadImage(path)
	img_arr = sitk.GetArrayFromImage(img)
	img_arr = np.transpose(img_arr, (2,1,0))
	return img_arr

def preprocess_CT_HU_values(arr):
	return arr - np.min(arr)

def generate_MIPs(Data, suv_min, suv_max, intensity_type=None, img_type=None):
    """
    Generate MIPs for PET Data.
    Maximum Intensity Projection
    
    Data - PET Data.
    MIP - Maximum/Mean/Std Intensity Projection.
    """
    PET = Data
    if img_type == "SUV":
    	PET = np.clip(Data, suv_min, suv_max)

    if intensity_type == "maximum":
    	MIP_PET = np.max(PET, axis=1).astype("float")
    elif intensity_type == "mean":
    	MIP_PET = np.mean(PET, axis=1).astype("float")
    elif intensity_type == "std":
    	MIP_PET = np.std(PET, axis=1).astype("float")
    elif intensity_type == "sum": 
    	MIP_PET = np.sum(PET, axis=1).astype("float")

    MIP_PET = MIP_PET/np.max(MIP_PET)# Pixel Normalization, value ranges b/w (0,1).
    if img_type != "CT_MIP":
    	MIP_PET = np.absolute(MIP_PET - np.amax(MIP_PET))
    MIP_PET = cv2.rotate(MIP_PET, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return MIP_PET

def generate_MIPs_Seg(Data):
    """
    Generate MIPs for Segmentation Data.
    
    Data - Segmentation Mask.
    """
    MIP_Seg = np.max(Data, axis=1).astype("float")
    #MIP_Seg = np.sum(Data, axis=1).astype("float")
    MIP_Seg = cv2.rotate(MIP_Seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return MIP_Seg



def generate_all_MIPs_SUV(save_path, SUV, SUV_B, SUV_LT, SUV_AT, SUV_A, SEG, suv_min, suv_max, rot_min=-90, rot_max=90, rot_interval=1):
	"""
	Generate rotating 2D MIPs along coronal direction from (-90, 90).
	"""
	for i in tqdm(range(rot_min, rot_max+1, rot_interval)):
		print("SUV angle: ", i)

		file_path = os.path.join(save_path, "SUV_MIP", str(i) + ".npy")
		if not os.path.isfile(file_path):
			suv_temp = scipy.ndimage.rotate(SUV, angle=i, axes=(0,1))
			suv_MIP = generate_MIPs(suv_temp, suv_min, suv_max, intensity_type="maximum", img_type="SUV")
			suv_MIP = suv_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SUV_MIP", str(i) + ".npy"), suv_MIP)

			del suv_temp

		file_path = os.path.join(save_path, "SUV_bone", str(i) + ".npy")
		if not os.path.isfile(file_path):
			suv_b_temp = scipy.ndimage.rotate(SUV_B, angle=i, axes=(0,1))
			suv_b_MIP = generate_MIPs(suv_b_temp, suv_min, suv_max, intensity_type="maximum", img_type="SUV")
			suv_b_MIP = suv_b_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SUV_bone", str(i) + ".npy"), suv_b_MIP)

			del suv_b_temp
			
		file_path = os.path.join(save_path, "SUV_lean", str(i) + ".npy")
		if not os.path.isfile(file_path):
			suv_lt_temp = scipy.ndimage.rotate(SUV_LT, angle=i, axes=(0,1))
			suv_lt_MIP = generate_MIPs(suv_lt_temp, suv_min, suv_max, intensity_type="maximum", img_type="SUV")
			suv_lt_MIP = suv_lt_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SUV_lean", str(i) + ".npy"), suv_lt_MIP)

			del suv_lt_temp

		file_path = os.path.join(save_path, "SUV_adipose", str(i) + ".npy")
		if not os.path.isfile(file_path):
			suv_at_temp = scipy.ndimage.rotate(SUV_AT, angle=i, axes=(0,1))
			suv_at_MIP = generate_MIPs(suv_at_temp, suv_min, suv_max, intensity_type="maximum", img_type="SUV")
			suv_at_MIP = suv_at_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SUV_adipose", str(i) + ".npy"), suv_at_MIP)

			del suv_at_temp

		file_path = os.path.join(save_path, "SUV_air", str(i) + ".npy")
		if not os.path.isfile(file_path):
			suv_a_temp = scipy.ndimage.rotate(SUV_A, angle=i, axes=(0,1))
			suv_a_MIP = generate_MIPs(suv_a_temp, suv_min, suv_max, intensity_type="maximum", img_type="SUV")
			suv_a_MIP = suv_a_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SUV_air", str(i) + ".npy"), suv_a_MIP)

			del suv_a_temp

		file_path = os.path.join(save_path, "SEG", str(i) + ".npy")
		if not os.path.isfile(file_path):
			seg_temp = scipy.ndimage.rotate(SEG, angle=i, axes=(0,1))
			seg_MIP = generate_MIPs_Seg(seg_temp)
			seg_MIP = seg_MIP[:,60:-60]
			np.save(os.path.join(save_path, "SEG", str(i) + ".npy"), seg_MIP)

			del seg_temp
		#break

def generate_all_MIPs_CT(save_path, CT, CT_B, CT_LT, CT_AT, CT_A, SEG, ct_min, ct_max, rot_min=-90, rot_max=90, rot_interval=1):
	"""
	Generate rotating 2D MIPs along coronal direction from (-90, 90).
	"""
	for i in tqdm(range(rot_min, rot_max+1, rot_interval)):
		print("CT angle: ", i)

		file_path = os.path.join(save_path, "CT_MIP", str(i) + ".npy")
		#if not os.path.isfile(file_path):
			#print("CT_MIP")
		ct_temp = scipy.ndimage.rotate(CT, angle=i, axes=(0,1))
		ct_MIP = generate_MIPs(ct_temp, ct_min, ct_max, intensity_type="sum", img_type="CT_MIP")
		ct_MIP = (ct_MIP - np.min(ct_MIP)) / (np.max(ct_MIP) - np.min(ct_MIP))
		ct_MIP = ct_MIP[:,60:-60]
		np.save(os.path.join(save_path, "CT_MIP", str(i) + ".npy"), ct_MIP)

		del ct_temp

		file_path = os.path.join(save_path, "CT_bone", str(i) + ".npy")
		if not os.path.isfile(file_path):
			ct_b_temp = scipy.ndimage.rotate(CT_B, angle=i, axes=(0,1))
			ct_b_MIP = generate_MIPs(ct_b_temp, ct_min, ct_max, intensity_type="sum", img_type="CT")
			ct_b_MIP = (ct_b_MIP - np.min(ct_b_MIP)) / (np.max(ct_b_MIP) - np.min(ct_b_MIP))
			ct_b_MIP = ct_b_MIP[:,60:-60]
			np.save(os.path.join(save_path, "CT_bone", str(i) + ".npy"), ct_b_MIP)

			del ct_b_temp

		file_path = os.path.join(save_path, "CT_lean", str(i) + ".npy")
		if not os.path.isfile(file_path):
			ct_lt_temp = scipy.ndimage.rotate(CT_LT, angle=i, axes=(0,1))
			ct_lt_MIP = generate_MIPs(ct_lt_temp, ct_min, ct_max, intensity_type="sum", img_type="CT")
			ct_lt_MIP = (ct_lt_MIP - np.min(ct_lt_MIP)) / (np.max(ct_lt_MIP) - np.min(ct_lt_MIP))
			ct_lt_MIP = ct_lt_MIP[:,60:-60]
			np.save(os.path.join(save_path, "CT_lean", str(i) + ".npy"), ct_lt_MIP)

			del ct_lt_temp
			
		file_path = os.path.join(save_path, "CT_adipose", str(i) + ".npy")
		if not os.path.isfile(file_path):
			ct_at_temp = scipy.ndimage.rotate(CT_AT, angle=i, axes=(0,1))
			ct_at_MIP = generate_MIPs(ct_at_temp, ct_min, ct_max, intensity_type="sum", img_type="CT")
			ct_at_MIP = (ct_at_MIP - np.min(ct_at_MIP)) / (np.max(ct_at_MIP) - np.min(ct_at_MIP))
			ct_at_MIP = ct_at_MIP[:,60:-60]
			np.save(os.path.join(save_path, "CT_adipose", str(i) + ".npy"), ct_at_MIP)

			del ct_at_temp	
			
		file_path = os.path.join(save_path, "CT_air", str(i) + ".npy")
		if not os.path.isfile(file_path):
			ct_a_temp = scipy.ndimage.rotate(CT_A, angle=i, axes=(0,1))
			ct_a_MIP = generate_MIPs(ct_a_temp, ct_min, ct_max, intensity_type="sum", img_type="CT")
			ct_a_MIP = (ct_a_MIP - np.min(ct_a_MIP)) / (np.max(ct_a_MIP) - np.min(ct_a_MIP))
			ct_a_MIP = ct_a_MIP[:,60:-60]
			np.save(os.path.join(save_path, "CT_air", str(i) + ".npy"), ct_a_MIP)

			del ct_a_temp	
		#break


def preprocess_df(df):
	#Bone HU window
	df["SUV_bone"] = df["SUV"]
	df["SUV_bone"] = df["SUV_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["SUV_bone"] = df["SUV_bone"].str.replace("SUV.nii.gz", "SUV_bone.nii.gz")

	df["CT_bone"] = df["CT"]
	df["CT_bone"] = df["CT_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["CT_bone"] = df["CT_bone"].str.replace("CTres.nii.gz", "CT_bone.nii.gz")

	#Lean HU window
	df["SUV_lean"] = df["SUV"]
	df["SUV_lean"] = df["SUV_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["SUV_lean"] = df["SUV_lean"].str.replace("SUV.nii.gz", "SUV_lean_tissue.nii.gz")

	df["CT_lean"] = df["CT"]
	df["CT_lean"] = df["CT_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["CT_lean"] = df["CT_lean"].str.replace("CTres.nii.gz", "CT_lean_tissue.nii.gz")

	#Adipose HU window
	df["SUV_adipose"] = df["SUV"]
	df["SUV_adipose"] = df["SUV_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["SUV_adipose"] = df["SUV_adipose"].str.replace("SUV.nii.gz", "SUV_adipose_tissue.nii.gz")

	df["CT_adipose"] = df["CT"]
	df["CT_adipose"] = df["CT_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["CT_adipose"] = df["CT_adipose"].str.replace("CTres.nii.gz", "CT_adipose_tissue.nii.gz")

	#Air HU window
	df["SUV_air"] = df["SUV"]
	df["SUV_air"] = df["SUV_air"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["SUV_air"] = df["SUV_air"].str.replace("SUV.nii.gz", "SUV_air.nii.gz")

	df["CT_air"] = df["CT"]
	df["CT_air"] = df["CT_air"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
	df["CT_air"] = df["CT_air"].str.replace("CTres.nii.gz", "CT_air.nii.gz")

	return df
