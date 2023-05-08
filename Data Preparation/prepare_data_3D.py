import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/GitHub/Tumor Segmentation followed by Outcome Prediction/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction')
from config import parse_args
from utils import read_nii, generate_binary_masks, generate_HU_channels, save_all_nii, generate_SUV_CT_collage, generate_SUV_CT_collage

def main(args):
	df = pd.read_csv(args.path_df)
	#df = df[df["diagnosis"]!="LUNG_CANCER"].reset_index(drop=True)

	for index, row in tqdm(df.iterrows(), total=df.shape[0]):
		path_CT, path_SUV, path_SEG = row["CT"], row["SUV"], row["SEG"]
		pat_ID, scan_date = row["pat_ID"], row["scan_date"]
		disease_type = row["diagnosis"]

		save_path_nii = os.path.join(args.output_path, "3D_CT_SUV_Data", pat_ID, scan_date)
		if not os.path.exists(save_path_nii):
			os.makedirs(save_path_nii)

		save_path_visualizations = os.path.join(args.output_path, "Visualization")
		if not os.path.exists(save_path_visualizations):
			os.makedirs(save_path_visualizations)

		#Load Image
		CT_arr, SUV_arr, SEG_arr = read_nii(path_CT), read_nii(path_SUV), read_nii(path_SEG)
		#Generate binary masks for different tissues, corresponding to different HU windows
		bone_mask, lean_mask, adipose_mask, air_mask = generate_binary_masks(CT_arr, args)

		#Generate SUV channels corresponding to different masks
		SUV_arr_B, SUV_arr_LT, SUV_arr_AT, SUV_arr_A = generate_HU_channels(SUV_arr, bone_mask, lean_mask, adipose_mask, air_mask)
		#Get CT channels corresponding to different masks
		CT_arr_B, CT_arr_LT, CT_arr_AT, CT_arr_A = generate_HU_channels(CT_arr, bone_mask, lean_mask, adipose_mask, air_mask)

		#Save all the nii Images for CT and SUV
		save_all_nii(path_CT, save_path_nii, CT_arr_B, CT_arr_LT, CT_arr_AT, CT_arr_A, "CT")
		save_all_nii(path_SUV, save_path_nii, SUV_arr_B, SUV_arr_LT, SUV_arr_AT, SUV_arr_A, "SUV")

		#Generate Collages for visualization
		generate_SUV_CT_collage(SEG_arr, SUV_arr, SUV_arr_B, SUV_arr_LT, SUV_arr_AT, SUV_arr_A, CT_arr, CT_arr_B, CT_arr_LT, CT_arr_AT, CT_arr_A, save_path_visualizations, pat_ID, scan_date, disease_type)
		break


if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")