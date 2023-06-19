import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import scipy.ndimage
import cv2
import os

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction')
#from utils import read_nii, generate_all_MIPs_SUV, generate_all_MIPs_CT, preprocess_df, preprocess_CT_HU_values
from utils import preprocess_df
from config import parse_args

def main(args):
	#path_data = args.data_path
	path_output = args.path_multi_angled_multi_channel_2D_projections
	df = pd.read_csv(args.path_df) #DataFrame containing all the original tumorous scans.
	#df = df[df["diagnosis"]=="NEGATIVE"].reset_index(drop=True)
	df = preprocess_df(df, args)
	#print("151 to end")
	#df = df[151:].reset_index(drop=True)

	for idx, row in tqdm(df.iterrows(), total=len(df)):
		pat_ID = row["pat_ID"]
		scan_date = row["scan_date"]

		SUV = read_nii(row["SUV"])
		SUV_B = read_nii(row["SUV_bone"])
		SUV_LT = read_nii(row["SUV_lean"])
		SUV_AT = read_nii(row["SUV_adipose"])
		SUV_A = read_nii(row["SUV_air"])

		CT = read_nii(row["CT"])
		CT_B = read_nii(row["CT_bone"])
		CT_LT = read_nii(row["CT_lean"])
		CT_AT = read_nii(row["CT_adipose"])
		CT_A = read_nii(row["CT_air"])

		CT_LT, CT_AT, CT_A = preprocess_CT_HU_values(CT_LT), preprocess_CT_HU_values(CT_AT), preprocess_CT_HU_values(CT_A)

		SEG = read_nii(row["SEG"])

		#Generate different multi-directional SUV channels based on CT HU values
		save_path = os.path.join(path_output, pat_ID, scan_date)
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		if not os.path.exists(os.path.join(save_path, "SUV_MIP")): #SUV_MIP, SUV_bone, SUV_lean, SUV_adipose, SUV_air
			os.makedirs(os.path.join(save_path, "SUV_MIP"))
		if not os.path.exists(os.path.join(save_path, "SUV_bone")):
			os.makedirs(os.path.join(save_path, "SUV_bone"))
		if not os.path.exists(os.path.join(save_path, "SUV_lean")):
			os.makedirs(os.path.join(save_path, "SUV_lean"))
		if not os.path.exists(os.path.join(save_path, "SUV_adipose")):
			os.makedirs(os.path.join(save_path, "SUV_adipose"))
		if not os.path.exists(os.path.join(save_path, "SUV_air")):
			os.makedirs(os.path.join(save_path, "SUV_air"))

		if not os.path.exists(os.path.join(save_path, "CT_MIP")): #SUV_MIP, SUV_bone, SUV_lean, SUV_adipose, SUV_air
			os.makedirs(os.path.join(save_path, "CT_MIP"))
		if not os.path.exists(os.path.join(save_path, "CT_bone")):
			os.makedirs(os.path.join(save_path, "CT_bone"))
		if not os.path.exists(os.path.join(save_path, "CT_lean")):
			os.makedirs(os.path.join(save_path, "CT_lean"))
		if not os.path.exists(os.path.join(save_path, "CT_adipose")):
			os.makedirs(os.path.join(save_path, "CT_adipose"))
		if not os.path.exists(os.path.join(save_path, "CT_air")):
			os.makedirs(os.path.join(save_path, "CT_air"))

		if not os.path.exists(os.path.join(save_path, "SEG")):
			os.makedirs(os.path.join(save_path, "SEG"))

		generate_all_MIPs_SUV(save_path, SUV, SUV_B, SUV_LT, SUV_AT, SUV_A, SEG, args.SUV_min, args.SUV_max, args.rotation_min, args.rotation_max, args.rotation_interval)
		generate_all_MIPs_CT(save_path, CT, CT_B, CT_LT, CT_AT, CT_A, SEG, args.CT_min, args.CT_max, args.rotation_min, args.rotation_max, args.rotation_interval)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")
