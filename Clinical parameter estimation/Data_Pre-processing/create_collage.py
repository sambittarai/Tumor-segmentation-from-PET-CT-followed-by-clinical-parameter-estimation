import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import scipy.ndimage
import cv2
import os

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Clinical parameter estimation')
#from utils import read_nii, generate_all_MIPs_SUV, generate_all_MIPs_CT, preprocess_df
from config import parse_args

def main(args):
	#path_data = args.data_path
	path_output = args.data_path_MIPs_collages
	df = pd.read_csv(args.path_df_rot_mips)
	df = df[df.angle.isin(args.include_angles_collages)].reset_index(drop=True)
	scan_dates = np.unique(df["scan_date"])

	for scan_date in tqdm(scan_dates):
		df_temp = df[df["scan_date"]==scan_date]
		save_path_temp = os.path.join(path_output, str(df_temp["pat_ID"].iloc[0]), str(df_temp["scan_date"].iloc[0]))
		if not os.path.exists(save_path_temp):
			os.makedirs(save_path_temp)

		SUV_MIP_list = []
		SUV_bone_list = []
		SUV_lean_list = []
		SUV_adipose_list = []
		SUV_air_list = []
		CT_MIP_list = []
		CT_bone_list = []
		CT_lean_list = []
		CT_adipose_list = []
		CT_air_list = []

		for idx, row in df_temp.iterrows():
			SUV_MIP_list.append(np.load(row["SUV_MIP"]))
			#SUV_MIP_list.append(np.load(row["SUV_MIP"])[:,85:-85]) #only for 45 degrees

			SUV_bone_list.append(np.load(row["SUV_bone"]))
			#SUV_bone_list.append(np.load(row["SUV_bone"])[:,85:-85])

			SUV_lean_list.append(np.load(row["SUV_lean"]))
			#SUV_lean_list.append(np.load(row["SUV_lean"])[:,85:-85])

			SUV_adipose_list.append(np.load(row["SUV_adipose"]))
			#SUV_adipose_list.append(np.load(row["SUV_adipose"])[:,85:-85])

			SUV_air_list.append(np.load(row["SUV_air"]))
			#SUV_air_list.append(np.load(row["SUV_air"])[:,85:-85])

			CT_MIP_list.append(np.load(row["CT_MIP"]))
			#CT_MIP_list.append(np.load(row["CT_MIP"])[:,85:-85])

			CT_bone_list.append(np.load(row["CT_bone"]))
			#CT_bone_list.append(np.load(row["CT_bone"])[:,85:-85])

			CT_lean_list.append(np.load(row["CT_lean"]))
			#CT_lean_list.append(np.load(row["CT_lean"])[:,85:-85])

			CT_adipose_list.append(np.load(row["CT_adipose"]))
			#CT_adipose_list.append(np.load(row["CT_adipose"])[:,85:-85])

			CT_air_list.append(np.load(row["CT_air"]))
			#CT_air_list.append(np.load(row["CT_air"])[:,85:-85])

		#Generate collages
		SUV_MIP = np.concatenate((SUV_MIP_list[0], SUV_MIP_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "SUV_MIP.npy"), SUV_MIP)

		SUV_bone = np.concatenate((SUV_bone_list[0], SUV_bone_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "SUV_bone.npy"), SUV_bone)

		SUV_lean = np.concatenate((SUV_lean_list[0], SUV_lean_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "SUV_lean.npy"), SUV_lean)

		SUV_adipose = np.concatenate((SUV_adipose_list[0], SUV_adipose_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "SUV_adipose.npy"), SUV_adipose)

		SUV_air = np.concatenate((SUV_air_list[0], SUV_air_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "SUV_air.npy"), SUV_air)

		CT_MIP = np.concatenate((CT_MIP_list[0], CT_MIP_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "CT_MIP.npy"), CT_MIP)

		CT_bone = np.concatenate((CT_bone_list[0], CT_bone_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "CT_bone.npy"), CT_bone)

		CT_lean = np.concatenate((CT_lean_list[0], CT_lean_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "CT_lean.npy"), CT_lean)

		CT_adipose = np.concatenate((CT_adipose_list[0], CT_adipose_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "CT_adipose.npy"), CT_adipose)

		CT_air = np.concatenate((CT_air_list[0], CT_air_list[1]), axis=1)
		np.save(os.path.join(save_path_temp, "CT_air.npy"), CT_air)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")