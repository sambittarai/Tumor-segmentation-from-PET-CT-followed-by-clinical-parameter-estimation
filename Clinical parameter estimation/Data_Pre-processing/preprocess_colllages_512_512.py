import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def process_img_arr(img, dim=0):
    target_size = 512
    h = img.shape[dim]
    if h < target_size:
        diff_h = target_size - h
        diff_temp_h = int(diff_h/2)
        pad_width = ((diff_temp_h, diff_temp_h), (0, 0))
        padded_image = np.pad(img, pad_width, mode='constant')
        
    elif h > target_size:
        diff_h = h - target_size
        diff_temp_h = int(diff_h/2)
        if diff_temp_h%2==0:
            if dim==0:
                padded_image = img[diff_temp_h:-diff_temp_h,:]
            elif dim==1:
                padded_image = img[:,diff_temp_h:-diff_temp_h]
        elif diff_temp_h%2!=0:
            if dim==0:
                padded_image = img[diff_temp_h:-diff_temp_h-1,:]
            elif dim==1:
                padded_image = img[:,diff_temp_h:-diff_temp_h-1]
    return padded_image

def resize_array(array):
    # Get the current shape of the array
    current_height, current_width = array.shape[:2]

    # If the height is greater than 512, crop the array
    if current_height > 512:
        # Calculate the starting and ending indices for the crop along the first dimension
        start_index = (current_height - 512) // 2
        end_index = start_index + 512

        # Crop the array along the first dimension
        resized_array = array[start_index:end_index, :]

    # If the height is less than 512, pad the array
    elif current_height < 512:
        # Calculate the amount of zero-padding needed along the first dimension
        padding_height = 512 - current_height

        # Pad the array with zeros along the first dimension
        resized_array = np.pad(array, ((0, padding_height), (0, 0)), mode='constant')

    # If the height is already 512, no resizing is needed
    else:
        resized_array = array

    return resized_array

df = pd.read_csv("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/df_rot_mips_collages.csv")

df["CT_MIP"] = df["CT_MIP"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["SUV_MIP"] = df["SUV_MIP"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["CT_bone"] = df["CT_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["SUV_bone"] = df["SUV_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["CT_lean"] = df["CT_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["SUV_lean"] = df["SUV_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["CT_adipose"] = df["CT_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["SUV_adipose"] = df["SUV_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["CT_air"] = df["CT_air"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")
df["SUV_air"] = df["SUV_air"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]")


save_path = "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[-45, +45]_512_512"
#save_path = "/media/sambit/Seagate Expansion Drive/Nouman/Original_Lean_window"

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
	channels = ["SUV_MIP", "CT_MIP", "SUV_bone", "CT_bone", "SUV_lean", "CT_lean", "SUV_adipose", "CT_adipose", "SUV_air", "CT_air"]
	for channel in channels:
		path = row[channel]
		img = np.load(path)
		img1 = process_img_arr(img, dim=0)
		img2 = process_img_arr(img1, dim=1)
		img_final = resize_array(img2)
		#print(img.shape)
		save_path_temp = os.path.join(save_path, row["pat_ID"], row["scan_date"])
		if not os.path.exists(save_path_temp):
		    os.makedirs(save_path_temp)
		np.save(os.path.join(save_path_temp, channel + ".npy"), img_final)