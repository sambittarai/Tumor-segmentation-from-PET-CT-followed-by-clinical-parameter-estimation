import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Clinical parameter estimation')
from config import parse_args

def main(args):

    #Dataframe containing all the scans (tumor and normal)
    #df = pd.read_csv("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_final.csv")
    df = pd.read_csv(args.path_df)
    df = df.drop(columns=["Unnamed: 0"])
    df['sex'] = df['sex'].replace({'M': 1, 'F': 0})
    df["age"] = df["age"].str.replace("Y", "")
    df["age"] = df["age"].astype("int")
    df["MTV (ml)"] = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        MTV = 0
        img = sitk.ReadImage(row["SEG"])
        spacing = img.GetSpacing()
        img_arr = sitk.GetArrayFromImage(img)
        _, freq = np.unique(img_arr, return_counts=True)
        if len(freq) == 2:
            tumor_count = freq[1]
            MTV = tumor_count*spacing[0]*spacing[1]*spacing[2]*1e-3
            df["MTV (ml)"].iloc[idx] = MTV
        else:
            df["MTV (ml)"].iloc[idx] = 0
            
    def prepare_df_2D_rot_MIPs(path, df_df):
        df_new = df_df
        df = pd.DataFrame(columns=['pat_ID', 'scan_date', 'SUV_MIP', 'SUV_bone', 'SUV_lean', 'SUV_adipose', 'SUV_air', 'SEG', 'angle', 'diagnosis', 'age', 'sex', 'MTV (ml)'])
        for pat_id in tqdm(sorted(os.listdir(path))):
            for scan_date in sorted(os.listdir(os.path.join(path, pat_id))):
                df_new_temp = df_new[df_new["scan_date"]==scan_date].reset_index(drop=True)
                diagnosis = df_new_temp["diagnosis"][0]
                age = df_new_temp["age"][0]
                sex = df_new_temp["sex"][0]
                MTV = df_new_temp["MTV (ml)"][0]
                for degree in range(-90, 90+1, 5):
                    SUV_MIP_path = os.path.join(path, pat_id, scan_date, "SUV_MIP", str(degree) + ".npy")
                    SUV_bone_path = os.path.join(path, pat_id, scan_date, "SUV_bone", str(degree) + ".npy")
                    SUV_lean_path = os.path.join(path, pat_id, scan_date, "SUV_lean", str(degree) + ".npy")
                    SUV_adipose_path = os.path.join(path, pat_id, scan_date, "SUV_adipose", str(degree) + ".npy")
                    SUV_air_path = os.path.join(path, pat_id, scan_date, "SUV_air", str(degree) + ".npy")
                    SEG_path = os.path.join(path, pat_id, scan_date, "SEG", str(degree) + ".npy")
                    df_temp = pd.DataFrame({'pat_ID': [pat_id], 'scan_date': [scan_date], 'SUV_MIP': [SUV_MIP_path], 'SUV_bone': [SUV_bone_path], 'SUV_lean': [SUV_lean_path], 'SUV_adipose': [SUV_adipose_path], 'SUV_air': [SUV_air_path], 'SEG': [SEG_path], 'angle': [degree], 'diagnosis': [diagnosis], 'age': [age], 'sex': [sex], 'MTV (ml)': [MTV]})
                    df = df.append(df_temp, ignore_index=True)
        return df

    #path = "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Data_preparation/Output/Multi_directional_2D_MIPs"
    path = args.data_path_MIPs
    df_angle_wise = prepare_df_2D_rot_MIPs(path, df)

    df_angle_wise["CT_MIP"] = df_angle_wise["SUV_MIP"]
    df_angle_wise["CT_bone"] = df_angle_wise["SUV_bone"]
    df_angle_wise["CT_lean"] = df_angle_wise["SUV_lean"]
    df_angle_wise["CT_adipose"] = df_angle_wise["SUV_adipose"]
    df_angle_wise["CT_air"] = df_angle_wise["SUV_air"]
    df_angle_wise["CT_MIP"] = df_angle_wise["CT_MIP"].str.replace("SUV_MIP", "CT_MIP")
    df_angle_wise["CT_bone"] = df_angle_wise["CT_bone"].str.replace("SUV_bone", "CT_bone")
    df_angle_wise["CT_lean"] = df_angle_wise["CT_lean"].str.replace("SUV_lean", "CT_lean")
    df_angle_wise["CT_adipose"] = df_angle_wise["CT_adipose"].str.replace("SUV_adipose", "CT_adipose")
    df_angle_wise["CT_air"] = df_angle_wise["CT_air"].str.replace("SUV_air", "CT_air")

    df_angle_wise = df_angle_wise[["pat_ID", "scan_date", "SUV_MIP", "CT_MIP", "SUV_bone", "CT_bone", "SUV_lean", "CT_lean", "SUV_adipose", "CT_adipose", "SUV_air", "CT_air", "angle", "diagnosis", "age", "sex", "MTV (ml)"]]

    df_angle_wise.to_csv("/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Clinical parameter estimation/Data_Pre-processing/Output/df_rot_mips.csv")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done")