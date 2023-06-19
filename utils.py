import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from config import parse_args
import cv2
from PIL import Image

def read_nii(path):
    img = sitk.ReadImage(path)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = np.transpose(img_arr, (2,1,0))
    return img_arr

def generate_binary_masks(CT_arr, args):
    """
    Takes the CT image as input and generates the following binary masks based on its HU cut-off values:
    Bone, Lean Tissue, Adipose Tissue, Air
    """
    #bone_mask = np.where(CT_arr > 200, 1, 0)
    bone_mask = np.where(CT_arr > args.bone_HU[0], 1, 0)

    #lean_mask = np.where(CT_arr < -29, 0, CT_arr)
    #lean_mask = np.where(lean_mask > 150, 0, lean_mask)
    lean_mask = np.where(CT_arr < args.lean_HU[0], 0, CT_arr)
    lean_mask = np.where(lean_mask > args.lean_HU[1], 0, lean_mask)
    lean_mask = np.where(lean_mask == 0, 0, 1)

    #adipose_mask = np.where(CT_arr < -190, 0, CT_arr)
    #adipose_mask = np.where(adipose_mask > -30, 0, adipose_mask)
    adipose_mask = np.where(CT_arr < args.adipose_HU[0], 0, CT_arr)
    adipose_mask = np.where(adipose_mask > args.adipose_HU[1], 0, adipose_mask)
    adipose_mask = np.where(adipose_mask == 0, 0, 1)

    #air_mask = np.where(CT_arr < -191, 1, 0)
    air_mask = np.where(CT_arr < args.air_HU[0], 1, 0)
    return bone_mask, lean_mask, adipose_mask, air_mask

def get_channels(arr, mask):
    arr_new = arr*mask
    return arr_new

def generate_HU_channels(arr, bone_mask, lean_mask, adipose_mask, air_mask):
    arr_B = get_channels(arr, bone_mask)
    arr_LT = get_channels(arr, lean_mask)
    arr_AT = get_channels(arr, adipose_mask)
    arr_A = get_channels(arr, air_mask)
    return arr_B, arr_LT, arr_AT, arr_A

def save_npy_nii(ref_path, arr, save_path):
    image = nib.load(ref_path)
    new_image = nib.Nifti1Image(arr, image.affine)
    nib.save(new_image, save_path)

def save_all_nii(path_ref, save_path, arr_B, arr_LT, arr_AT, arr_A, prefix):
    save_npy_nii(path_ref, arr_B, os.path.join(save_path, str(prefix) + "_bone.nii.gz"))
    save_npy_nii(path_ref, arr_LT, os.path.join(save_path, str(prefix) + "_lean_tissue.nii.gz"))
    save_npy_nii(path_ref, arr_AT, os.path.join(save_path, str(prefix) + "_adipose_tissue.nii.gz"))
    save_npy_nii(path_ref, arr_A, os.path.join(save_path, str(prefix) + "_air.nii.gz"))

def preprocess_CT_HU_values(arr):
	return arr - np.min(arr)

def save_MIP(save_path, Data, factor=1.):
    """
    Save the Image using PIL.
    
    save_path - Absolute Path.
    Data - (2D Image) Pixel value should lie between (0,1).
    """
    Data = Data[:,85:-85]
    MIP_img = (255. * Data).astype(np.uint8)
    MIP_img = np.asarray(Image.fromarray(MIP_img).convert('RGB'))
    im = (factor * MIP_img).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')
    im.save(save_path)

def generate_MIPs_PET(Data, type_MIP, intensity_type, img_type):
    """
    Generate MIPs for PET Data.
    
    Data - PET Data.
    MIP - Maximum/Mean Intensity Projection.
    """
    #PET = np.clip(Data, 0, 3600)# Enhance the contrast of the soft tissue.
    PET = np.clip(Data, 0, 14)
    #PET = Data
    if type_MIP == "coronal":
        if intensity_type == "max":
            MIP_PET = np.max(PET, axis=1).astype("float") # (267, 512).
        elif intensity_type == "sum":
            MIP_PET = np.sum(PET, axis=1).astype("float") # (267, 512).
    elif type_MIP == "saggital":
        if intensity_type == "max":
            MIP_PET = np.max(PET, axis=0).astype("float") # (267, 512).
        elif intensity_type == "sum":
            MIP_PET = np.sum(PET, axis=0).astype("float") # (267, 512).
    MIP_PET = MIP_PET/np.max(MIP_PET)# Pixel Normalization, value ranges b/w (0,1).
    if img_type == "negative":
        MIP_PET = np.absolute(MIP_PET - np.amax(MIP_PET))
    MIP_PET = cv2.rotate(MIP_PET, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if type_MIP == "saggital":
        MIP_PET = np.flip(MIP_PET, 1)
    return MIP_PET


def generate_MIPs_CT(Data, type_MIP, intensity_type, img_type):
    """
    Generate MIPs for PET Data.
    
    Data - PET Data.
    MIP - Maximum/Mean Intensity Projection.
    """
    #PET = np.clip(Data, 0, 3600)# Enhance the contrast of the soft tissue.
    #PET = np.clip(Data, np.min(Data) + np.min(Data)/2, np.max(Data) - np.max(Data)/2)
    PET = Data
    if type_MIP == "coronal":
        if intensity_type == "max":
            MIP_PET = np.max(PET, axis=1).astype("float") # (267, 512).
        elif intensity_type == "sum":
            MIP_PET = np.sum(PET, axis=1).astype("float") # (267, 512).
    elif type_MIP == "saggital":
        if intensity_type == "max":
            MIP_PET = np.max(PET, axis=0).astype("float") # (267, 512).
        elif intensity_type == "sum":
            MIP_PET = np.sum(PET, axis=0).astype("float") # (267, 512).

    MIP_PET = MIP_PET/np.max(MIP_PET)# Pixel Normalization, value ranges b/w (0,1).
    if img_type == "negative":
        MIP_PET = np.absolute(MIP_PET - np.amax(MIP_PET))
    MIP_PET = cv2.rotate(MIP_PET, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if type_MIP == "saggital":
        MIP_PET = np.flip(MIP_PET, 1)
    return MIP_PET


def generate_MIPs_Seg(Data, type_MIP):
    """
    Generate MIPs for Segmentation Data.
    
    Data - Segmentation Mask.
    """
    if type_MIP == "coronal":
        MIP_Seg = np.max(Data, axis=1).astype("float")
    elif type_MIP == "saggital":
        MIP_Seg = np.max(Data, axis=0).astype("float")
    #MIP_Seg = MIP_Seg/np.max(MIP_Seg)
    MIP_Seg = cv2.rotate(MIP_Seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if type_MIP == "saggital":
        MIP_Seg = np.flip(MIP_Seg, 1)
    return MIP_Seg

def overlap(MIP_img, MIP_Seg):
    """
    Overlay the MIP_img and MIP_Seg into one.
    """
    temp = np.zeros((MIP_img.shape[0],MIP_img.shape[1],MIP_img.shape[2]))
    mul = np.where(MIP_Seg == 0, 1, 0)
    add = np.where(MIP_Seg == 0, 0, 255)
    for i in range(MIP_img.shape[2]):
        temp[:,:,i] = MIP_img[:,:,i] * mul
    temp[:,:,0] = temp[:,:,0] + add
    return temp

def get_contours(img):
    gx, gy = np.gradient(img)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    temp_edge = np.asarray(temp_edge, dtype=np.uint8)
    return temp_edge

def create_collage(path_c, path_s, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for MIP in os.listdir(path_c):
        image1 = Image.open(os.path.join(path_c, MIP))
        image2 = Image.open(os.path.join(path_s, MIP))
        collage = Image.new('RGB', (image1.size[0]*2, image1.size[1]))
        # Paste the first image onto the collage
        collage.paste(image1, (0, 0))
        # Paste the second image onto the collage
        collage.paste(image2, (image1.size[0], 0))

        # Save the collage
        collage.save(os.path.join(save_path, MIP))

def create_final_collage(path, save_path, pat_ID):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image1 = Image.open(os.path.join(path, "MIP_SUV.jpg"))
    image2 = Image.open(os.path.join(path, "MIP_SUV_bone.jpg"))
    image3 = Image.open(os.path.join(path, "MIP_SUV_lean.jpg"))
    image4 = Image.open(os.path.join(path, "MIP_SUV_adipose.jpg"))
    image5 = Image.open(os.path.join(path, "MIP_SUV_air.jpg"))
    image6 = Image.open(os.path.join(path, "MIP_SUV_SEG.jpg"))
    image7 = Image.open(os.path.join(path, "SIP_SUV_SEG.jpg"))

    image8 = Image.open(os.path.join(path, "SIP_CT.jpg"))
    image9 = Image.open(os.path.join(path, "SIP_CT_bone.jpg"))
    image10 = Image.open(os.path.join(path, "SIP_CT_lean.jpg"))
    image11 = Image.open(os.path.join(path, "SIP_CT_adipose.jpg"))
    image12 = Image.open(os.path.join(path, "SIP_CT_air.jpg"))
    image13 = Image.open(os.path.join(path, "MIP_CT_SEG.jpg"))
    image14 = Image.open(os.path.join(path, "SIP_CT_SEG.jpg"))

    collage = Image.new('RGB', (image1.size[0]*7, image1.size[1]*2))

    # Paste SUV
    collage.paste(image1, (0, 0))
    collage.paste(image2, (image1.size[0]*1, 0))
    collage.paste(image3, (image1.size[0]*2, 0))
    collage.paste(image4, (image1.size[0]*3, 0))
    collage.paste(image5, (image1.size[0]*4, 0))
    collage.paste(image6, (image1.size[0]*5, 0))
    collage.paste(image7, (image1.size[0]*6, 0))

    # # Paste CT
    collage.paste(image8, (0, image1.size[1]))
    collage.paste(image9, (image1.size[0]*1, image1.size[1]))
    collage.paste(image10, (image1.size[0]*2, image1.size[1]))
    collage.paste(image11, (image1.size[0]*3, image1.size[1]))
    collage.paste(image12, (image1.size[0]*4, image1.size[1]))
    collage.paste(image13, (image1.size[0]*5, image1.size[1]))
    collage.paste(image14, (image1.size[0]*6, image1.size[1]))

    # Save the collage
    collage.save(os.path.join(save_path, pat_ID + ".jpg"))


def generate_SUV_CT_collage(args, SEG_arr, SUV_arr, SUV_arr_B, SUV_arr_LT, SUV_arr_AT, SUV_arr_A, CT_arr, CT_arr_B, CT_arr_LT, CT_arr_AT, CT_arr_A, save_path, pat_ID, scan_date, disease_type):
    """
    B - Bone; LT - Lean Tissue; AT - Adipose Tissue; A - Air; L - Lesion
    """
    #save_path_MIP = os.path.join(save_path, "Visualization", "MIPs", pat_ID + "_" + scan_date)
    save_path_MIP = os.path.join(save_path, "MIPs", pat_ID + "_" + scan_date)
    if not os.path.exists(save_path_MIP):
        os.makedirs(save_path_MIP)

    SUV_arr_L = SUV_arr*SEG_arr
    CT_arr_L = CT_arr*SEG_arr
    CT_arr_LT, CT_arr_AT, CT_arr_A, CT_arr_L = preprocess_CT_HU_values(CT_arr_LT), preprocess_CT_HU_values(CT_arr_AT), preprocess_CT_HU_values(CT_arr_A), preprocess_CT_HU_values(CT_arr_L)

    #MIP_types = ["coronal", "saggital"]
    MIP_types = args.MIP_types

    for i in MIP_types:
        save_path_MIP_temp = os.path.join(save_path_MIP, i)
        if not os.path.exists(save_path_MIP_temp):
            os.makedirs(save_path_MIP_temp)

        P_SUV_bone = generate_MIPs_PET(SUV_arr_B, i, intensity_type="max", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV_bone.jpg"), P_SUV_bone, factor=1.)
        P_CT_bone = generate_MIPs_CT(CT_arr_B, i, intensity_type="sum", img_type="negative")
        P_CT_bone = (P_CT_bone - np.min(P_CT_bone)) / (np.max(P_CT_bone) - np.min(P_CT_bone))
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT_bone.jpg"), P_CT_bone, factor=1.)

        P_SUV_lean = generate_MIPs_PET(SUV_arr_LT, i, intensity_type="max", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV_lean.jpg"), P_SUV_lean, factor=1.)
        P_CT_lean = generate_MIPs_CT(CT_arr_LT, i, intensity_type="sum", img_type="negative")
        P_CT_lean = (P_CT_lean - np.min(P_CT_lean)) / (np.max(P_CT_lean) - np.min(P_CT_lean))
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT_lean.jpg"), P_CT_lean, factor=1.)

        P_SUV_adipose = generate_MIPs_PET(SUV_arr_AT, i, intensity_type="max", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV_adipose.jpg"), P_SUV_adipose, factor=1.)
        P_CT_adipose = generate_MIPs_CT(CT_arr_AT, i, intensity_type="sum", img_type="positive")
        P_CT_adipose = (P_CT_adipose - np.min(P_CT_adipose)) / (np.max(P_CT_adipose) - np.min(P_CT_adipose))
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT_adipose.jpg"), P_CT_adipose, factor=1.)
        
        P_SUV_air = generate_MIPs_PET(SUV_arr_A, i, intensity_type="max", img_type="negative")
        #P_SUV_air = (P_SUV_air - np.min(P_SUV_air)) / (np.max(P_SUV_air) - np.min(P_SUV_air))
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV_air.jpg"), P_SUV_air, factor=1.)
        P_CT_air = generate_MIPs_CT(CT_arr_A, i, intensity_type="sum", img_type="negative")
        P_CT_air = (P_CT_air - np.min(P_CT_air)) / (np.max(P_CT_air) - np.min(P_CT_air))
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT_air.jpg"), P_CT_air, factor=1.)

        MIP_SUV = generate_MIPs_PET(SUV_arr, i, intensity_type="max", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV.jpg"), MIP_SUV, factor=1.)
        MIP_SUV_SEG = generate_MIPs_PET(SUV_arr_L, i, intensity_type="max", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_SUV_SEG.jpg"), MIP_SUV_SEG, factor=1.)

        MIP_SUV_SEG = generate_MIPs_PET(SUV_arr_L, i, intensity_type="sum", img_type="negative")
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_SUV_SEG.jpg"), MIP_SUV_SEG, factor=1.)

        CT_arr_new = CT_arr - np.min(CT_arr)
        MIP_CT = generate_MIPs_CT(CT_arr_new, i, intensity_type="sum", img_type="positive")
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT.jpg"), MIP_CT, factor=1.)

        MIP_CT_SEG = generate_MIPs_CT(CT_arr_L, i, intensity_type="sum", img_type="negative")
        MIP_SEG = generate_MIPs_Seg(SEG_arr, i)
        MIP_CT_SEG = (MIP_CT_SEG - np.min(MIP_CT_SEG)) / (np.max(MIP_CT_SEG) - np.min(MIP_CT_SEG))
        MIP_CT_SEG_new = MIP_CT_SEG*MIP_SEG
        MIP_CT_SEG_new = np.absolute(MIP_CT_SEG_new - np.amax(MIP_CT_SEG_new))
        save_MIP(os.path.join(save_path_MIP_temp, "SIP_CT_SEG.jpg"), MIP_CT_SEG_new, factor=1.)

        MIP_CT_SEG = generate_MIPs_CT(CT_arr_L, i, intensity_type="max", img_type="negative")
        MIP_SEG = generate_MIPs_Seg(SEG_arr, i)
        MIP_CT_SEG = (MIP_CT_SEG - np.min(MIP_CT_SEG)) / (np.max(MIP_CT_SEG) - np.min(MIP_CT_SEG))
        MIP_CT_SEG_new = MIP_CT_SEG*MIP_SEG
        MIP_CT_SEG_new = np.absolute(MIP_CT_SEG_new - np.amax(MIP_CT_SEG_new))
        save_MIP(os.path.join(save_path_MIP_temp, "MIP_CT_SEG.jpg"), MIP_CT_SEG_new, factor=1.)

    path_collages = os.path.join(save_path_MIP, "collages")
    path_c = os.path.join(save_path_MIP, "coronal")
    path_s = os.path.join(save_path_MIP, "saggital")
    create_collage(path_c, path_s, path_collages)
    create_final_collage(path_collages, os.path.join(save_path, "Collages"), disease_type + "_" + pat_ID + "_" + scan_date)

def preprocess_df(df, args):
    #Bone HU window
    df["SUV_bone"] = df["SUV"]
    #df["SUV_bone"] = df["SUV_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["SUV_bone"] = df["SUV_bone"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["SUV_bone"] = df["SUV_bone"].str.replace("SUV.nii.gz", "SUV_bone.nii.gz")

    df["CT_bone"] = df["CT"]
    #df["CT_bone"] = df["CT_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["CT_bone"] = df["CT_bone"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["CT_bone"] = df["CT_bone"].str.replace("CTres.nii.gz", "CT_bone.nii.gz")

    #Lean HU window
    df["SUV_lean"] = df["SUV"]
    #df["SUV_lean"] = df["SUV_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["SUV_lean"] = df["SUV_lean"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["SUV_lean"] = df["SUV_lean"].str.replace("SUV.nii.gz", "SUV_lean_tissue.nii.gz")

    df["CT_lean"] = df["CT"]
    #df["CT_lean"] = df["CT_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["CT_lean"] = df["CT_lean"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["CT_lean"] = df["CT_lean"].str.replace("CTres.nii.gz", "CT_lean_tissue.nii.gz")

    #Adipose HU window
    df["SUV_adipose"] = df["SUV"]
    #df["SUV_adipose"] = df["SUV_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["SUV_adipose"] = df["SUV_adipose"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["SUV_adipose"] = df["SUV_adipose"].str.replace("SUV.nii.gz", "SUV_adipose_tissue.nii.gz")

    df["CT_adipose"] = df["CT"]
    #df["CT_adipose"] = df["CT_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["CT_adipose"] = df["CT_adipose"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["CT_adipose"] = df["CT_adipose"].str.replace("CTres.nii.gz", "CT_adipose_tissue.nii.gz")

    #Air HU window
    df["SUV_air"] = df["SUV"]
    #df["SUV_air"] = df["SUV_air"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["SUV_air"] = df["SUV_air"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["SUV_air"] = df["SUV_air"].str.replace("SUV.nii.gz", "SUV_air.nii.gz")

    df["CT_air"] = df["CT"]
    #df["CT_air"] = df["CT_air"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data")
    df["CT_air"] = df["CT_air"].str.replace(args.data_path, args.path_multi_channel_3D_CT_SUV)
    df["CT_air"] = df["CT_air"].str.replace("CTres.nii.gz", "CT_air.nii.gz")

    return df