import numpy as np
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensity,
    RandSpatialCropSamplesd,
    ScaleIntensityd,
    DivisiblePadd
)
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset, pad_list_data_collate

#import sys
#sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/2D_UNET_rotating_MIPs')
#from utils import get_df

def prepare_data(args, df_train, df_val):
	#df_train = get_df(args.data_path_train)
	SUV_MIP_train = sorted(df_train['SUV_MIP'].tolist())
	SUV_bone_train = sorted(df_train['SUV_bone'].tolist())
	SUV_lean_train = sorted(df_train['SUV_lean'].tolist())
	SUV_adipose_train = sorted(df_train['SUV_adipose'].tolist())
	SUV_air_train = sorted(df_train['SUV_air'].tolist())
	SEG_train = sorted(df_train['SEG'].tolist())

	#df_val = get_df(args.data_path_val)
	SUV_MIP_val = sorted(df_val['SUV_MIP'].tolist())
	SUV_bone_val = sorted(df_val['SUV_bone'].tolist())
	SUV_lean_val = sorted(df_val['SUV_lean'].tolist())
	SUV_adipose_val = sorted(df_val['SUV_adipose'].tolist())
	SUV_air_val = sorted(df_val['SUV_air'].tolist())
	SEG_val = sorted(df_val['SEG'].tolist())

	train_files = [
		{"SUV_MIP": SUV_name, "SUV_bone": SUV_bone_name, "SUV_lean": SUV_lean_name, "SUV_adipose": SUV_adipose_name, "SUV_air": SUV_air_name, "SEG": SEG_name}
		for SUV_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, SEG_name in zip(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, SEG_train)
	]
	val_files = [
		{"SUV_MIP": SUV_name, "SUV_bone": SUV_bone_name, "SUV_lean": SUV_lean_name, "SUV_adipose": SUV_adipose_name, "SUV_air": SUV_air_name, "SEG": SEG_name}
		for SUV_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, SEG_name in zip(SUV_MIP_val, SUV_bone_val, SUV_lean_val, SUV_adipose_val, SUV_air_val, SEG_val)
	]

	# define transforms for image and segmentation
	train_transforms = Compose(
	    [

	        LoadImaged(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"]),
	        AddChanneld(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"]),

	        #DivisiblePadd(keys=["SUVmax", "SUVmean", "SUVstd", "SEG"], k=128, allow_missing_keys=False),
	        DivisiblePadd(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], k=16, allow_missing_keys=False),

	        #RandSpatialCropSamplesd(keys=["SUV", "CT", "SEG"], roi_size=[160, 160, 160], num_samples=2, random_size=False),

	        #Augmentations
	        #RandFlipd(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], prob=0.5),
	        #RandAffined(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], prob=0.5, translate_range=10),
	        #RandRotate90d(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], prob=0.5),
	        #RandGaussianNoised(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], prob=0.2),
	        #RandShiftIntensityd(keys=["SUV"], offsets=(0,30), prob=0.2),

	        ConcatItemsd(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        ToTensord(keys=["PET_CT", "SEG"]),
	        #ToTensord(keys=["SUV_bone", "SEG"]),
	    ]
	)
	val_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"]),
	        AddChanneld(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"]),

	        #DivisiblePadd(keys=["SUVmax", "SUVmean", "SUVstd", "SEG"], k=128, allow_missing_keys=False),
	        DivisiblePadd(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "SEG"], k=16, allow_missing_keys=False),

	        ConcatItemsd(keys=["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        ToTensord(keys=["PET_CT", "SEG"]),
	        #ToTensord(keys=["SUV_bone", "SEG"]),
	    ]
	)

	train_dset = Dataset(data=train_files, transform=train_transforms)
	train_loader = DataLoader(train_dset, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn = list_data_collate)
	#train_loader = DataLoader(train_dset, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn = pad_list_data_collate)

	val_dset = Dataset(data=val_files, transform=val_transforms)
	val_loader = DataLoader(val_dset, batch_size=args.batch_size_val, num_workers=args.num_workers, collate_fn = list_data_collate)
	#val_loader = DataLoader(val_dset, batch_size=args.batch_size_val, num_workers=args.num_workers, collate_fn = pad_list_data_collate)
	return train_loader, train_files, val_loader, val_files
