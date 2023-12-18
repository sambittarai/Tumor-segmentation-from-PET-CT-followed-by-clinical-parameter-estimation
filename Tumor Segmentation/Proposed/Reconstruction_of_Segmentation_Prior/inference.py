import numpy as np
import torch
import os
from tqdm import tqdm
import pandas as pd
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
    ToTensord
)
from monai.networks.layers import Norm
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric, compute_meandice
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from create_dataset import prepare_data

import Nested_UNET
from utils import prepare_df_2D_rot_MIPs, load_checkpoint, make_dirs, inference_full_image, generate_TPDMs
import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET')
from config import parse_args
#from utils import prepare_df_2D_rot_MIPs, make_dirs, inference, generate_TPDMs, inference_new
import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.multiprocessing.set_start_method('spawn', force=True)

def main(args):
	#print("device: ", torch.cuda.device_count())
	#K fold cross validation Inference
	K = args.K_fold_CV
	#df_2D_MIPs = prepare_df_2D_rot_MIPs(args.path_data_MIPs, args)
	#df_2D_MIPs.to_csv("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/2D_UNET_rotating_MIPs/Data_Preparation/df.csv")
	df_2D_MIPs = pd.read_csv(args.path_df_2D_MIPs)
	df_2D_MIPs = df_2D_MIPs[df_2D_MIPs["diagnosis"]=="LYMPHOMA"].reset_index(drop=True)

	df_filtered = pd.read_csv(args.path_df) #Tumor Scans
	df_filtered = df_filtered[df_filtered["diagnosis"]=="LYMPHOMA"].reset_index(drop=True)
	path_Output = args.path_inference_Output

	for k in tqdm(range(K)):
		if k == 3:
			print("Inference for fold: {}".format(k))
			print("Network Initialization")
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			#model = UNet(
			#			dimensions=args.dimensions,
			#			in_channels=args.in_channels,
			#			out_channels=args.out_channels,
			#			channels=(16, 32, 64, 128, 256),
			#			strides=(2, 2, 2, 2),
			#			num_res_units=2,
			#			norm=Norm.BATCH,
			#			dropout=args.dropout,
			#		).to(device)
			model = Nested_UNET.U_net().to(device)
			#model = UNET_architectures.NestedUNet().to(device)
			#model = UNET_architectures.AttU_Net().to(device)
			
			#Load the checkpoint from where the training broke
			print("Checkpoint Loading for Cross Validation Fold: {}".format(k))
			checkpoint_path = load_checkpoint(args, k)
			checkpoint = torch.load(checkpoint_path)
			model.load_state_dict(checkpoint['net'])

			#Make all the relevant directories
			make_dirs(path_Output, k)
			#Dataloader Preparation
			factor = round(df_filtered.shape[0]/5)
			if k == (args.K_fold_CV - 1):
				#df_val = df_filtered[100*k:].reset_index(drop=True)
				df_val = df_filtered[factor*k:].reset_index(drop=True)
			else:
				#df_val = df_filtered[100*k:100*k+100].reset_index(drop=True)
				df_val = df_filtered[factor*k:factor*k+factor].reset_index(drop=True)
			df_train = df_filtered[~df_filtered.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

			df_train_new = df_2D_MIPs[df_2D_MIPs.scan_date.isin(df_train.scan_date)].reset_index(drop=True)
			df_val_new = df_2D_MIPs[df_2D_MIPs.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

			#train_loader, train_files, val_loader, val_files = prepare_data(args, df_train_new, df_val_new)
			val_loader, val_files = prepare_data(args, df_val_new)
			print("Length of val_loader: {}".format(len(val_loader)))

			#Inference (Generate 2D MIP predictions)
			inference_full_image(path_Output, val_loader, model, device, val_files, k, args)
			#Generate Tumor Probability Distribution Maps (TPDMs)
			path_TPDMs = args.path_TPDMs
			if not os.path.exists(path_TPDMs):
				os.makedirs(path_TPDMs)
			path_predictions = os.path.join(path_Output, "CV_" + str(k), "Predictions")
			generate_TPDMs(path_TPDMs, path_predictions, df_val)


if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")
