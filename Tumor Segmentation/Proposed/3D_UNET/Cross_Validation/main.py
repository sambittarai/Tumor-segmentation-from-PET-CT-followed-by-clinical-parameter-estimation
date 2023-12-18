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
from monai.networks.nets import UNet, SegResNet
from monai.losses import DiceLoss, DiceFocalLoss, TverskyLoss
from monai.inferers import sliding_window_inference

from create_dataset import prepare_data
#from create_dataset_multi_channel_inputs import prepare_data

from dynUNET import build_net

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET')
from config import parse_args
from utils import get_stratified_df_split, DICE_Score, train, validation, get_df, filter_tumor_scans, make_dirs, plot_dice, load_checkpoint
import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.multiprocessing.set_start_method('spawn', force=True)
from torch.nn.parallel import DistributedDataParallel

def main(args):
	#print("device: ", torch.cuda.device_count())
	#K fold cross validation
	K = args.K_fold_CV
	df_filtered = pd.read_csv(args.path_df_disease_wise)
	df_filtered = df_filtered[df_filtered["diagnosis"]=="LYMPHOMA"].reset_index(drop=True)
	#print(df_filtered.shape)

	##TPDMs generated using (SUV_MIP, SUV_bone, SUV_lean, SUV_adipose, SUV_air) as inputs to the 2D UNET++ network
	#df_filtered["TPDM"] = df_filtered["SUV"]
	#df_filtered["TPDM"] = df_filtered["TPDM"].str.replace("SUV.nii.gz", "TPDM.nii.gz").reset_index(drop=True)
	#df_filtered["TPDM"] = df_filtered["TPDM"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/home/sambit/Move_files_SSD/TPDM_SUV/TPDM_SUV_MIP_bone_lean_adipose_air").reset_index(drop=True)

	#TPDMs generated using only the SUV_MIP as input to the 2D UNET++ network
	df_filtered["TPDM"] = df_filtered["SUV"]
	df_filtered["TPDM"] = df_filtered["TPDM"].str.replace("SUV.nii.gz", "TPDM.nii.gz").reset_index(drop=True)
	df_filtered["TPDM"] = df_filtered["TPDM"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/home/sambit/Move_files_SSD/TPDM_SUV/TPDM_SUV_MIP").reset_index(drop=True)
	

	#df_filtered["SUV_bone"] = df_filtered["SUV"]
	#df_filtered["SUV_bone"] = df_filtered["SUV_bone"].str.replace("SUV.nii.gz", "SUV_bone.nii.gz").reset_index(drop=True)
	#df_filtered["SUV_bone"] = df_filtered["SUV_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data").reset_index(drop=True)

	#df_filtered["SUV_lean"] = df_filtered["SUV"]
	#df_filtered["SUV_lean"] = df_filtered["SUV_lean"].str.replace("SUV.nii.gz", "SUV_lean_tissue.nii.gz").reset_index(drop=True)
	#df_filtered["SUV_lean"] = df_filtered["SUV_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data").reset_index(drop=True)

	#df_filtered["SUV_adipose"] = df_filtered["SUV"]
	#df_filtered["SUV_adipose"] = df_filtered["SUV_adipose"].str.replace("SUV.nii.gz", "SUV_adipose_tissue.nii.gz").reset_index(drop=True)
	#df_filtered["SUV_adipose"] = df_filtered["SUV_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data").reset_index(drop=True)

	#df_filtered["SUV_air"] = df_filtered["SUV"]
	#df_filtered["SUV_air"] = df_filtered["SUV_air"].str.replace("SUV.nii.gz", "SUV_air.nii.gz").reset_index(drop=True)
	#df_filtered["SUV_air"] = df_filtered["SUV_air"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Data_Preparation/Output/3D_CT_SUV_Data").reset_index(drop=True)

	#df_filtered["CT"] = df_filtered["CT"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/home/sambit/Move_Files_SSD/Data").reset_index(drop=True)
	#df_filtered["SUV"] = df_filtered["SUV"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/home/sambit/Move_Files_SSD/Data").reset_index(drop=True)
	#df_filtered["SEG"] = df_filtered["SEG"].str.replace("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", "/home/sambit/Move_Files_SSD/Data").reset_index(drop=True)

	path_Output = args.path_CV_Output

	for k in tqdm(range(K)):
		if k == 0:
			print("Cross Validation for fold: {}".format(k))
			#print("SUV_MIP only; LUNG_CANCER")
			max_epochs = args.max_epochs
			#epoch_loss_values = []
			val_interval = args.validation_interval
			best_metric = args.best_metric
			best_metric_epoch = args.best_metric_epoch
			metric_values = []
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
			#model = SegResNet(
			#			spatial_dims=args.dimensions,
			#			blocks_down=[1, 2, 2, 4], #[1, 2, 2, 4, 4, 4]
			#			blocks_up=[1, 1, 1],
			#			init_filters=16,
			#			in_channels=3,
			#			out_channels=2,
			#			dropout_prob=0.2,
			#		).to(device)
			model = build_net().to(device)
			#model = DistributedDataParallel(module=model, device_ids=[device])
			#model = torch.nn.DataParallel(model).to(device)
			
			#Load the checkpoint from where the training broke
			if args.pre_trained_weights:
				print("Checkpoint Loading for Cross Validation: {}".format(k))
				checkpoint_path = load_checkpoint(args, k)
				checkpoint = torch.load(checkpoint_path)
				model.load_state_dict(checkpoint['net'])
			else:
				print("Training from Scratch!")

			checkpoint_path = "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_0/Network_Weights/best_model_78.pth.tar"
			checkpoint = torch.load(checkpoint_path)
			model.load_state_dict(checkpoint['net'])

			#DICE Loss
			loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)

			#Tversky Loss
			#loss_function = TverskyLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)

			#Generalized DICE Focal Loss
			#loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)


			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
			#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
			lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
			dice_metric = DiceMetric(include_background=False, reduction="mean")

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

			#remove_ids = ["PETCT_4011ffe8ae", "PETCT_0223010e46", "PETCT_108c1763d4", "PETCT_1f6b6b0548"]
			remove_ids = []
			df_train = df_train[~df_train.pat_ID.isin(remove_ids)].reset_index(drop=True)
			#df_train = df_train[:4]
			df_val = df_val[~df_val.pat_ID.isin(remove_ids)].reset_index(drop=True)

			#df_train, df_val = get_stratified_df_split(df_filtered, k)
			train_loader, train_files, val_loader, val_files = prepare_data(args, df_train, df_val)
			print("Length of TrainLoader: {} & ValidationLoader: {}".format(len(train_loader), len(val_loader)))

			post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
			post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

			for epoch in range(max_epochs):
				epoch += 79
				#Training
				epoch_loss = train(model, train_loader, train_files, optimizer, loss_function, device)
				lr_scheduler.step()
				#epoch_loss_values.append(epoch_loss)
				print(f"Training epoch {epoch + 1} average loss: {epoch_loss:.4f}")
				#Validation
				if epoch >= args.validation_start:
					if (epoch + 1) % val_interval == 0:
						metric_values, best_metric_new = validation(epoch, optimizer, post_pred, post_label, model, val_loader, device, args, dice_metric, metric_values, best_metric, k, val_files, path_Output)
						best_metric = best_metric_new

					#Save and plot DICE
					np.save(os.path.join(path_Output, "CV_" + str(k) + "/DICE.npy"), metric_values)
					path_dice = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_dice.jpg")
					if len(metric_values) > 2:
						plot_dice(metric_values, path_dice, args.validation_interval)
			

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")
