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
from monai.losses import DiceLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from create_dataset import prepare_data

import Nested_UNET
#import Nested_UNET_with_regression_head_vanilla

from dynUNET import build_net
#import sys
#sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/Clean_Code/Proposed/Multi_directional_2D_MIPs')
#import UNET_architectures
from config import parse_args
from utils import prepare_df_2D_rot_MIPs, make_dirs, train, validation, plot_dice
import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.multiprocessing.set_start_method('spawn', force=True)

def main(args):
	#print("device: ", torch.cuda.device_count())
	#K fold cross validation
	K = args.K_fold_CV

	df_2D_MIPs = pd.read_csv(args.path_df_2D_MIPs)
	df_2D_MIPs = df_2D_MIPs[df_2D_MIPs["diagnosis"]=="LYMPHOMA"].reset_index(drop=True)

	df_filtered = pd.read_csv(args.path_df) #Tumor Scans
	df_filtered = df_filtered[df_filtered["diagnosis"]=="LYMPHOMA"].reset_index(drop=True)
	path_Output = args.path_CV_Output

	for k in tqdm(range(K)):
		if k == 0:
			print("Cross Valdation for fold: {}".format(k))
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
			
			model = Nested_UNET.U_net().to(device)
			#model = Nested_UNET_with_regression_head_vanilla.UNetWithRegression().to(device)

			#model = build_net().to(device)
			#model = UNET_architectures.AttU_Net().to(device)
			
			#Load the checkpoint from where the training broke
			if args.pre_trained_weights:
				print("Checkpoint Loading for Cross Validation: {}".format(k))
				checkpoint_path = load_checkpoint(args, k)
				checkpoint = torch.load(checkpoint_path)
				model.load_state_dict(checkpoint['net'])
			else:
				print("Training from Scratch!")

			#checkpoint_path = "/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/Part_2/LYMPHOMA/CV_0/Network_Weights/best_model_3.pth.tar"
			#checkpoint = torch.load(checkpoint_path)
			#model.load_state_dict(checkpoint['net'])

			#loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
			loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)

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

			df_train_new = df_2D_MIPs[df_2D_MIPs.scan_date.isin(df_train.scan_date)].reset_index(drop=True)
			df_val_new = df_2D_MIPs[df_2D_MIPs.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

			#df_train_new = df_train_new[:1].reset_index(drop=True)
			#df_val_new = df_train_new

			train_loader, train_files, val_loader, val_files = prepare_data(args, df_train_new, df_val_new)
			print("Length train_loader: {} & val_loader: {}".format(len(train_loader), len(val_loader)))

			post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
			post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

			for epoch in range(max_epochs):
				#epoch += 4
				#Training
				epoch_loss = train(model, train_loader, optimizer, loss_function, device)
				lr_scheduler.step()
				#epoch_loss_values.append(epoch_loss)
				print(f"Training epoch {epoch + 1} average loss: {epoch_loss:.4f}")
				#Validation
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
