import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, ImageDataset
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from tqdm import tqdm
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from create_dataset import prepare_data
from net_resnet import getPretrainedResNet50

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Clinical parameter estimation')
from config import parse_args
from utils import make_dirs, train_regression, validation_regression, plot

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121

def main(args):
	#K fold cross validation
	K = args.K_fold_CV
	df = pd.read_csv(args.path_df)
	#df = df[df["diagnosis"]!="NEGATIVE"].reset_index(drop=True)

	df_rot_mips_collages = pd.read_csv(args.path_df_rot_mips)
	include_angles = [90]
	df_rot_mips_collages = df_rot_mips_collages[df_rot_mips_collages.angle.isin(include_angles)].reset_index(drop=True)

	df_rot_mips_collages["SUV_MIP"] = df_rot_mips_collages["SUV_MIP"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["SUV_bone"] = df_rot_mips_collages["SUV_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["SUV_lean"] = df_rot_mips_collages["SUV_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["SUV_adipose"] = df_rot_mips_collages["SUV_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["SUV_air"] = df_rot_mips_collages["SUV_air"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")

	df_rot_mips_collages["CT_MIP"] = df_rot_mips_collages["CT_MIP"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["CT_bone"] = df_rot_mips_collages["CT_bone"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["CT_lean"] = df_rot_mips_collages["CT_lean"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["CT_adipose"] = df_rot_mips_collages["CT_adipose"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")
	df_rot_mips_collages["CT_air"] = df_rot_mips_collages["CT_air"].str.replace("/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_512_512", "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/Multi_directional_2D_MIPs_collages_[0, 90, -45, +45]_512_512")

	path_Output = args.path_CV_Output
	outcome = args.regression_type[0]

	for k in tqdm(range(K)):
		if k >= 0:
			print("Cross Valdation for fold: {}".format(k))
			max_epochs = args.max_epochs
			val_interval = args.validation_interval
			best_metric = args.best_metric_regression
			best_metric_epoch = args.best_metric_epoch
			metric_values = []
			metric_values_r_squared = []
			print("Network Initialization")
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			
			model = DenseNet121(spatial_dims=args.dimensions, in_channels=args.in_channels,
								out_channels=args.num_classes_age, dropout_prob=args.dropout).to(device)
			#model = WideResNetRegression(widen_factor=6, depth=20, num_channels=10, dropout_rate=0.25).to(device)

			#model = torch.nn.DataParallel(model).to(device)
			#model = ModifiedDenseNet121(in_channels=10, out_units=1, dropout_rate=0.25).to(device)
			#model = AgeResNet(args.num_classes_age).to(device)
			
			#Load the checkpoint from where the training broke
			if args.pre_trained_weights:
				print("Checkpoint Loading for Cross Validation: {}".format(k))
				checkpoint_path = load_checkpoint(args, k)
				checkpoint = torch.load(checkpoint_path)
				model.load_state_dict(checkpoint['net'])
			else:
				print("Training from Scratch!")

			#checkpoint_path = "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Cross_Validation/Output/Regression/MTV/Experiment_9/CV_1/Network_Weights/best_model_190.pth.tar"
			#checkpoint = torch.load(checkpoint_path)
			#model.load_state_dict(checkpoint['net'])

			#loss_function = torch.nn.MSELoss()
			loss_function = torch.nn.SmoothL1Loss()
			#loss_function = torch.nn.SmoothL1Loss(reduction='mean', delta=5) #Huber loss
			#loss_function = torch.nn.L1Loss()

			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
			#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
			lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

			#Make all the relevant directories
			make_dirs(path_Output, k)
			#Dataloader Preparation
			factor = round(df.shape[0]/K)
			if k == (args.K_fold_CV - 1):
				df_val = df[factor*k:].reset_index(drop=True)
			else:
				df_val = df[factor*k:factor*k+factor].reset_index(drop=True)
			df_train = df[~df.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

			df_train_new = df_rot_mips_collages[df_rot_mips_collages.scan_date.isin(df_train.scan_date)].reset_index(drop=True)
			df_val_new = df_rot_mips_collages[df_rot_mips_collages.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

			#df_train_new = df_train_new[df_train_new["pat_ID"]=="PETCT_0011f3deaf"].reset_index(drop=True)
			#df_val_new = df_train_new

			print("Number of patients in Training set: ", len(df_train))
			print("Number of patients in Valdation set: ", len(df_val))

			#train_files, train_loader = prepare_data(args, df_train_new, shuffle=True, label="age")
			train_files, train_loader = prepare_data(args, df_train_new, args.batch_size_train, shuffle=True, label=outcome)

			train_loss = []
			for epoch in tqdm(range(max_epochs)):
				#epoch += 191
				#Training
				epoch_loss, train_loss = train_regression(model, train_loader, optimizer, loss_function, device, train_loss)
				print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")
				#path_train_loss = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_train_loss.jpg")
				#if len(train_loss) > 2:
					#plot_auc(train_loss, path_train_loss)
				#Validation
				if (epoch + 1) % val_interval == 0:
					#metric_values, best_metric_new, metric_values_r_squared = validation_regression(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, metric_values_r_squared, path_Output, outcome, loss_function)
					metric_values, best_metric_new = validation_regression(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, metric_values_r_squared, path_Output, outcome, loss_function)

					#validation(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, path_Output)
					best_metric = best_metric_new

				#Save and plot
				np.save(os.path.join(path_Output, "CV_" + str(k) + "/MAE.npy"), metric_values)
				#np.save(os.path.join(path_Output, "CV_" + str(k) + "/R2.npy"), metric_values_r_squared)

				path_MAE = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_MAE.jpg")
				#path_r_squared = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_R2.jpg")

				if len(metric_values) > 2:
					plot(metric_values, path_MAE, "MAE")
					#plot(metric_values_r_squared, path_r_squared, "R2")

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")
