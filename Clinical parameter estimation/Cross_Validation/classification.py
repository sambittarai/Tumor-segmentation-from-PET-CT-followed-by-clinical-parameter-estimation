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
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction')
from config import parse_args
from utils import make_dirs, train_classification, validation_classification, plot_auc

def main(args):
	#K fold cross validation
	K = args.K_fold_CV
	df = pd.read_csv(args.path_df)
	df_rot_mips = pd.read_csv(args.path_df_rot_mips)

	df_rot_mips_collages = pd.read_csv(args.path_df_rot_mips_collages)

	path_Output = args.path_CV_Output
	outcome = args.classification_type[0]
	#if outcome == "diagnosis":
	#	df['diagnosis'] = df['diagnosis'].replace(['MELANOMA', 'LUNG_CANCER', 'LYMPHOMA'], 1)
	#	df['diagnosis'] = df['diagnosis'].replace('NEGATIVE', 0)		

	for k in tqdm(range(K)):
		if k >= 0:
			print("Cross Valdation for fold: {}".format(k))
			max_epochs = args.max_epochs
			val_interval = args.validation_interval
			best_metric = args.best_metric_classification
			best_metric_epoch = args.best_metric_epoch
			metric_values = []
			print("Network Initialization")
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			model = DenseNet121(spatial_dims=args.dimensions, in_channels=args.in_channels,
								out_channels=args.num_classes_sex, init_features=64, dropout_prob=0.25).to(device)
			#model = torch.nn.DataParallel(model).to(device)
			#model = getPretrainedResNet50(class_count=args.num_classes, channel_count=args.in_channels, dim=1).to(device)
			
			#Load the checkpoint from where the training broke
			if args.pre_trained_weights:
				print("Checkpoint Loading for Cross Validation: {}".format(k))
				checkpoint_path = load_checkpoint(args, k)
				checkpoint = torch.load(checkpoint_path)
				model.load_state_dict(checkpoint['net'])
			else:
				print("Training from Scratch!")

			#checkpoint_path = "/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Cross_Validation/Output/Classification/Disease_Type/Binary/CV_0/Network_Weights/best_model_31.pth.tar"
			#checkpoint = torch.load(checkpoint_path)
			#model.load_state_dict(checkpoint['net'])

			#loss_function = torch.nn.CrossEntropyLoss()
			#loss_function = torch.nn.NLLLoss()
			#loss_function = torch.nn.BCEWithLogitsLoss()

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

			print("Number of patients in Training set: ", len(df_train))
			print("Number of patients in Valdation set: ", len(df_val))

			class_freq = np.unique(df_train_new["sex"], return_counts=True)[1]
			class_weights = torch.tensor([float(class_freq[0]/np.sum(class_freq)), float(class_freq[1]/np.sum(class_freq))]).to(device)
			#class_weights = torch.tensor([1.0, 2.0]).to(device)
			loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

			train_files, train_loader = prepare_data(args, df_train_new, args.batch_size_train, shuffle=True, label=outcome)

			train_loss = []
			for epoch in tqdm(range(max_epochs)):
				#epoch += 32
				#Training
				epoch_loss, train_loss = train_classification(model, train_loader, optimizer, loss_function, device, train_loss, outcome)
				print(f"Training epoch {epoch} average loss: {epoch_loss:.4f}")
				#path_train_loss = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_train_loss.jpg")
				#if len(train_loss) > 2:
					#plot_auc(train_loss, path_train_loss)
				#Validation
				if (epoch + 1) % val_interval == 0:
					metric_values, best_metric_new = validation_classification(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, path_Output, outcome)
					#validation(args, k, epoch, optimizer, model, df_val_new, device, best_metric, metric_values, path_Output)
					best_metric = best_metric_new

				#Save and plot DICE
				np.save(os.path.join(path_Output, "CV_" + str(k) + "/AUC.npy"), metric_values)
				path_dice = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_auc.jpg")
				if len(metric_values) > 2:
					plot_auc(metric_values, path_dice)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print("Done")
