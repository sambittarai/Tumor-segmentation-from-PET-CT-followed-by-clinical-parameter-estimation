import numpy as np
import pandas as pd
import os
import torch
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
from tqdm import tqdm
import cc3d
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import nibabel as nib
import torch
import numpy as np
import scipy.ndimage

def load_checkpoint(args, k):
	if k == 0:
		checkpoint_path = args.path_checkpoint_CV_0
	elif k == 1:
		checkpoint_path = args.path_checkpoint_CV_1
	elif k == 2:
		checkpoint_path = args.path_checkpoint_CV_2
	elif k == 3:
		checkpoint_path = args.path_checkpoint_CV_3
	elif k == 4:
		checkpoint_path = args.path_checkpoint_CV_4
	return checkpoint_path

def convert_npy_to_nii(GT, prediction, pat_id, scan_date, files, args):
	FN = GT - prediction
	FN = np.where(FN == -1, 0, FN)
	path = os.path.join(args.path_FN_Data, pat_id, scan_date)
	if len(np.unique(FN)) == 2:
		SUV = nib.load(files['SUV'])
		FN_img = nib.Nifti1Image(FN, SUV.affine)
		if not os.path.exists(path):
			os.makedirs(path)
		nib.save(FN_img, os.path.join(path, "SEG.nii.gz"))

	else:
		#Don't generate data for this scan
		print(pat_id, scan_date)

def read_nii(path):
	img = sitk.ReadImage(path)
	img_arr = sitk.GetArrayFromImage(img)
	img_arr = np.transpose(img_arr, (2,1,0))
	return img_arr

def generate_MIPs(Data, suv_min, suv_max, intensity_type=None):
    """
    Generate MIPs for PET Data.
    Maximum Intensity Projection
    
    Data - PET Data.
    MIP - Maximum/Mean/Std Intensity Projection.
    """
    PET = np.clip(Data, suv_min, suv_max)
    if intensity_type == "maximum":
    	MIP_PET = np.max(PET, axis=1).astype("float")
    elif intensity_type == "mean":
    	MIP_PET = np.mean(PET, axis=1).astype("float")
    elif intensity_type == "std":
    	MIP_PET = np.std(PET, axis=1).astype("float")

    MIP_PET = MIP_PET/np.max(MIP_PET)# Pixel Normalization, value ranges b/w (0,1).
    MIP_PET = np.absolute(MIP_PET - np.amax(MIP_PET))
    MIP_PET = cv2.rotate(MIP_PET, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return MIP_PET

def generate_MIPs_Seg(Data):
    """
    Generate MIPs for Segmentation Data.
    
    Data - Segmentation Mask.
    """
    MIP_Seg = np.max(Data, axis=1).astype("float")
    #MIP_Seg = np.sum(Data, axis=1).astype("float")
    MIP_Seg = cv2.rotate(MIP_Seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return MIP_Seg

def generate_all_MIPs(save_path, SUV, SEG, suv_min, suv_max, ct_min, ct_max, rot_min=-90, rot_max=90, rot_interval=1):
	"""
	Generate rotating 2D MIPs along coronal direction from (-90, 90).
	"""
	for i in tqdm(range(rot_min, rot_max+1, rot_interval)):
		suv_temp = scipy.ndimage.rotate(SUV, angle=i, axes=(0,1))
		#ct_temp = scipy.ndimage.rotate(CT, angle=i, axes=(0,1))
		seg_temp = scipy.ndimage.rotate(SEG, angle=i, axes=(0,1))

		suv_max_IP = generate_MIPs(suv_temp, suv_min, suv_max, intensity_type="maximum")
		suv_mean_IP = generate_MIPs(suv_temp, suv_min, suv_max, intensity_type="mean")
		suv_std_IP = generate_MIPs(suv_temp, suv_min, suv_max, intensity_type="std")
		#ct_mean_IP = generate_MIPs(ct_temp, ct_min, ct_max, intensity_type="mean")
		seg_MIP = generate_MIPs_Seg(seg_temp)

		suv_max_IP = suv_max_IP[:,60:-60]
		suv_mean_IP = suv_mean_IP[:,60:-60]
		suv_std_IP = suv_std_IP[:,60:-60]
		#ct_mean_IP = ct_mean_IP[:,60:-60]
		seg_MIP = seg_MIP[:,60:-60]
		ZIP = (suv_max_IP-suv_mean_IP)/suv_std_IP

		np.save(os.path.join(save_path, "SUV_maxIP", str(i) + ".npy"), suv_max_IP)
		np.save(os.path.join(save_path, "SUV_meanIP", str(i) + ".npy"), suv_mean_IP)
		np.save(os.path.join(save_path, "SUV_stdIP", str(i) + ".npy"), suv_std_IP)
		#np.save(os.path.join(save_path, "CT_meanIP", str(i) + ".npy"), ct_mean_IP)
		np.save(os.path.join(save_path, "SEG", str(i) + ".npy"), seg_MIP)
		np.save(os.path.join(save_path, "ZIP", str(i) + ".npy"), ZIP)

def plot_dice(dice, path, val_interval):
	"""
	Plot the DICE score vs the number of epochs.
	"""
	epoch = [val_interval * (i + 1) for i in range(len(dice))]
	plt.plot(epoch, dice)
	plt.savefig(path, dpi=400)
	plt.xlabel("Number of Epochs")
	plt.ylabel("DICE")

def prepare_df_2D_rot_MIPs(path, args):
	df = pd.DataFrame(columns=['pat_ID', 'scan_date', 'SUV_maxIP', 'SUV_meanIP', 'SUV_stdIP', 'CT_meanIP', 'SEG', 'angle'])
	for pat_id in sorted(os.listdir(path)):
		for scan_date in sorted(os.listdir(os.path.join(path, pat_id))):
			for degree in range(args.rotation_min, args.rotation_max+1, args.rotation_interval):
				SUVmax_path = os.path.join(path, pat_id, scan_date, "SUV_maxIP", str(degree) + ".npy")
				SUVmean_path = os.path.join(path, pat_id, scan_date, "SUV_meanIP", str(degree) + ".npy")
				SUVstd_path = os.path.join(path, pat_id, scan_date, "SUV_stdIP", str(degree) + ".npy")
				CTmean_path = os.path.join(path, pat_id, scan_date, "CT_meanIP", str(degree) + ".npy")
				SEG_path = os.path.join(path, pat_id, scan_date, "SEG", str(degree) + ".npy")
				df_temp = pd.DataFrame({'pat_ID': [pat_id], 'scan_date': [scan_date], 'SUV_maxIP': [SUVmax_path], 'SUV_meanIP': [SUVmean_path], 'SUV_stdIP': [SUVstd_path], 'CT_meanIP': [CTmean_path], 'SEG': [SEG_path], 'angle': [degree]})
				df = df.append(df_temp, ignore_index=True)
	return df

def make_dirs(path, k):
	path_CV = os.path.join(path, "CV_" + str(k))
	if not os.path.exists(path_CV):
		os.mkdir(path_CV)
	path_Network_weights = os.path.join(path_CV, "Network_Weights")
	if not os.path.exists(path_Network_weights):
		os.mkdir(path_Network_weights)
	path_MIPs = os.path.join(path_CV, "MIPs")
	if not os.path.exists(path_MIPs):
		os.mkdir(path_MIPs)
	path_Metrics = os.path.join(path_CV, "Metrics")
	if not os.path.exists(path_Metrics):
		os.mkdir(path_Metrics)

def train(model, train_loader, optimizer, loss_function, device):
	model.train()
	epoch_loss = 0
	step = 0
	for train_data in tqdm(train_loader):
		step += 1
		inputs, labels = (
			train_data["PET_CT"].to(device),
			train_data["SEG"].to(device),
		)
		#print(inputs.shape)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
	epoch_loss /= step

	return epoch_loss

def save_model(model, epoch, optimizer, args, k, path_Output):
	best_metric_epoch = epoch + 1
	state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
	torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))

def writeTxtLine(input_path, values):
	with open(input_path, "a") as f:
		f.write("\n")
		f.write("{}".format(values[0]))
		for i in range(1, len(values)):
			f.write(",{}".format(values[i]))

def compute_metrics_validation(GT, pred, pat_ID, scan_date, degree, path):
	"""
	Computes, DICE, TP, FP, FN.
	Also Computes the Final Score, i.e. (0.5*DICE + 0.25*FP + 0.25*FN)
	"""
	pred = np.where(pred>1, 1, pred)

	if len(np.unique(GT)) == 1:
		dice = 11
		TP = 0
		disease_type = "Normal_Scan"
	else:
		dice = np.sum(pred[GT==1])*2.0 / (np.sum(pred) + np.sum(GT))
		TP = np.where(GT != pred, 0, GT)
		disease_type = "Tumorous_Scan"

	#TP = np.where(GT != pred, 0, GT)
	FP = pred - GT
	FP = np.where(FP == -1, 0, FP)
	FN = GT - pred
	FN = np.where(FN == -1, 0, FN)

	if np.count_nonzero(GT == 1) == 0:
		denominator = 1
	else:
		denominator = np.count_nonzero(GT == 0)

	tp_freq = np.count_nonzero(TP == 1)
	tp_percent = tp_freq/denominator
	fp_freq = np.count_nonzero(FP == 1)
	fp_percent = fp_freq/denominator
	fn_freq = np.count_nonzero(FN == 1)
	fn_percent = fn_freq/denominator

	if not os.path.isfile(path):
		with open(path, "w") as f:
			f.write("ID,scan_date,DISEASE_TYPE,rotation_angle,DICE,TP,TP_%,FP,FP_%,FN,FN_%")
	writeTxtLine(path, [pat_ID,scan_date,disease_type,degree,dice,tp_freq,tp_percent,fp_freq,fp_percent,fn_freq,fn_percent])

	return dice, fp_freq, fn_freq

def get_patient_id(files):
	pat_scan = files['SEG'].split('PETCT_')[-1]
	pat_id = 'PETCT_' + pat_scan.split('/')[0]
	scan_date = pat_scan.split('/')[1]
	degree = files['SEG'].split('/')[-1][:-4]
	return pat_id, scan_date, degree

def DICE_Score(prediction, GT):
    dice = np.sum(2.0 * prediction * GT) / (np.sum(prediction) + np.sum(GT))
    return dice

def save_MIP(save_path, Data, factor=1.):
    """
    Save the Image using PIL.
    
    save_path - Absolute Path.
    Data - (2D Image) Pixel value should lie between (0,1).
    """
    im = (factor * Data).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')
    im.save(save_path)

def overlap(MIP_img, GT, Pred):
    #Overlay the MIP_img and MIP_Seg into one.
    
    #TP - Blue, FP - Red, FN - Green.
    #TP = green, FN = red, FP = blue.
    
    temp = np.zeros((MIP_img.shape[0],MIP_img.shape[1],MIP_img.shape[2]))

    #TP = GT + Pred
    #TP = np.where(TP == 2, 1, 0)
    #FP = Pred - GT
    #FP = np.where(FP == -1, 0, FP)
    #FN = GT - Pred
    #FN = np.where(FN == -1, 0, FN)

    TP = np.where(GT != Pred, 0, GT)
    FP = Pred - GT
    FP = np.where(FP == -1, 0, FP)
    FN = GT - Pred
    FN = np.where(FN == -1, 0, FN)
    #TP = GT + Pred
    #TP = np.where(TP == 2, 1, 0)
    #TP = TP - FP - FN
    #TP = np.where(TP < 0, 0, TP)

    mul1 = np.where(TP == 0, 1, 0)
    add1 = np.where(TP == 0, 0, 255)
    mul2 = np.where(FP == 0, 1, 0)
    add2 = np.where(FP == 0, 0, 255)
    mul3 = np.where(FN == 0, 1, 0)
    add3 = np.where(FN == 0, 0, 255)

    for i in range(temp.shape[2]):
        temp[:,:,i] = MIP_img[:,:,i] * mul1
        
    for i in range(temp.shape[2]):
        temp[:,:,i] = temp[:,:,i] * mul2

    for i in range(temp.shape[2]):
        temp[:,:,i] = temp[:,:,i] * mul3
        
    #TPs (Green)
    temp[:,:,1] = temp[:,:,1] + add1
    #FPs (Blue)
    temp[:,:,2] = temp[:,:,2] + add2
    #FNs (Red)
    temp[:,:,0] = temp[:,:,0] + add3

    return temp

def overlay_segmentation(path, SUV, GT, prediction):
	MIP_img = (255. * SUV).astype(np.uint8)
	MIP_img = np.asarray(Image.fromarray(MIP_img).convert('RGB'))
	overlap_img = overlap(MIP_img, GT, prediction)
	save_MIP(path, overlap_img)

def validation(epoch, optimizer, post_pred, post_label, model, val_loader, device, args, dice_metric, metric_values, best_metric, k, val_files, path_Output):
	model.eval()
	pat_file = 0
	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date, degree = get_patient_id(val_files[pat_file])

			val_inputs, val_labels = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
			)
			roi_size = args.roi_size
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:].data.cpu().numpy()
			#print("Dice of Patient_ID: {} & scan_date: {} & degree {} is {}".format(pat_id, scan_date, degree, DICE_Score(prediction, GT)))

			# compute metric for current iteration
			if len(np.unique(GT)) == 2:
				val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
				val_labels = [post_label(i) for i in decollate_batch(val_labels)]
				dice_metric(y_pred=val_outputs, y=val_labels)
			#del val_data

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch+1) + ".txt")
			dice, fp, fn = compute_metrics_validation(GT, prediction, pat_id, scan_date, degree, path_dice)
			
			#Generate MIPs
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:]
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date))):
				os.makedirs(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date)))
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date), str(degree) + ".jpg")
			overlay_segmentation(path_MIP, SUV, GT, prediction)
			#try:
			#	overlay_segmentation(path_MIP, SUV, GT, prediction)
			#except:
			#	print("pat_ID = {}, scan_date = {}, degree = {}".format(pat_id, scan_date, degree))
			pat_file += 1

		#aggregate the final mean dice result
		metric = dice_metric.aggregate().item()
		#reset the status for next validation round
		dice_metric.reset()
		print("Validation DICE: {}".format(metric))
		metric_values.append(metric)
		#Save the model if DICE is increasing
		if metric > best_metric:
			best_metric = metric
			save_model(model, epoch, optimizer, args, k, path_Output)

	return metric_values, best_metric

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp, len(np.unique(conn_comp))

def lesion_wise_analysis(path, pred, GT, spacing, TP_threshold=0.0001):
	"""
	TP_Threshold:	DICE threshold b/w 2 lesions for it to be considered as a TP (cut-off = 0.1%).
	"""
	
	if not os.path.isfile(path):
		with open(path, "w") as f:
			#f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred,SUV_mean,SUV_max,CT_mean")
			f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred")

	#GT_multi_label, num_lesions_GT = label(GT)
	#pred_multi_label, num_lesions_pred = label(pred)
	GT_multi_label, num_lesions_GT = con_comp(GT.astype(int))
	pred_multi_label, num_lesions_pred = con_comp(pred.astype(int))

	tp_count = 0
	fn_count = 0
	fp_count = 0

	for lesion_num in range(1, num_lesions_pred):
		temp_arr_pred = np.where(pred_multi_label==lesion_num, 1, 0)
		lesion_type = None
		dice_lesion = dice_score(temp_arr_pred,GT)
		if dice_lesion < TP_threshold:
			#print("FP")
			fp_count += 1
			lesion_type = "FP"
			mtv_gt = 0
			mtv_pred = MTV_calculation(temp_arr_pred, spacing)
			SUV_mean = get_mean_median(SUV, temp_arr_pred, "mean")
			SUV_median = get_mean_median(SUV, temp_arr_pred, "median")
			SUV_max = get_mean_median(SUV, temp_arr_pred, "max")
			CT_mean = get_mean_median(CT, temp_arr_pred, "mean")
			writeTxtLine(path, [lesion_num,lesion_type,dice_lesion,mtv_gt,mtv_pred,SUV_mean,SUV_median,SUV_max,CT_mean])

	for lesion_num in range(1, num_lesions_GT):
		temp_arr_GT = np.where(GT_multi_label==lesion_num, 1, 0)
		lesion_type = None
		dice_lesion = dice_score(pred, temp_arr_GT)
		if dice_lesion < TP_threshold:
			#print("FN")
			fn_count += 1
			lesion_type = "FN"
			mtv_gt = MTV_calculation(temp_arr_GT, spacing)
			mtv_pred = 0
			SUV_mean = get_mean_median(SUV, temp_arr_GT, "mean")
			SUV_median = get_mean_median(SUV, temp_arr_GT, "median")
			SUV_max = get_mean_median(SUV, temp_arr_GT, "max")
			CT_mean = get_mean_median(CT, temp_arr_GT, "mean")
			writeTxtLine(path, [lesion_num,lesion_type,dice_lesion,mtv_gt,mtv_pred,SUV_mean,SUV_median,SUV_max,CT_mean])
		else:
			#print("TP")
			tp_count += 1
			lesion_type = "TP"
			mtv_gt = MTV_calculation(temp_arr_GT, spacing)
			mtv_pred = MTV_calculation(pred*temp_arr_GT, spacing)
			SUV_mean = get_mean_median(SUV, temp_arr_GT, "mean")
			SUV_median = get_mean_median(SUV, temp_arr_GT, "median")
			SUV_max = get_mean_median(SUV, temp_arr_GT, "max")
			CT_mean = get_mean_median(CT, temp_arr_GT, "mean")
			writeTxtLine(path, [lesion_num,lesion_type,dice_lesion,mtv_gt,mtv_pred,SUV_mean,SUV_median,SUV_max,CT_mean])

	Total_Number_of_Lesions = num_lesions_GT
	MTV_GT = MTV_calculation(GT, spacing)
	if len(np.unique(pred)) == 1:
		MTV_pred = 0
	else:
		MTV_pred = MTV_calculation(pred, spacing)
	return Total_Number_of_Lesions, MTV_GT, MTV_pred, tp_count, fn_count, fp_count

def inference(path_Output, val_loader, model, device, val_files, k, args):
	pat_file = 0
	DICE = []
	model.eval()

	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date, degree = get_patient_id(val_files[pat_file])
			pat_id_current = pat_id
			#try:
			val_inputs, val_labels = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
			)
			#print(val_inputs.shape)
			roi_size = args.roi_size
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:].data.cpu().numpy()

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(1) + ".txt")
			dice, fp, fn = compute_metrics_validation(GT, prediction, pat_id, scan_date, degree, path_dice)
			
			#Generate MIPs
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:]
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date))):
				os.makedirs(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date)))
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date), str(degree) + ".jpg")
			overlay_segmentation(path_MIP, SUV, GT, prediction)

			#Generate Predictions
			path_predictions = os.path.join(path_Output, "CV_" + str(k), "Predictions", str(pat_id), str(scan_date))
			if not os.path.exists(path_predictions):
				os.makedirs(path_predictions)
			np.save(os.path.join(path_predictions, "pred_" + str(degree) + ".npy"), prediction)

			#except:
			#	print("Inference Failed for Patient ID: {} and scan date: {}".format(pat_id, scan_date))

			pat_file += 1

		#np.save(os.path.join(path_Inference, "CV_" + str(k), "DICE", "DICE_" + str(roi_size[0]) + ".npy"), DICE)

def appropriate_crop(GT, prediction, GT_org, SUV):
	diff1 = GT.shape[0] - GT_org.shape[0]
	diff2 = GT.shape[1] - GT_org.shape[1]

	if diff1 != 0:
		if diff1%2 == 0: #Even
			diff_temp = int(diff1/2)
			GT_new = GT[diff_temp:-diff_temp,:]
			prediction_new = prediction[diff_temp:-diff_temp,:]
			SUV_new = SUV[diff_temp:-diff_temp,:]
		elif diff1%2 == 1: #Odd
			diff_temp = int(diff1/2)
			GT_new = GT[diff_temp:-diff_temp-1,:]
			prediction_new = prediction[diff_temp:-diff_temp-1,:]
			SUV_new = SUV[diff_temp:-diff_temp-1,:]
	else:
		print("Unpadding is not required along 0th dimension")

	if diff2 != 0:
		if diff2%2 == 0: #Even
			diff_temp = int(diff2/2)
			try:
				GT_new = GT_new[:,diff_temp:-diff_temp]
				prediction_new = prediction_new[:,diff_temp:-diff_temp]
				SUV_new = SUV_new[:,diff_temp:-diff_temp]
			except:
				GT_new = GT[:,diff_temp:-diff_temp]
				prediction_new = prediction[:,diff_temp:-diff_temp]
				SUV_new = SUV[:,diff_temp:-diff_temp]
		elif diff2%2 == 1: #Odd
			diff_temp = int(diff2/2)
			try:
				GT_new = GT_new[:,diff_temp:-diff_temp-1]
				prediction_new = prediction_new[:,diff_temp:-diff_temp-1]
				SUV_new = SUV_new[:,diff_temp:-diff_temp-1]
			except:
				GT_new = GT[:,diff_temp:-diff_temp-1]
				prediction_new = prediction[:,diff_temp:-diff_temp-1]
				SUV_new = SUV[:,diff_temp:-diff_temp-1]
	else:
		print("Unpadding is not required along 1st dimension")

	return GT_new, prediction_new, SUV_new

def inference_full_image(path_Output, val_loader, model, device, val_files, k, args):
	pat_file = 0
	DICE = []
	model.eval()

	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date, degree = get_patient_id(val_files[pat_file])
			pat_id_current = pat_id
			#try:
			val_inputs, val_labels, val_original = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
				val_data["SEG_org"].to(device),
			)
			GT_org = val_original.data.cpu().numpy()[0,0,:,:]
			val_outputs = model(val_inputs)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:].data.cpu().numpy()
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:]

			GT_final, prediction_final, SUV_final = appropriate_crop(GT, prediction, GT_org, SUV)

			if int(DICE_Score(GT_final, GT_org)) != 1:
				print("Unpadding Mismatch for Patient ID: {} & Scan Date: {}".format(pat_id, scan_date))

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(1) + ".txt")
			dice, fp, fn = compute_metrics_validation(GT_final, prediction_final, pat_id, scan_date, degree, path_dice)
			
			#Generate MIPs
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date))):
				os.makedirs(os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date)))
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date), str(degree) + ".jpg")
			overlay_segmentation(path_MIP, SUV_final, GT_final, prediction_final)

			#Generate Predictions
			path_predictions = os.path.join(path_Output, "CV_" + str(k), "Predictions", str(pat_id), str(scan_date))
			if not os.path.exists(path_predictions):
				os.makedirs(path_predictions)
			np.save(os.path.join(path_predictions, "pred_" + str(degree) + ".npy"), prediction_final)

			#except:
			#	print("Inference Failed for Patient ID: {} and scan date: {}".format(pat_id, scan_date))

			pat_file += 1


def get_val_files(df):
	SUV = sorted(df['SUV'].tolist())
	CT = sorted(df['CT'].tolist())
	SEG = sorted(df['SEG'].tolist())
	val_files = [
		{"SUV": SUV_name, "CT": CT_name, "SEG": SEG_name}
		for SUV_name, CT_name, SEG_name in zip(SUV, CT, SEG)
	]
	return val_files

def get_patient_id_new(files):
	pat_scan = files['SEG'].split('PETCT_')[-1]
	pat_id = 'PETCT_' + pat_scan.split('/')[0]
	scan_date = pat_scan.split('/')[1]
	return pat_id, scan_date

def save_npy_as_nii(arr, SUV_arr, SUV_path, save_path):
    SUV = nib.load(SUV_path) 
    #Clip the SUV values between (1, 5) after setting all the SUV values < 1 to 0.
    SUV_arr_new = np.where(SUV_arr < 1, 0, SUV_arr)
    SUV_arr_new = np.clip(SUV_arr_new, 1, 5)
    arr_new = arr*SUV_arr_new

    arr_nii = nib.Nifti1Image(arr, SUV.affine)
    #nib.save(arr_nii, os.path.join(save_path, "pred.nii.gz"))

    arr_nii_SUV = nib.Nifti1Image(arr_new, SUV.affine)
    #nib.save(arr_nii_SUV, os.path.join(save_path, "pred*suv.nii.gz"))

    arr_new_norm = arr_new/np.max(arr_new)
    arr_new_norm = np.where(arr_new_norm <= 0.05, 0, arr_new_norm)#take the threshold @ 0.10
    arr_nii_norm = nib.Nifti1Image(arr_new_norm, SUV.affine)
    #nib.save(arr_nii_norm, os.path.join(save_path, "pred*suv_normalized.nii.gz"))
    nib.save(arr_nii_norm, os.path.join(save_path, "TPDM.nii.gz"))



def generate_TPDM_patient_wise(SUV_path, save_path, pred_path):
	SUV_arr = sitk.GetArrayFromImage(sitk.ReadImage(SUV_path))
	SUV_arr = np.transpose(SUV_arr, (2,1,0))

	final_arr = np.zeros(SUV_arr.shape)
	rot_MIPs_list = []
	degree_list = []

	for i in sorted(os.listdir(pred_path)):
		degree = int(i.split("_")[-1][:-4])
		temp_arr = np.load(os.path.join(pred_path, i))
		temp_arr = cv2.rotate(temp_arr, cv2.ROTATE_90_CLOCKWISE)
		temp_arr_new = np.zeros((400, temp_arr.shape[1]))
		if temp_arr.shape[0] < 400:
			factor = (400 - temp_arr.shape[0])/2
			if (2*factor)%2 == 1:
				temp_arr_new[int(factor):-int(factor)-1,:] = temp_arr
			elif (2*factor)%2 == 0:
				temp_arr_new[int(factor):-int(factor),:] = temp_arr
		elif temp_arr.shape[0] > 400:
			factor = (temp_arr.shape[0] - 400)/2
			if (2*factor)%2 == 1:
				temp_arr_new = temp_arr[int(factor):-int(factor)-1,:]
			elif (2*factor)%2 == 0:
				temp_arr_new = temp_arr[int(factor):-int(factor),:]
		rot_MIPs_list.append(temp_arr_new)
		degree_list.append(degree)

	for i in tqdm(range(len(rot_MIPs_list))):
		temp = np.zeros(SUV_arr.shape)
		for j in range(400):
			temp[:,j,:] = rot_MIPs_list[i]
		temp_new = scipy.ndimage.rotate(temp, angle=degree_list[i], axes=(0,1))
		factor = (temp_new.shape[0] - 400)/2

		if temp_new.shape[0] > 400:
			if (2*factor)%2 == 1:
				temp_new_new = temp_new[int(factor):-int(factor)-1, int(factor):-int(factor)-1,:]
			elif (2*factor)%2 == 0:
				temp_new_new = temp_new[int(factor):-int(factor), int(factor):-int(factor),:]
		elif temp_new.shape[0] == 400:
			temp_new_new = temp_new
		final_arr += temp_new_new

	final_arr_new = np.zeros(SUV_arr.shape)
	for i in range(final_arr_new.shape[-1]):
		final_arr_new[:,:,i] = np.fliplr(final_arr[:,:,i])

	save_npy_as_nii(final_arr_new, SUV_arr, SUV_path, save_path)


def generate_TPDMs(path, path_predictions, df_val):
	"""
	path:	path to save the generated TPDMs.
	path_predictions:	path where patient wise prediction of tumor segmentation from 2D MIPs exist.
	df_val:	DataLoader containing all the patient information for generating TPDMs.
	"""
	val_files = get_val_files(df_val)

	for pat_file in tqdm(range(len(val_files))):
		pat_id, scan_date = get_patient_id_new(val_files[pat_file])
		#if pat_id == "PETCT_02ba7e20f5": #"PETCT_0b57b247b6"
		SUV_path = val_files[pat_file]['SUV']
		save_path = os.path.join(path, str(pat_id), str(scan_date))
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		if not os.path.isfile(os.path.join(path, str(pat_id), str(scan_date), "TPDM.nii.gz")):
			print(pat_id)
			path_temp = os.path.join(path_predictions, str(pat_id), str(scan_date))
			generate_TPDM_patient_wise(SUV_path, save_path, path_temp)
