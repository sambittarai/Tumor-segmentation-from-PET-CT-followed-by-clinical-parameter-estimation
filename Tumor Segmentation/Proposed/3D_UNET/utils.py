import numpy as np
import pandas as pd
import os
from config import parse_args
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
from sklearn.model_selection import StratifiedShuffleSplit
from medpy import metric

def get_stratified_df_split(df, k):
	grouped = df.groupby('pat_ID').agg({'sex': 'nunique', 'age': 'nunique'}).reset_index()
	split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)  # Adjust the test_size and random_state as desired
	dfs = []
	for train_index, test_index in split.split(grouped, grouped[['sex', 'age']]):
		train_patients = grouped['pat_ID'].iloc[train_index]
		test_patients = grouped['pat_ID'].iloc[test_index]
		train_df = df[df['pat_ID'].isin(train_patients)]
		test_df = df[df['pat_ID'].isin(test_patients)]
		dfs.append((train_df, test_df))
	train_df_fold1, test_df_fold1 = dfs[k]
	return train_df_fold1, test_df_fold1

def plot_dice(dice, path, validation_interval):
	epoch = [validation_interval * (i + 1) for i in range(len(dice))]
	plt.plot(epoch, dice)
	plt.savefig(path, dpi=400)
	plt.xlabel("Number of Epochs")
	plt.ylabel("DICE")

def overlap(MIP_img, GT, Pred):
    #Overlay the MIP_img and MIP_Seg into one.
    
    #TP - Blue, FP - Red, FN - Green.
    #TP = green, FN = red, FP = blue.
    
    temp = np.zeros((MIP_img.shape[0],MIP_img.shape[1],MIP_img.shape[2]))

    TP = np.where(GT != Pred, 0, GT)
    #TP = 0
    FP = Pred - GT
    FP = np.where(FP == -1, 0, FP)
    #FP = 0
    FN = GT - Pred
    FN = np.where(FN == -1, 0, FN)
    #FN = 0
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
    #temp[:,:,0] = temp[:,:,0] + add2
    #FNs (Red)
    temp[:,:,0] = temp[:,:,0] + add3

    return temp

"""
def overlap(MIP_img, GT, Pred):
    
    #TP - Blue, FP - Red, FN - Green.
    #TP = green, FN = red, FP = blue.
    
    temp = np.zeros((MIP_img.shape[0],MIP_img.shape[1],MIP_img.shape[2]))

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

    add_gt1 = np.where(GT == 0, 0, 125)
    add_gt2 = np.where(GT == 0, 0, 125)

    mul_gt = np.where(GT == 0, 1, 0)

    for i in range(temp.shape[2]):
        temp[:,:,i] = MIP_img[:,:,i] * mul_gt
        
    for i in range(temp.shape[2]):
        temp[:,:,i] = temp[:,:,i] * mul2

    #GTs (Green)
    temp[:,:,1] = temp[:,:,1] + add_gt1
    temp[:,:,2] = temp[:,:,2] + add_gt2

    #FPs (Blue)
    temp[:,:,0] = temp[:,:,0] + add2

    return temp
"""

def save_MIP(save_path, Data, factor=1.):
    """
    Save the Image using PIL.
    
    save_path - Absolute Path.
    Data - (2D Image) Pixel value should lie between (0,1).
    """
    im = (factor * Data).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')
    #im = im.resize((4288, 2848))
    im.save(save_path)

def generate_MIPs_PET(Data):
    """
    Generate MIPs for PET Data.
    
    Data - PET Data.
    MIP - Maximum/Mean Intensity Projection.
    """
    #PET = np.clip(Data, 0, 3600)# Enhance the contrast of the soft tissue.
    #PET = np.clip(Data, 0, 2.5)
    #PET = np.clip(Data, 0, 2.5)
    PET = Data
    
    MIP_PET = np.max(PET, axis=1).astype("float") # (267, 512).
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
    #MIP_Seg = MIP_Seg/np.max(MIP_Seg)
    MIP_Seg = cv2.rotate(MIP_Seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return MIP_Seg

def overlay_segmentation(PET, Seg1, Seg2, path):
    """
    Overlay Tumor Segmentation & normal atlas mask onto the PET Data.
    
    Seg1 - Tumor Prediction.
    Seg2 - Ground Truth.
    """
    if PET.shape == Seg1.shape:
        MIP_PET = generate_MIPs_PET(PET) # (267,512)
        MIP_Seg1 = generate_MIPs_Seg(Seg1) # (267,512)
        MIP_Seg2 = generate_MIPs_Seg(Seg2) # (267,512)

        MIP_Seg1 = np.where(MIP_Seg1 != 1, 0, 1) #Not sure why, but look at this later.
                
        MIP_img = (255. * MIP_PET).astype(np.uint8)
        MIP_img = np.asarray(Image.fromarray(MIP_img).convert('RGB'))
        
        overlap_img = overlap(MIP_img, MIP_Seg2, MIP_Seg1)
        save_MIP(path, overlap_img)
    else:
        print("PET and Segmentation dimension does not match.")

def filter_tumor_scans(df):
	no_tumors = []
	for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
		Seg = sitk.GetArrayFromImage(sitk.ReadImage(row["SEG"]))
		if len(np.unique(Seg)) == 1:
			no_tumors.append(row["scan_date"])
	df_filtered = df[~df.scan_date.isin(no_tumors)].reset_index(drop=True)
	df_unfiltered = df[df.scan_date.isin(no_tumors)].reset_index(drop=True)
	return df_filtered, df_unfiltered

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

def get_df(path):
	df = pd.DataFrame(columns=['pat_ID', 'scan_date', 'CT', 'SUV', 'SEG'])
	for pat_id in sorted(os.listdir(path)):
		#scan_date = os.listdir(os.path.join(path, pat_id))[0]
		for scan_date in sorted(os.listdir(os.path.join(path, pat_id))):
			CT_path = os.path.join(path, pat_id, scan_date, "CTres.nii.gz")
			SUV_path = os.path.join(path, pat_id, scan_date, "SUV.nii.gz")
			SEG_path = os.path.join(path, pat_id, scan_date, "SEG.nii.gz")
			df_temp = pd.DataFrame({'pat_ID':[pat_id], 'scan_date': [scan_date], 'CT':[CT_path], 'SUV':[SUV_path], 'SEG':[SEG_path]})
			df = df.append(df_temp, ignore_index=True)
	return df

def DICE_Score(prediction, GT):
    dice = np.sum(2.0 * prediction * GT) / (np.sum(prediction) + np.sum(GT))
    return dice

def save_model(model, epoch, optimizer, args, k, path_Output):
	best_metric_epoch = epoch + 1
	state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
	torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))


def train(model, train_loader, train_files, optimizer, loss_function, device):
	model.train()
	epoch_loss = 0
	step = 0
	pat_file = 0
	for train_data in tqdm(train_loader):
		pat_id, scan_date = get_patient_id(train_files[pat_file])
		step += 1
		inputs, labels = (
			train_data["PET_CT"].to(device),
			train_data["SEG"].to(device),
		)
		#print(inputs.shape)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_function(outputs, labels)
		#print("Patient ID: {}; Scan Date: {}; loss: {}".format(pat_id, scan_date, loss))
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		pat_file += 1
	epoch_loss /= step

	return epoch_loss

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp, len(np.unique(conn_comp))

"""

def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg
"""


def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score

def writeTxtLine(input_path, values):
	with open(input_path, "a") as f:
		f.write("\n")
		f.write("{}".format(values[0]))
		for i in range(1, len(values)):
			f.write(",{}".format(values[i]))
            

def calculate_HD95(GT, pred):
	hd = metric.binary.hd95(GT, pred)
	return hd

def compute_metrics(GT, pred, pat_ID, scan_date, path, Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise, FP_lesion_wise):
	"""
	Computes, DICE, TP, FP, FN.
	Also Computes the Final Score, i.e. (0.5*DICE + 0.25*FP + 0.25*FN)
	"""

	try:
		hd95 = calculate_HD95(GT, pred)
		asd = metric.binary.asd(GT, pred)
	except:
		hd95 = 1.1111
		asd = 1.1111

	pred = np.where(pred>1, 1, pred)
	dice = np.sum(pred[GT==1])*2.0 / (np.sum(pred) + np.sum(GT))

	TP = np.where(GT != pred, 0, GT)
	FP = pred - GT
	FP = np.where(FP == -1, 0, FP)
	FN = GT - pred
	FN = np.where(FN == -1, 0, FN)

	if np.count_nonzero(GT == 1) == 0:
		denominator = 1
	else:
		denominator = np.count_nonzero(GT == 0)

	tp_freq = np.count_nonzero(TP == 1)
	#tp_percent = tp_freq/denominator
	fp_freq = np.count_nonzero(FP == 1)
	#fp_percent = fp_freq/denominator
	fn_freq = np.count_nonzero(FN == 1)
	#fn_percent = fn_freq/denominator

	if not os.path.isfile(path):
		with open(path, "w") as f:
			f.write("ID,scan_date,DICE,Total_Number_of_lesions,MTV_GT,MTV_pred,Number_of_TP_lesions,Number_of_FN_lesions,Number_of_FP_lesions,TP_voxels,FP_voxels,FN_voxels,HD95,ASD")
	writeTxtLine(path, [pat_ID,scan_date,dice,Total_Number_of_lesions,MTV_GT,MTV_pred,TP_lesion_wise,FN_lesion_wise,FP_lesion_wise,tp_freq,fp_freq,fn_freq,hd95,asd])

	return dice, fp_freq, fn_freq

def compute_metrics_validation(GT, pred, pat_ID, scan_date, path):
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
			f.write("ID,scan_date,DISEASE_TYPE,DICE,TP,TP_%,FP,FP_%,FN,FN_%")
	writeTxtLine(path, [pat_ID,scan_date,disease_type,dice,tp_freq,tp_percent,fp_freq,fp_percent,fn_freq,fn_percent])

	return dice, fp_freq, fn_freq

def get_patient_id(files):
	pat_scan = files['SUV'].split('PETCT_')[-1]
	pat_id = 'PETCT_' + pat_scan.split('/')[0]
	scan_date = pat_scan.split('/')[1]
	return pat_id, scan_date


def validation(epoch, optimizer, post_pred, post_label, model, val_loader, device, args, dice_metric, metric_values, best_metric, k, val_files, path_Output):
	model.eval()
	pat_file = 0
	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])

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
			GT = val_labels[0,0,:,:,:].data.cpu().numpy()
			#print("Patient ID: {} Scan Date: {} dice: {}".format(pat_id, scan_date, DICE_Score(prediction, GT)))

			# compute metric for current iteration
			if len(np.unique(GT)) == 2:
				val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
				val_labels = [post_label(i) for i in decollate_batch(val_labels)]
				dice_metric(y_pred=val_outputs, y=val_labels)
			#del val_data

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch+1) + ".txt")
			#print("unique: ", np.unique(GT))
			#print("len(np.unique(GT))", len(np.unique(GT)))
			dice, fp, fn = compute_metrics_validation(GT, prediction, pat_id, scan_date, path_dice)

			#Generate MIPs for the predictions when the current metric (DICE) is > best_metric (DICE).
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date) + ".jpg")
			overlay_segmentation(SUV, prediction, GT, path_MIP)
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

def npy2nii(npy_arr, target, path):
	"""
	npy_arr - Numpy array.
	target - sitk Image.
	"""
	if len(npy_arr.shape) == len(target.GetSize()):
		if target.GetSize()[2] == npy_arr.shape[0]:
			final_img = sitk.GetImageFromArray(temp)
			final_img.CopyInformation(target)
			sitk.WriteImage(final_img, path)
		else:
			temp = np.transpose(npy_arr, (2,1,0))
			final_img = sitk.GetImageFromArray(temp)
			final_img.CopyInformation(target)
			sitk.WriteImage(final_img, path)
	else:
		print("Size Mismatch b/w npy array and sitk image.")

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


def threshold_liver_SUV(SUV, prediction, path_liver_SUV, scan_date):
	df = pd.read_csv(path_liver_SUV)
	threshold = np.array(df[df["scan_date"]==scan_date]["intensity_normalization_factor"])[0]

	if threshold > 4:
		threshold = 2.5
	else:
		threshold += 0.5
	SUV_new = np.where(SUV > threshold, 1, 0)
	prediction_new = prediction*SUV_new
	return prediction_new

def MTV_calculation(arr, spacing, volume=False):
	if len(np.unique(arr)==2):
		_, freq = np.unique(arr, return_counts=True)
		if volume==True:
			mtv = freq[1]*spacing[0]*spacing[1]*spacing[2]
		else:
			mtv = freq[1]
	else:
		mtv = 0
	return mtv

def get_mean_median(PET, Seg, Intensity_type):
    index_z = []
    for i in range(Seg.shape[2]):
        if len(np.unique(Seg[:,:,i])) == 2:
            index_z.append(i)

    coordinates = []
    for i in index_z:
        X_temp, Y_temp = np.where(Seg[:,:,i]==1)
        for j in range(len(X_temp)):
            coordinates.append([X_temp[j], Y_temp[j], i])

    pixel_values = []
    for i in coordinates:
        pixel_values.append(PET[i[0], i[1], i[2]])
        
    if Intensity_type == "mean":
        return (np.mean(pixel_values))
    elif Intensity_type == "median":
        return (np.median(pixel_values))
    elif Intensity_type == "max":
    	return (np.max(pixel_values))


"""
def lesion_wise_analysis(path, pred, GT, SUV, CT, spacing, TP_threshold=0.01):
	#GT_multi_label, num_lesions_GT = label(GT)
	#pred_multi_label, num_lesions_pred = label(pred)

	GT_multi_label, num_lesions_GT = con_comp(GT.astype(int))
	pred_multi_label, num_lesions_pred = con_comp(pred.astype(int))

	fn_count = 0
	#fp_count = 0
	tp_count = 0
	#gt_temp = []
	#pred_temp = []
	#fn_val = []
	#lesion_type = None

	for i in range(1, num_lesions_GT):
		#prediction_arr = []
		if not os.path.isfile(path):
			with open(path, "w") as f:
				#f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred,SUV_mean,SUV_max,CT_mean")
				f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred,SUV_mean,SUV_median,SUV_max,CT_mean")

		gt_temp = []
		fn_val = []
		tp_count_temp = 0
		lesion_num_GT = i
		temp_arr_GT = np.where(GT_multi_label==lesion_num_GT, 1, 0)
		lesion_type = None
		for j in range(1, num_lesions_pred):
			lesion_num_pred = j
			temp_arr_pred = np.where(pred_multi_label==lesion_num_pred, 1, 0)
			overlap_img = temp_arr_GT*temp_arr_pred
			dice = dice_score(temp_arr_GT,temp_arr_pred)
	    #TPs
			if dice >= TP_threshold:
				tp_count_temp += 1
				if i not in gt_temp:
					tp_count += 1
					lesion_type = "TP"
					dice_lesion = dice
					mtv_gt = MTV_calculation(temp_arr_GT, spacing)
					mtv_pred = MTV_calculation(temp_arr_pred, spacing)
					SUV_mean = get_mean_median(SUV, temp_arr_GT, "mean")
					SUV_median = get_mean_median(SUV, temp_arr_GT, "median")
					SUV_max = get_mean_median(SUV, temp_arr_GT, "max")
					CT_mean = get_mean_median(CT, temp_arr_GT, "mean")
					writeTxtLine(path, [lesion_num_GT,lesion_type,dice_lesion,mtv_gt,mtv_pred,SUV_mean,SUV_median,SUV_max,CT_mean])
				gt_temp.append(i)
	    #FNs
		if tp_count_temp == 0:
			fn_val.append(lesion_num_GT)
			fn_count += 1
			lesion_type = "FN"
			dice_lesion = 0
			mtv_gt = MTV_calculation(temp_arr_GT, spacing)
			mtv_pred = 0
			SUV_mean = get_mean_median(SUV, temp_arr_GT, "mean")
			SUV_median = get_mean_median(SUV, temp_arr_GT, "median")
			SUV_max = get_mean_median(SUV, temp_arr_GT, "max")
			CT_mean = get_mean_median(CT, temp_arr_GT, "mean")
			writeTxtLine(path, [lesion_num_GT,lesion_type,dice_lesion,mtv_gt,mtv_pred,SUV_mean,SUV_median,SUV_max,CT_mean])

	Total_Number_of_lesions = num_lesions_GT
	MTV_GT = MTV_calculation(GT, spacing)
	if len(np.unique(pred)) == 1:
		MTV_pred = 0
	else:
		MTV_pred = MTV_calculation(pred, spacing)
	TP_lesion_wise = tp_count
	FN_lesion_wise = fn_count
	return Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise
"""
def lesion_wise_analysis(path, pred, GT, SUV, CT, spacing, TP_threshold=0.0001):
	"""
	TP_Threshold:	DICE threshold b/w 2 lesions for it to be considered as a TP (cut-off = 0.1%).
	"""
	
	if not os.path.isfile(path):
		with open(path, "w") as f:
			#f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred,SUV_mean,SUV_max,CT_mean")
			f.write("Lesion_Number,Lesion_Type,DICE,MTV_GT,MTV_pred,SUV_mean,SUV_median,SUV_max,CT_mean")

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

def inference(val_loader, post_pred, post_label, model, device, val_files, k, dice_metric, metric_values, args):
	path_Inference = args.path_Inference_Output
	pat_file = 0
	DICE = []
	model.eval()

	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])
			#try:

			val_inputs, val_labels, val_inputs_temp = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
				val_data["PET_CT_old"].to(device),
			)
			#val_inputs = val_inputs_all[:,:-2,:,:,:]
			roi_size = args.roi_size_val
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:,:].data.cpu().numpy()

			SUV_img = val_inputs_temp[0,0,:,:,:].data.cpu().numpy()
			CT_img = val_inputs_temp[0,1,:,:,:].data.cpu().numpy()

			path_Lesions = os.path.join(path_Inference, "CV_" + str(k), "Lesion_wise_Metrics")
			if not os.path.exists(path_Lesions):
				os.mkdir(path_Lesions)

			path_lesion_wise_analysis = os.path.join(path_Inference, "CV_" + str(k), "Lesion_wise_Metrics", str(pat_id) + "_" + str(scan_date) + ".txt") #Lesion wise analysis
			Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise, FP_lesion_wise = lesion_wise_analysis(path_lesion_wise_analysis, prediction, GT, SUV_img, CT_img, args.spacing, args.TP_threshold)

			path_dice = os.path.join(path_Inference, "CV_" + str(k), "Metrics", "metrics.txt")
			dice, fp, fn = compute_metrics(GT, prediction, pat_id, scan_date, path_dice, Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise, FP_lesion_wise)
			DICE.append(dice)
			#Generate MIPs for the predictions
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			#path_MIP = os.path.join(path_Inference, "CV_" + str(k), "MIPs", "roi_" + str(roi_size[0]), str(pat_id) + "_" + str(scan_date) + "_" + str(dice) + "_" + str(fp) + "_" + str(fn) + ".jpg")
			path_MIP = os.path.join(path_Inference, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date) + ".jpg")
			overlay_segmentation(SUV, prediction, GT, path_MIP)
			#except:
			#	print("Inference Failed for Patient ID: {} and scan date: {}".format(pat_id, scan_date))

			pat_file += 1

def save_predictions(path_prediction, prediction, pat_id, scan_date, path_SUV):
	SUV = nib.load(path_SUV)
	pred_img = nib.Nifti1Image(prediction, SUV.affine)
	save_path = os.path.join(path_prediction, pat_id, scan_date)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	nib.save(pred_img, os.path.join(save_path, "prediction.nii.gz"))

def validation_inference(val_loader, model, device, val_files, k, dice_metric, metric_values, args):
	path_Inference = args.path_Inference_Output
	path_prediction = os.path.join(path_Inference, "Predictions")
	if not os.path.exists(path_prediction):
		os.mkdir(path_prediction)

	pat_file = 0
	DICE = []
	model.eval()
	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])
			val_inputs, val_labels, val_inputs_temp = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
				val_data["PET_CT_old"].to(device),
			)
			roi_size = args.roi_size_val
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:,:].data.cpu().numpy()
			path_dice = os.path.join(path_Inference, "CV_" + str(k), "Metrics", "metrics.txt")
			dice, fp, fn = compute_metrics_validation(GT, prediction, pat_id, scan_date, path_dice)
			DICE.append(dice)
			#Generate MIPs for the predictions
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			path_MIP = os.path.join(path_Inference, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date) + ".jpg")
			overlay_segmentation(SUV, prediction, GT, path_MIP)
			#ensure_matrix_size(prediction, GT, val_files[pat_file]['SEG'], pat_id, path_prediction)
			save_predictions(path_prediction, prediction, pat_id, scan_date, val_files[pat_file]['SUV'])
			pat_file += 1

def save_npy_as_nii(arr, SUV_path, save_path):
    SUV = nib.load(SUV_path) 
    arr_nii = nib.Nifti1Image(arr, SUV.affine)
    nib.save(arr_nii, save_path)

def generate_MIPs_SUV_FP_analysis(SUV, direction=None):
	# direction == 1 (coronal); direction == 0 (saggital)

	#SUV = np.clip(SUV, 0, 5)
	MIP_SUV = np.max(SUV, axis=direction).astype("float")
	MIP_SUV = MIP_SUV/np.max(MIP_SUV)# Pixel Normalization, value ranges b/w (0,1).
	MIP_SUV = np.absolute(MIP_SUV - np.amax(MIP_SUV))
	MIP_SUV = cv2.rotate(MIP_SUV, cv2.ROTATE_90_COUNTERCLOCKWISE)
	return MIP_SUV

def generate_MIPs_SEG_FP_analysis(SEG, direction=None):
	# direction == 1 (coronal); direction == 0 (saggital)

	MIP_SEG = np.max(SEG, axis=direction).astype("float")
	MIP_SEG = cv2.rotate(MIP_SEG, cv2.ROTATE_90_COUNTERCLOCKWISE)
	return MIP_SEG

def overlay_segmentation_FP_analysis(SUV, Seg1, Seg2, path):
	"""
	Overlay Tumor Segmentation & normal atlas mask onto the PET Data.

	Seg1 - Tumor Prediction.
	Seg2 - Ground Truth.
	"""
	if SUV.shape == Seg1.shape:
		#SUV = np.clip(SUV, 0, 15)

		#Generate MIP SUV from saggital and coronal direction
		#coronal
		MIP_SUV_cor = generate_MIPs_SUV_FP_analysis(SUV, direction=1)
		#saggital
		MIP_SUV_sag = generate_MIPs_SUV_FP_analysis(SUV, direction=0)

		#Generate MIP SEG
		#coronal
		MIP_Seg1_cor = generate_MIPs_SEG_FP_analysis(Seg1, direction=1)
		MIP_Seg2_cor = generate_MIPs_SEG_FP_analysis(Seg2, direction=1)
		#saggital
		MIP_Seg1_sag = generate_MIPs_SEG_FP_analysis(Seg1, direction=0)
		MIP_Seg2_sag = generate_MIPs_SEG_FP_analysis(Seg2, direction=0)

		MIP_Seg1_cor = np.where(MIP_Seg1_cor != 1, 0, 1)#Not sure why, but look at this later.
		MIP_Seg1_sag = np.where(MIP_Seg1_sag != 1, 0, 1)#Not sure why, but look at this later.

		MIP_img_cor = (255. * MIP_SUV_cor).astype(np.uint8)
		MIP_img_cor = np.asarray(Image.fromarray(MIP_img_cor).convert('RGB'))
		MIP_img_sag = (255. * MIP_SUV_sag).astype(np.uint8)
		MIP_img_sag = np.asarray(Image.fromarray(MIP_img_sag).convert('RGB'))

		overlap_img_cor = overlap(MIP_img_cor, MIP_Seg2_cor, MIP_Seg1_cor)
		#save_MIP(os.path.join(path, "coronal_GT.jpg"), overlap_img_cor)
		#save_MIP(os.path.join(path, "coronal_FP.jpg"), overlap_img_cor)
		save_MIP(os.path.join(path, "coronal.jpg"), overlap_img_cor)

		overlap_img_sag = overlap(MIP_img_sag, MIP_Seg2_sag, MIP_Seg1_sag)
		#save_MIP(os.path.join(path, "saggital_GT.jpg"), overlap_img_sag)
		#save_MIP(os.path.join(path, "saggital_FP.jpg"), overlap_img_sag)
		save_MIP(os.path.join(path, "saggital.jpg"), overlap_img_sag)

		# overlap_img_cor - np array; overlap_img_sag - np array; 
		# MIP_SUV_cor - np array; MIP_SUV_sag - np array

	else:
		print("PET and Segmentation dimension does not match.")

def FP_Analysis(val_loader, post_pred, post_label, model, device, val_files, k, dice_metric, metric_values, args):
	"""
	False Positive Analysis by Alexander
	"""
	path_FP_Analysis_Output = args.path_FP_Analysis_Output
	pat_file = 0
	DICE = []
	model.eval()

	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])
			#try:
			val_inputs, val_labels, val_inputs_temp = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
				val_data["PET_CT_old"].to(device),
			)
			
			roi_size = args.roi_size_val
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)

			GT = val_labels[0,0,:,:,:].data.cpu().numpy()
			FP = prediction - GT
			FP = np.where(FP < 0, 0, FP)

			prediction_final = prediction + FP
			prediction_final = np.where(prediction_final > 1, 1, prediction_final)

			SUV_img = val_inputs_temp[0,0,:,:,:].data.cpu().numpy()
			CT_img = val_inputs_temp[0,1,:,:,:].data.cpu().numpy()

			path_Lesions = os.path.join(path_FP_Analysis_Output, "CV_" + str(k), "Lesion_wise_Metrics")
			if not os.path.exists(path_Lesions):
				os.mkdir(path_Lesions)

			path_FP_outputs = os.path.join(path_FP_Analysis_Output, "CV_" + str(k), "FP_predictions")
			if not os.path.exists(path_FP_outputs):
				os.mkdir(path_FP_outputs)

			path_temp = os.path.join(path_FP_outputs, str(pat_id), str(scan_date))
			if not os.path.exists(path_temp):
				os.makedirs(path_temp)

			SUV_path = os.path.join("/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", str(pat_id), str(scan_date), "SUV.nii.gz")

			save_path_SUV = os.path.join(path_temp, "SUV.nii.gz")
			save_npy_as_nii(SUV_img, SUV_path, save_path_SUV)
			save_path_CT = os.path.join(path_temp, "CT.nii.gz")
			save_npy_as_nii(CT_img, SUV_path, save_path_CT)
			save_path_GT = os.path.join(path_temp, "GT.nii.gz")
			save_npy_as_nii(GT, SUV_path, save_path_GT)
			save_path_FP = os.path.join(path_temp, "FP.nii.gz")
			save_npy_as_nii(FP, SUV_path, save_path_FP)
			save_path_pred = os.path.join(path_temp, "prediction.nii.gz")
			save_npy_as_nii(prediction_final, SUV_path, save_path_pred)

			#Generate MIPs for the predictions
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			path_MIP = os.path.join(path_FP_Analysis_Output, "CV_" + str(k), "MIPs", str(pat_id), str(scan_date))
			if not os.path.exists(path_MIP):
				os.makedirs(path_MIP)
			overlay_segmentation_FP_analysis(SUV, prediction, GT, path_MIP)
			pat_file += 1

def ensemble(val_loader_1, val_loader_2, model_1, model_2, device, val_files_1, val_files_2, k, path_Output, args):
	path_prediction = os.path.join(path_Output, "CV_" + str(k), "prediction")
	if not os.path.exists(path_prediction):
		os.mkdir(path_prediction)

	#Lesionwise Analysis
	for pat_id in tqdm(sorted(os.listdir(path_prediction))):
		for scan_date in sorted(os.listdir(os.path.join(path_prediction, pat_id))):
			pred_list = os.listdir(os.path.join(path_prediction, pat_id, scan_date))
			if len(pred_list) == 5:
				#Ensemble
				pred_TPDM = np.load(os.path.join(path_prediction, pat_id, scan_date, "pred_TPDM1.npy"))
				pred = np.load(os.path.join(path_prediction, pat_id, scan_date, "pred_TPDM2.npy"))
				SUV = np.load(os.path.join(path_prediction, pat_id, scan_date, "SUV.npy"))
				CT = np.load(os.path.join(path_prediction, pat_id, scan_date, "CT.npy"))
				GT = np.load(os.path.join(path_prediction, pat_id, scan_date, "GT.npy"))

				if args.ensemble_type == "average":
					prediction = torch.from_numpy(pred_TPDM + pred).argmax(dim=0).numpy()

				elif args.ensemble_type == "union":
					pred1 = torch.from_numpy(pred_TPDM).argmax(dim=0).numpy()
					pred2 = torch.from_numpy(pred).argmax(dim=0).numpy()
					prediction = pred1 + pred2
					prediction = np.where(prediction == 2, 1, prediction)
			else:
				#Take the prediction of model_2
				print("Consider the prediction of model_2 from previous run")
				print(pat_id)
				pred = np.load(os.path.join(path_prediction, pat_id, scan_date, "pred_TPDM2.npy"))
				SUV = np.load(os.path.join(path_prediction, pat_id, scan_date, "SUV.npy"))
				CT = np.load(os.path.join(path_prediction, pat_id, scan_date, "CT.npy"))
				GT = np.load(os.path.join(path_prediction, pat_id, scan_date, "GT.npy"))
				prediction = torch.from_numpy(pred).argmax(dim=0).numpy()

			path_Lesions = os.path.join(path_Output, "CV_" + str(k), "Lesion_wise_Metrics")
			if not os.path.exists(path_Lesions):
				os.mkdir(path_Lesions)

			if not os.path.exists(os.path.join(path_Lesions, "union")):
				os.mkdir(os.path.join(path_Lesions, "union"))
			if not os.path.exists(os.path.join(path_Lesions, "average")):
				os.mkdir(os.path.join(path_Lesions, "average"))
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "MIPs", "union")):
				os.mkdir(os.path.join(path_Output, "CV_" + str(k), "MIPs", "union"))
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "MIPs", "average")):
				os.mkdir(os.path.join(path_Output, "CV_" + str(k), "MIPs", "average"))
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "Metrics", "union")):
				os.mkdir(os.path.join(path_Output, "CV_" + str(k), "Metrics", "union"))
			if not os.path.exists(os.path.join(path_Output, "CV_" + str(k), "Metrics", "average")):
				os.mkdir(os.path.join(path_Output, "CV_" + str(k), "Metrics", "average"))

			path_lesion_wise_analysis = os.path.join(path_Output, "CV_" + str(k), "Lesion_wise_Metrics", args.ensemble_type, str(pat_id) + "_" + str(scan_date) + ".txt") #Lesion wise analysis
			Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise = lesion_wise_analysis(path_lesion_wise_analysis, prediction, GT, SUV, CT, args.spacing, args.TP_threshold)

			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", args.ensemble_type, "metrics.txt")
			dice, fp, fn = compute_metrics(GT, prediction, pat_id, scan_date, path_dice, Total_Number_of_lesions, MTV_GT, MTV_pred, TP_lesion_wise, FN_lesion_wise)
			#DICE.append(dice)
			#Generate MIPs for the predictions
			#SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			#path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", "roi_" + str(roi_size[0]), str(pat_id) + "_" + str(scan_date) + "_" + str(dice) + "_" + str(fp) + "_" + str(fn) + ".jpg")
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", args.ensemble_type, str(pat_id) + "_" + str(scan_date) + ".jpg")
			overlay_segmentation(SUV, prediction, GT, path_MIP)

