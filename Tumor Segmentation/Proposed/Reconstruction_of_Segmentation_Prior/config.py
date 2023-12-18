import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the Patient's Data.")
    parser.add_argument("--path_data_MIPs", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Data_preparation/Output/Multi_directional_2D_MIPs", help="Path to generate the rotating MIPs for all the Patients.")

    #Filtered DataFrame
    #parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_disease_wise.csv", help="DataFrame containing path of all the patients.")
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_final.csv", help="DataFrame containing path of all the patients.")

    parser.add_argument("--path_df_2D_MIPs", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Data_preparation/Output/df.csv", help="DataFrame containing the patients' without tumors.")

    parser.add_argument("--rotation_min", default=-45, help="Minimum angle by which the MIP will be roated.")
    parser.add_argument("--rotation_max", default=45, help="Maximum angle by which the MIP will be rotated.")
    parser.add_argument("--rotation_interval", default=90, help="Degrees by which the MIP will be rotated each time.")

    #parser.add_argument("--path_CV_Output", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/2D_UNET_rotating_MIPs/Cross_Validation/Full_Image_as_Input/Output/5_Fold_CV_Full_Image_Pad_UNET_1_channel_10_degrees", help="Cross Validation Output.")

    #Dataloader
    parser.add_argument("--batch_size_train", default=2, help="Batch Size for TrainLoader.") #(MONAI_UNET - 22), (UNET++ - 2)
    parser.add_argument("--batch_size_val", default=1, help="Batch Size for ValLoader.")
    parser.add_argument("--num_workers", default=0, help="Number of Workers.")
    parser.add_argument("--CT_min", default=-100, help="Minimum CT ScaleIntensityRanged.")
    parser.add_argument("--CT_max", default=250, help="Maximum CT ScaleIntensityRanged.")
    parser.add_argument("--SUV_min", default=0, help="Minimum SUV ScaleIntensityRanged.")
    parser.add_argument("--SUV_max", default=15, help="Maximum SUV ScaleIntensityRanged.")
    parser.add_argument("--pos", default=1, help="Data Sampling (Pos to Neg) Ratio for tumor class.")
    parser.add_argument("--neg", default=1, help="Data Sampling (Pos to Neg) Ratio for background class.")
    parser.add_argument("--num_samples", default=12, help="Number of 3D Samples extracted from a single patient during Training.") #(MONAI_UNET - 16), (UNET++ - 14)

    #Network Architecture
    parser.add_argument("--dimensions", default=2, help="Dimension of the UNET.")
    parser.add_argument("--in_channels", default=3, help="Total number of input channels.")
    parser.add_argument("--out_channels", default=2, help="Number of segmentation classes.")
    parser.add_argument("--dropout", default=0.2, help="Dropout Percentage.")
    parser.add_argument("--lr", default=1e-4, help="Learning Rate.")
    parser.add_argument("--weight_decay", default=1e-5, help="Weight Decay.")
    parser.add_argument("--momentum", default=0.99, help="Momentum used in SGD.")

    #Training Parameters
    parser.add_argument("--max_epochs", default=10000, help="Maximum Number of Epochs for Training.")
    parser.add_argument("--validation_interval", default=1, help="The interval after which you want to perform validation.")
    parser.add_argument("--best_metric", default=0, help="Best Dice Metric.")
    parser.add_argument("--best_metric_epoch", default=-1, help="Epoch corresponding to Best Dice Metric.")
    parser.add_argument("--roi_size", default=(208,208), help="Image Patch used for training.")
    parser.add_argument("--K_fold_CV", default=5, help="K fold Cross Validation.")

    parser.add_argument("--pre_trained_weights", default=False, help="If True, then load the pretrained weights to the network.")

    #############################Inference
    parser.add_argument("--path_inference_Output", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Tumor Segmentation/Proposed/Reconstruction_of_Segmentation_Prior/Output/Lymphoma", help="Inference Path.")
    parser.add_argument("--path_TPDMs", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Tumor Segmentation/Proposed/Reconstruction_of_Segmentation_Prior/Output/Segmentation_Prior", help="Inference Path.")
 
    #Best Model's Path
    parser.add_argument("--path_checkpoint_CV_0", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/LYMPHOMA/CV_0/Network_Weights/best_model_13.pth.tar", help="CV_0's best model.")
    parser.add_argument("--path_checkpoint_CV_1", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/LYMPHOMA/CV_1/Network_Weights/best_model_22.pth.tar", help="CV_1's best model.")
    parser.add_argument("--path_checkpoint_CV_2", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/LYMPHOMA/CV_2/Network_Weights/best_model_89.pth.tar", help="CV_2's best model.")
    parser.add_argument("--path_checkpoint_CV_3", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/LYMPHOMA/CV_3/Network_Weights/best_model_44.pth.tar", help="CV_3's best model.")
    parser.add_argument("--path_checkpoint_CV_4", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/2D_UNET/Cross_Validation/Output/UNET++/LYMPHOMA/CV_4/Network_Weights/best_model_91.pth.tar", help="CV_4's best model.")



    args = parser.parse_args()
    return args
