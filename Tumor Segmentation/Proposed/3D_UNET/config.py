import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the Patient's Data.")
    #Train Data Path
    #parser.add_argument("--data_path_train", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/Data_Subset/Train", help="Path containing all the Training Data.")
    #Validation Data Path
    #parser.add_argument("--data_path_val", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/Data_Subset/Validation", help="Path containing all the validation Data.")

    #Filtered DataFrame
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df.csv", help="DataFrame containing path of all the patients.")
    parser.add_argument("--path_df_patients_with_tumors", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df_patients_with_tumors.csv", help="DataFrame containing the patients' with tumors.")
    parser.add_argument("--path_df_patients_without_tumors", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df_patients_without_tumors.csv", help="DataFrame containing the patients' without tumors.")
    parser.add_argument("--path_df_disease_wise", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_disease_wise.csv", help="DataFrame containing the patients' without tumors.")

    ##################################Cross_Validation

    #Dataloader
    parser.add_argument("--batch_size_train", default=1, help="Batch Size for TrainLoader.")
    parser.add_argument("--batch_size_val", default=1, help="Batch Size for ValLoader.")
    parser.add_argument("--num_workers", default=4, help="Number of Workers.")
    parser.add_argument("--CT_min", default=-100, help="Minimum CT ScaleIntensityRanged.")
    parser.add_argument("--CT_max", default=250, help="Maximum CT ScaleIntensityRanged.")
    parser.add_argument("--SUV_min", default=0, help="Minimum SUV ScaleIntensityRanged.")
    parser.add_argument("--SUV_max", default=15, help="Maximum SUV ScaleIntensityRanged.")
    parser.add_argument("--pos", default=4, help="Data Sampling (Pos to Neg) Ratio for tumor class.")
    parser.add_argument("--neg", default=1, help="Data Sampling (Pos to Neg) Ratio for background class.")
    parser.add_argument("--num_samples", default=2, help="Number of 3D Samples extracted from a single patient during Training.")


    #Network Architecture
    parser.add_argument("--dimensions", default=3, help="Dimension of the UNET.")
    parser.add_argument("--in_channels", default=3, help="Total number of input channels.")
    parser.add_argument("--out_channels", default=2, help="Number of segmentation classes.")
    parser.add_argument("--dropout", default=0.2, help="Dropout Percentage.")
    parser.add_argument("--lr", default=1e-4, help="Learning Rate.")
    parser.add_argument("--weight_decay", default=1e-5, help="Weight Decay.")
    parser.add_argument("--momentum", default=0.99, help="Momentum used in SGD.")

    #Training Parameters
    parser.add_argument("--max_epochs", default=2000, help="Maximum Number of Epochs for Training.")
    parser.add_argument("--validation_interval", default=1, help="The interval after which you want to perform validation.")
    parser.add_argument("--validation_start", default=0, help="The interval after which you want to perform validation.")
    parser.add_argument("--best_metric", default=0, help="Best Dice Metric.")
    parser.add_argument("--best_metric_epoch", default=-1, help="Epoch corresponding to Best Dice Metric.")
    parser.add_argument("--roi_size", default=(160,160,160), help="Image Patch used for training.")
    parser.add_argument("--K_fold_CV", default=5, help="K fold Cross Validation.")

    parser.add_argument("--pre_trained_weights", default=False, help="If True, then load the pretrained weights to the network.")

    #Save Paths
    parser.add_argument("--path_CV_Output", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA", help="Output Path.")


    ##################################Inference
    parser.add_argument("--roi_size_val", default=(160,160,160), help="Image patches during validation.")
    parser.add_argument("--TP_threshold", default=0.0001, help="Threshold for DICE score used during tumor detection.")
    parser.add_argument("--spacing", default=(2.0364201068878174, 2.0364201068878174, 3.0), help="Threshold for DICE score used during tumor detection.")
    parser.add_argument("--path_Inference_Output", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Inference/Output/dynUNET/SUV_MIP_only/LYMPHOMA", help="Inference Path.")

    ##################################False_Postive_Analsis
    parser.add_argument("--path_FP_Analysis_Output", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/2D_rotating_MIPs_+_3D_UNET/False_Positive_Analysis_by_Alexander/Output", help="False Positive Analysis.")

    ##################################Ensembling
    parser.add_argument("--path_Ensembling_Output", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/2D_rotating_MIPs_+_3D_UNET/Ensembling/Output/Experiment_2", help="Inference Path.")
    #parser.add_argument("--ensemble_type", default="union", help="Type of Ensembling.")
    parser.add_argument("--ensemble_type", default="union", help="Type of Ensembling.")

    #parser.add_argument("--mean_liver_SUV_uptake", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Intensity_Normalization/wrt_Liver/Save_Path/Intensity_Normalization_Factors.txt", help="Mean values of liver uptake as per Liver Segmentation Model.")


    #3D UNET
    #Best Network Weights (This will be same for "Inference", "Ensembling")
    parser.add_argument("--path_checkpoint_CV_0", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_0/Network_Weights/best_model_78.pth.tar", help="Best Model for CV_0.")
    parser.add_argument("--path_checkpoint_CV_1", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_1/Network_Weights/best_model_87.pth.tar", help="Best Model for CV_1.")
    parser.add_argument("--path_checkpoint_CV_2", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_2/Network_Weights/best_model_99.pth.tar", help="Best Model for CV_2.")
    parser.add_argument("--path_checkpoint_CV_3", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_3/Network_Weights/best_model_158.pth.tar", help="Best Model for CV_3.")
    parser.add_argument("--path_checkpoint_CV_4", default="/media/sambit/HDD/Sambit/Projects/Project_5/Framework/Proposed/3D_UNET/Cross_Validation/Output/DynUNET/SUV_MIP_only/LYMPHOMA/CV_4/Network_Weights/best_model_192.pth.tar", help="Best Model for CV_4.")

    args = parser.parse_args()
    return args
