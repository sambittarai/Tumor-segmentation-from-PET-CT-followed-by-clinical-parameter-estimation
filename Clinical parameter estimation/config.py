import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    #parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the Patient's Data.")
    parser.add_argument("--data_path_MIPs", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Data Preparation/Output/Multi-angled_multi-channel_2D_projections", help="Path containing multi-channel PET/CT projections from different angles for all the patients.")
    parser.add_argument("--data_path_MIPs_collages", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Clinical parameter estimation/Data_Pre-processing/Output/MIPs_collages", help="Path containing multi-channel PET/CT collages of coronal and sagittal projections for all the patients.")

    parser.add_argument("--include_angles_collages", default=[0, 90], help="Only include the provided angles during the generation of collages.")

    #DataFrame
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_final.csv", help="DataFrame containing path of all the patients.")
    parser.add_argument("--path_df_rot_mips", default="/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/df_rot_mips.csv", help="DataFrame with rotating mips path.")
    parser.add_argument("--path_df_rot_mips_collages", default="/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Data_Preparation/Output/df_rot_mips_collages_512_512.csv", help="DataFrame with rotating mips path.")

    parser.add_argument("--rotation_min", default=-90, help="Minimum angle by which the MIP will be roated.")
    parser.add_argument("--rotation_max", default=90, help="Maximum angle by which the MIP will be rotated.")
    parser.add_argument("--rotation_interval", default=10, help="Degrees by which the MIP will be rotated each time.")

    parser.add_argument("--path_CV_Output", default="/media/sambit/HDD/Sambit/Projects/Project_6/Outcome_Prediction/Cross_Validation/Output/Classification/Disease_Type/Binary/Experiment_3", help="Cross Validation Output.")

    #Dataloader
    parser.add_argument("--batch_size_train", default=30, help="Batch Size for TrainLoader.") #(MONAI_UNET - 22), (UNET++ - 2)
    parser.add_argument("--batch_size_val", default=1, help="Batch Size for ValLoader.")
    parser.add_argument("--num_workers", default=4, help="Number of Workers.")
    #parser.add_argument("--CT_min", default=-100, help="Minimum CT ScaleIntensityRanged.")
    #parser.add_argument("--CT_max", default=250, help="Maximum CT ScaleIntensityRanged.")
    parser.add_argument("--SUV_min", default=0, help="Minimum SUV ScaleIntensityRanged.")
    parser.add_argument("--SUV_max", default=15, help="Maximum SUV ScaleIntensityRanged.")
    parser.add_argument("--pos", default=1, help="Data Sampling (Pos to Neg) Ratio for tumor class.")
    parser.add_argument("--neg", default=1, help="Data Sampling (Pos to Neg) Ratio for background class.")
    parser.add_argument("--num_samples", default=1, help="Number of 3D Samples extracted from a single patient during Training.") #(MONAI_UNET - 16), (UNET++ - 14)

    parser.add_argument("--remove_pat_IDs_sex", default=["PETCT_605369e88d", "PETCT_9d6699f215"], help="Batch Size for TrainLoader.") #Male: 1; Female: 0.

    #parser.add_argument("--regression_type", default=["age"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.
    #parser.add_argument("--regression_type", default=["MTV"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.
    #parser.add_argument("--regression_type", default=["lean_volume"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.
    parser.add_argument("--regression_type", default=["lesion_count"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.

    #parser.add_argument("--classification_type", default=["sex"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.
    parser.add_argument("--classification_type", default=["diagnosis"], help="What outcome are you trying to predict.") #Male: 1; Female: 0.

    #Network Architecture
    parser.add_argument("--dimensions", default=2, help="Dimension of the UNET.")
    parser.add_argument("--in_channels", default=5, help="Total number of input channels.")
    parser.add_argument("--num_classes_sex", default=2, help="Number of segmentation classes.")
    parser.add_argument("--num_classes_age", default=1, help="Number of segmentation classes.")
    parser.add_argument("--dropout", default=0.2, help="Dropout Percentage.")
    parser.add_argument("--lr", default=1e-3, help="Learning Rate.")
    parser.add_argument("--weight_decay", default=1e-4, help="Weight Decay.")
    parser.add_argument("--momentum", default=0.9, help="Momentum used in SGD.")

    #Training Parameters
    parser.add_argument("--max_epochs", default=50, help="Maximum Number of Epochs for Training.")
    parser.add_argument("--validation_interval", default=1, help="The interval after which you want to perform validation.")
    parser.add_argument("--best_metric_classification", default=0, help="Best Dice Metric.")
    parser.add_argument("--best_metric_regression", default=0, help="Best Dice Metric.")


    parser.add_argument("--best_metric_epoch", default=-1, help="Epoch corresponding to Best Dice Metric.")
    #parser.add_argument("--roi_size", default=(208,208), help="Image Patch used for training.")
    parser.add_argument("--K_fold_CV", default=10, help="K fold Cross Validation.")

    parser.add_argument("--pre_trained_weights", default=False, help="If True, then load the pretrained weights to the network.")

    args = parser.parse_args()
    return args
