import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the autoPET Data.")
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_final.csv", help="DataFrame path which contains the full path of CT, SUV, SEG, and other clinical information such as age, sex, diagnosis.")
    parser.add_argument("--MIP_types", default=["coronal", "saggital"], help="Projections along the particular directions for the purpose of visualization of the multi-channel SUV and CT projections.")

    parser.add_argument("--path_multi_channel_3D_CT_SUV", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Data Preparation/Output/Multi_channel_3D_SUV_CT", help="Path where output will be generated.")
    parser.add_argument("--path_multi_angled_multi_channel_2D_projections", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Data Preparation/Output/Multi-angled_multi-channel_2D_projections", help="Multi-angled_multi-channel 2D projections.")

    parser.add_argument("--bone_HU", default=[200], help="Bone HU limit (x > +200)")
    parser.add_argument("--lean_HU", default=[-29, 150], help="Lean tissue HU limit (-29 < x < +150)")
    parser.add_argument("--adipose_HU", default=[-190, -30], help="Adipose tissue HU limit (-190 < x < -30)")
    parser.add_argument("--air_HU", default=[-191], help="Air HU limit (< -191)")

    parser.add_argument("--SUV_max_collage", default=14, help="Maximum SUV threshold to be used during generation of collages for the purpose of visualization")

    parser.add_argument("--rotation_min", default=-90, help="Starting angle of multi-angle multi-channel 2D projections.")
    parser.add_argument("--rotation_max", default=90, help="Ending angle of multi-angle multi-channel 2D projections.")
    parser.add_argument("--rotation_interval", default=90, help="Interval angle by which each of the projections will be rotated.")

    parser.add_argument("--CT_min", default=-100, help="Dummy.")
    parser.add_argument("--CT_max", default=250, help="Dummy.")
    parser.add_argument("--SUV_min", default=0, help="Minimum SUV ScaleIntensityRanged.")
    parser.add_argument("--SUV_max", default=15, help="Maximum SUV ScaleIntensityRanged.")

    args = parser.parse_args()
    return args

