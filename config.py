import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the autoPET Data.")
    parser.add_argument("--df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_final.csv", help="DataFrame path which contains the full path of CT, SUV, SEG, and other clinical information such as age, sex, diagnosis.")
    parser.add_argument("--MIP_types", default=["coronal", "saggital"], help="Projections along the particular directions for the purpose of visualization of the multi-channel SUV and CT projections.")

    parser.add_argument("--output_path", default="/media/sambit/HDD/Sambit/Projects/Project_5/GitHub/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Data Preparation/Output", help="Path where output will be generated.")

    parser.add_argument("--bone_HU", default=[200], help="Bone HU limit (x > +200)")
    parser.add_argument("--lean_HU", default=[-29, 150], help="Lean tissue HU limit (-29 < x < +150)")
    parser.add_argument("--adipose_HU", default=[-190, -30], help="Adipose tissue HU limit (-190 < x < -30)")
    parser.add_argument("--air_HU", default=[-191], help="Air HU limit (< -191)")

    parser.add_argument("--SUV_max_collage", default=14, help="Maximum SUV threshold to be used during generation of collages for the purpose of visualization")

    args = parser.parse_args()
    return args

