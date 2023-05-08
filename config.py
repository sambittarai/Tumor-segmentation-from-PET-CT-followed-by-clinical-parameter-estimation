import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="Path containing all the Patient's Data.")
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/DataFrame_with_Paths/df_disease_wise.csv", help="Path containing all the Patient's Data (CT, SUV, SEG).")
    parser.add_argument("--MIP_directions", default=["coronal", "saggital"], help="Path containing all the Patient's Data.")

    parser.add_argument("--output_path", default="/media/sambit/HDD/Sambit/GitHub/Tumor Segmentation followed by Outcome Prediction/Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction/Data Preparation/Output", help="Path containing all the Patient's Data.")
    #parser.add_argument("--include_ids", default=["PETCT_0b98dbe00d", "PETCT_13b40a817b", "PETCT_30c4b7062b", "PETCT_39159c05c2", "PETCT_335a00191d", "PETCT_3b1c9155f5", "PETCT_5a58935b68", "PETCT_5e2da717db", "PETCT_7a77b26403", "PETCT_92c5c944a5", "PETCT_99a7bfad23", "PETCT_7948aa0e26", "PETCT_983a76fd43", "PETCT_9b982e72cb", "PETCT_bf178a41b2", "PETCT_b327726c24", "PETCT_ca58410fad", "PETCT_d4b2ff9721", "PETCT_e03b96666f", "PETCT_ef9d41b836", "PETCT_0cda25453b", "PETCT_1b1bdfc35b", "PETCT_11e258cc1f", "PETCT_1956667fce", "PETCT_2b60c8135a", "PETCT_30001118d0", "PETCT_55ca11402a", "PETCT_5d10be5b89", "PETCT_6a3477cd9a", "PETCT_8a52353a72", "PETCT_8ec13728df", "PETCT_685d7c09b5", "PETCT_901573a747", "PETCT_63508c679d", "PETCT_790246c76c", "PETCT_ac75e49284", "PETCT_ad7cd4a9d2", "PETCT_aa27cb9156", "PETCT_c29aba73da", "PETCT_b53ba7c6bf", "PETCT_ddbb3c69f0", "PETCT_ea42c88cc7", "PETCT_ea051a3e6c", "PETCT_fe705ea1cc", "PETCT_2f7200f771", "PETCT_4c75fa4a5d", "PETCT_4b688f46b0", "PETCT_a82f03863a", "PETCT_e252be4334"], help="Path containing all the Patient's Data.")

    parser.add_argument("--bone_HU", default=[200], help="Bone HU limit (x > +200)")
    parser.add_argument("--lean_HU", default=[-29, 150], help="Lean tissue HU limit (-29 < x < +150)")
    parser.add_argument("--adipose_HU", default=[-190, -30], help="Adipose tissue HU limit (-190 < x < -30)")
    parser.add_argument("--air_HU", default=[-191], help="Air HU limit (< -191)")

    parser.add_argument("--SUV_max_collage", default=15, help="Maximum SUV to be used during generation of collages for visualization")

    args = parser.parse_args()
    return args

