# Tumor-segmentation-from-PET-CT-followed-by-clinical-parameter-estimation
The overall framework has two primary modules: [1] Tumor segmentation from PET/CT using segmentation prior; [2] Clinical parameter estimation (Regression and Classification) from multi-channel PET/CT projections.

## Directories
[1] Original_Data: This repository contains the original autoPET data. The directory contains a list of patient which then includes a list of scan dates for each of the patient. The directory structure looks like this:
   - Patient_ID
      - Scan_Date
         - CT.nii.gz (Original CT scan)
         - CTres.nii.gz (Resampled CT scan)
         - SUV.nii.gz (Resampled SUV scan)
         - SEG.nii.gz (Tumor segmentation)

## Files
[1] config.py: Contains information about all the hyperparameters.
[2] utils.py: Contains all the helper functions.


## Follow the steps below to run your own tumor segmentation network

## Follow the steps below to run your own outcome prediction network
* First make sure all the PET/CT autoPET data is placed in the correct folder (named autoPET_Data).
* Make sure to update all the paths (set them to your local paths) and other related hyperparameters in the config.py file. Note that after this you don't need to change any other hyperparameters elsewhere in the pipeline.
* Go the folder named "Data Preparation" and run "multi_channel_SUV_CT_generation.py" 
