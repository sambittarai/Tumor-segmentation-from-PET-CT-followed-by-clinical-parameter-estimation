# Tumor-segmentation-from-PET-CT-followed-by-outcome-prediction
The overall framework has two primary modules: [1] Tumor segmentation from PET/CT using segmentation prior; [2] Outcome prediction from PET/CT 2D projections.

[1] Original_Data: This repository contains the original autoPET data. The directory contains a list of patient which then includes a list of scan dates for each of the patient. The directory structure looks like this:
   Patient_ID
      Scan_Date
        #### CT.nii.gz (Original CT scan)
        #### CTres.nii.gz (Resampled CT scan)
        #### SUV.nii.gz (Resampled SUV scan)
        #### SEG.nii.gz (Tumor segmentation)
