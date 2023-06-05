This directory contains two main folders:

[1] Data: This contains the original autoPET dataset in the following order:
   - Patient_ID
      - Scan_Date
         - CT.nii.gz (Original CT scan)
         - CTres.nii.gz (Resampled CT scan)
         - SUV.nii.gz (Resampled SUV scan)
         - SEG.nii.gz (Tumor segmentation)
         
[2] DataFrames: This contains a dataframe whose columns are:
- pat_ID
- scan_date
- CT (path)
- SUV (path)
- SEG (path)
