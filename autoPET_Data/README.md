# autoPET data directory

This directory contains autoPET data in the Data folder and the corresponding dataFrame (with absolute paths for CTres, SUV, SEG along with other clinical parameters such as age, diagnosis, sex) in the DataFrame folder.

## autoPET_Data
* In order to run the code successfully download your own copy of the data from the autoPET website (https://autopet.grand-challenge.org/Dataset/) and put it in this folder. Note that it follows the same structure as the original.
* The data directory structure looks something like this:
  - Patient ID
    - Scan Date
      - CT.nii.gz
      - CTres.nii.gz
      - SUV.nii.gz
      - PET.nii.gz
      - SEG.nii.gz
