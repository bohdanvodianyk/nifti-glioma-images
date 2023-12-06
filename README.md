# NifTi Image Processing and Conversion

This repository contains a Python script for processing and converting NifTi medical images into various formats including DICOM, PNG, JPG, and GIF. It is particularly designed for researchers and practitioners working with brain imaging data, with a focus on glioma cases.

## Data Source

The script is developed and tested using the [Glioma Brain Tumor Imaging Data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830) from The Cancer Imaging Archive (TCIA). This dataset provides a comprehensive set of brain tumor MRI scans, which is crucial for advancing research in medical imaging and computational diagnosis.

## Features

- **DICOM Conversion**: Convert NifTi images to DICOM format.
- **Image Normalization**: Normalize image matrices for consistent processing.
- **Image Format Conversion**: Convert images to PNG and JPG formats.
- **Composite Image Creation**: Generate composite images from multiple slices.
- **Tumor Segmentation**: Apply segmentation algorithms to identify tumors.
- **GIF Creation**: Create animated GIFs from image slices.

## Usage

1. Install required libraries:

pip install nibabel pandas numpy matplotlib pydicom pillow imageio opencv-python
2. Configure paths and parameters in 'config.json' and 'dicom_initial_parameters.json'.
3. Run the script:
python nifti_conversion.py

## Requirements

- Python 3.x
- Libraries: nibabel, pandas, numpy, matplotlib, pydicom, PIL, json, imageio, cv2

## Contributions

Contributions are welcome. Please open an issue or submit a pull request.

## Contact

For any queries or feedback, please raise an issue in the repository.

