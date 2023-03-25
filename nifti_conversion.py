import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import pydicom
import os
from PIL import Image
import json
import imageio
import cv2


def dicom_convert_save(nifti_data, output_dir, file_name, index=0, format='uint16'):
    """
    Function that convert NifTi image to DICOM.

    :param nifti_data: numpy array, multidimensional matrix of nifti data
    :param output_dir: str, file path to save output
    :param file_name: str, tha name that file will have
    :param index: int, the number of slice in nifti data
    :param format: str, format of data to perform the number of shades of gray
    """

    matrix = normilize_matrix(nifti_data[:, :, index], format=format)
    dicom_file = pydicom.Dataset.from_json(json.loads(open("dicom_initial_parameters.json").read()))
    '''
    dicom_initaial_parameters.json consists all necessary values to create base DICOM file.
    Parameters can be changed according to the work and image processing requirements.
    '''
    # change dicom-dataset pararmeters
    arr = matrix.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.is_little_endian = True
    dicom_file.is_implicit_VR = False
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    
    # save dicom
    dicom_file.save_as(os.path.join(output_dir, f'{file_name}_slice_{index}.dcm'))


def normilize_matrix(matrix, format='uint16'):
    """
    Function that normalize the matrix according to the chosen format of data type.

    :param matrix: numpy array, initial matrix
    :param format: str, data type (uint8 or uint16)
    :return: numpy array, normalized matrix
    """
    max_matrix = np.max(matrix)
    if max_matrix != 0:
        if format == 'uint16':
            norm_matrix = ((matrix / max_matrix) * (np.iinfo('uint16').max - 1)).astype(int)
        elif format == 'uint8':
            norm_matrix = ((matrix / max_matrix) * (np.iinfo('uint8').max - 1)).astype(int)
        else:
            print('Error! Format: uint16 or uint8!')
            return 1
    else:
        norm_matrix = matrix
    return norm_matrix


def convert_to_png(nifti_data, output_dir, file_name, index=0):
    """
    Save a png file from the nifti data in uint16 format.

    :param nifti_data: numpy array, multidimensional matrix of nifti data
    :param output_dir: str, file path to save output
    :param file_name: str, tha name that file will have
    :param index: int, the number of slice in nifti data
    """
    image = np.uint16(normilize_matrix(nifti_data[:, :, index], format='uint16'))
    final_image = Image.fromarray(image)
    final_image.save(os.path.join(output_dir, f'{file_name}_slice_{index}.png'))


def convert_to_jpg(nifti_data, output_dir, file_name, index=0):
    """
    Save a jpg file from the nifti data in uint8 format.

    :param nifti_data: numpy array, multidimensional matrix of nifti data
    :param output_dir: str, file path to save output
    :param file_name: str, the name that file will have
    :param index: int, the number of slice in nifti data
    """
    image = np.uint8(normilize_matrix(nifti_data[:, :, index], format='uint8'))
    final_image = Image.fromarray(image)
    final_image.save(os.path.join(output_dir, f'{file_name}_slice_{index}.jpg'))


def all_slices_image(nifti_data, output_dir, file_name, tumor_segmentation=False, index=0):
    """
    Function that create a figure with all chosen slices that described in config file by start/stop slice.

    :param nifti_data: numpy array, multidimensional matrix of nifti data
    :param output_dir: str, file path to save output
    :param file_name: str, tha name that file will have
    :param tumor_segmentation: bool, apply segmentation algorithm
    :param index: int, the number of slice in nifti data
    """
    global config

    # define the limits of slice
    start_slice = config.get('start_slice')
    stop_slice = config.get('stop_slice')

    # normalization of matrix
    matrix_3d_img = np.uint16(normilize_matrix(nifti_data, format='uint16'))

    # define numbers of figures in size of total image
    size_1 = round(np.sqrt(stop_slice - start_slice))
    size_2 = math.ceil((stop_slice - start_slice) / size_1)

    # plot the figures
    fig, axs = plt.subplots(size_1, size_2, figsize=[15, 15])
    for idx, img in enumerate(range(start_slice, stop_slice, 1)):
        axs.flat[idx].imshow(matrix_3d_img[:, :, img], cmap='gray')
        axs.flat[idx].axis('off')
    plt.suptitle(f'{file_name}\nStart={start_slice}, Stop={stop_slice}')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{file_name}.png'), dpi=500)
    plt.close(fig)

    # segmentation preprocessing of images
    if (tumor_segmentation == True) and (file_name.endswith('FLAIR')):
        # save image with segmentation to general folder of patient
        output_dir = output_dir.split('/')[0] + '/' + output_dir.split('/')[1] + '/'
        fig, axs = plt.subplots(size_1, size_2, figsize=[15, 15])
        for idx, img in enumerate(range(start_slice, stop_slice, 1)):
            image_gray = matrix_3d_img[:, :, img]
            # set the low threshold for tumor segmentation
            # according to the task we can use madian+std or mean+std
            threshold_gray = np.median(image_gray[image_gray > 0]) + np.std(image_gray[image_gray > 0])
            # threshold_gray = np.mean(image_gray[image_gray > 0]) + np.std(image_gray[image_gray > 0])
            # obtain whole brain and tumor segments
            brain = segmentation_tumor_brain(image_gray, threshold_gray=1.0)
            tumor = segmentation_tumor_brain(image_gray, threshold_gray=threshold_gray)
            segmentation_result = brain + tumor
            # plot results of segmentation
            axs.flat[idx].imshow(segmentation_result, cmap='gray')
            axs.flat[idx].axis('off')
            if img == index:
                slice_image = np.uint8(segmentation_result)
                slice_final = Image.fromarray(slice_image)
                slice_final.save(os.path.join(output_dir, f'{file_name}_slice_{index}_segmentation.png'))
        plt.suptitle(f'{file_name}\nStart={start_slice}, Stop={stop_slice}')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{file_name}_segmentation.png'), dpi=500)
        plt.close(fig)

def segmentation_tumor_brain(image_gray, threshold_gray=1.0):
    """
    Function that perform segmentation for the special threshold.

    :param image_gray: numpy array, matrix of the image in gray shades
    :param treshhold_gray: float, low limit (threshold) of gray
    :return: numpy array, matrix with segmentation according to the threshold
    """

    # applying threshold separating
    (_, thresh) = cv2.threshold(image_gray, threshold_gray, 255, cv2.THRESH_BINARY)
    # perfom morphological filter to clear small areas and pixels
    thresh_segment = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # perform Gaussian blur to make edges soft and clear segment
    blur = cv2.GaussianBlur(thresh_segment, (7, 7), cv2.BORDER_WRAP)
    segment = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)[1]
    # set to the segment half of gray maximum (255)
    segment[segment == 255] = 125

    return segment


def main_data_generation():
    print("--- NifTi Images Convertation and Tumor Segmentation ---")
    global config
    config = json.loads(open("config.json").read())
    data_dir_path = config.get("data_dir")
    output_dir_path = config.get("output_dir")
    index = config.get("slice_index")
    clinical_info = pd.read_csv('UCSF-PDGM-metadata.csv')

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Get the list of all files in directory tree at given path
    files = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir_path):
        files += [os.path.join(dirpath, file).replace('\\', '/') for file in filenames if file.endswith('.gz')]
        # files += [os.path.join(dirpath, file).replace('\\', '/') for file in filenames if file.endswith('.nii')]
    patients_ids = [f.split("/")[-2].split('_')[0] for f in files]
    file_names = [f.split("/")[-1][:-7] for f in files]
    # genders = [clinical_info[clinical_info['ID']==patient]['Sex'] for patient in patients_ids]
    # ages = [clinical_info[clinical_info['ID']==patient]['Age at MRI'] for patient in patients_ids]
    data_list = [X for X in zip(files, patients_ids, file_names)]
    for file in data_list:
        conversion_data(file, index=index, to_png=True, to_jpg=True, create_gif=True)
    print('Main data generation is done!')


def conversion_data(file, index=0, to_png=False, to_jpg=False, create_gif=False):
    """
    Function that do all main conversions.

    :param file: list str, str list with information about file
    :param index: int, the number of slice in NifTi data
    :param to_png: bool, converse slice to png
    :param to_jpg: bool, converse slice to jpg
    :param create_gif: bool, create a gif video with all slices from NifTi
    """

    global config
    config = json.loads(open("config.json").read())
    file_path, patient_id, file_name = file
    print(f'{file_name} is processing')
    format = config.get("format")
    output_dir_path = config.get("output_dir") + patient_id + '/' + file_name + '/'

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    nifti_data = nib.load(file_path).get_fdata()

    dicom_convert_save(nifti_data, output_dir_path, file_name, index=index, format=format)
    all_slices_image(nifti_data, output_dir_path, file_name, tumor_segmentation=True, index=index)
    if to_png:
        convert_to_png(nifti_data, output_dir_path, file_name, index=index)
    if to_jpg:
        convert_to_jpg(nifti_data, output_dir_path, file_name, index=index)
    if create_gif:
        gif_from_nifti(nifti_data, output_dir_path, file_name)


def gif_from_nifti(nifti_data, output_dir, file_name):
    """
    Function that create the gif video with all slices from NifTi data file.

    :param nifti_data: numpy array, multidimensional matrix of nifti data
    :param output_dir: str, file path to save output
    :param file_name: str, tha name that file will have
    """
    images = []
    for i in range(nifti_data.shape[2]):
        image = np.uint8(normilize_matrix(nifti_data[:, :, i], format='uint8'))
        final_image = Image.fromarray(image)
        images.append(final_image)
    imageio.mimsave(f'{output_dir}/{file_name}.gif', images)


if __name__ == '__main__':
    main_data_generation()
    print('Done!')
