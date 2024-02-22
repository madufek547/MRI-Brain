import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.signal import wiener
from skimage import feature, util
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import scipy.signal as signal
from mayavi import mlab
from scipy.ndimage import gaussian_filter
from mayavi import mlab
from mayavi import mlab
from tvtk.util import ctf
from tvtk.api import tvtk
from tvtk.util.ctf import PiecewiseFunction
from mayavi.modules.volume import Volume
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
from mayavi.modules.scalar_cut_plane import ScalarCutPlane
import cv2

#def create_ellipsoid(volume_shape, center, radii):
    # Create coordinates for the volume
 #   z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
    
    # Equation of an ellipsoid (x - a)^2/rx^2 + (y - b)^2/ry^2 + (z - c)^2/rz^2 = 1
  #  ellipsoid = (x - center[0])**2 / radii[0]**2 + (y - center[1])**2 / radii[1]**2 + (z - center[2])**2 / radii[2]**2 <= 1
   # return ellipsoid


def gamma_correction(image, correction):
    image = image/255.0
    image = cv2.pow(image, correction)
    return np.uint8(image*255)


def load_dicom_images(dicom_files):
    images = []
    for dicom_file_path in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file_path, force=True)  # Add force=True here
            if 'PixelData' in ds:
                image = ds.pixel_array
                image = image - np.min(image)
                image = image / np.max(image)
                image = (image * 255).astype(np.uint8)
                images.append(image)
            else:
                print(f"No pixel data found in {dicom_file_path}. Skipping file.")
        except Exception as e:
            print(f"Error processing {dicom_file_path}: {str(e)}")
    return images

def load_dicom_files(directory, image_prefix):
    dicom_file_paths = []
    for i in tqdm(range(1, 209), desc=f"Loading {image_prefix}"):
        filename = f"{image_prefix}{i:04}.dcm"
        dicom_file_path = os.path.join(directory, filename)
        dicom_file_paths.append(dicom_file_path)
    return dicom_file_paths


def convert_dicom_to_png(dicom_files, png_directory):
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)
    for dicom_file_path in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file_path, force=True)  # Add force=True here
            if 'PixelData' in ds:
                image = ds.pixel_array
                image = image - np.min(image)
                image = image / np.max(image)
                image = (image * 255).astype(np.uint8)
                im = Image.fromarray(image)
                png_filename = os.path.join(png_directory, os.path.basename(dicom_file_path).replace('.dcm','.png'))
                im.save(png_filename)
            else:
                print(f"No pixel data found in {dicom_file_path}. Skipping file.")
        except Exception as e:
            print(f"Error processing {dicom_file_path}: {str(e)}")

def dewow(data, window_length):
    """Remove low frequency trends using a high-pass filter."""
    # Create a window
    window = signal.windows.hann(window_length)
    # Reshape the window to make it 2D
    window = np.outer(window, window)
    # Convolve the window with the data
    filtered_data = signal.convolve(data, window, mode='same') / window.sum()
    return filtered_data


# Load PNG images from a directory
def load_images(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            k = 100
            # Perform dewowing
            image_array = dewow(image_array, k)

            images.append(image_array)
    return images

def trilinear_interpolation(volume, new_shape):
    mgrid = np.mgrid[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]]
    mgrid = mgrid.astype(np.float64)
    mgrid[0] *= volume.shape[0] / new_shape[0]
    mgrid[1] *= volume.shape[1] / new_shape[1]
    mgrid[2] *= volume.shape[2] / new_shape[2]
    return map_coordinates(volume, mgrid, order=1)

def remove_edges(volume):
    slices = [volume[:, :, i] for i in range(volume.shape[2])]
    masked_slices = []
    for slice in slices:
        edges = feature.canny(slice)
        mask = ndi.binary_dilation(edges, iterations=2)
        mask = util.invert(mask)
        masked_slice = np.where(mask, slice, 0)
        masked_slices.append(masked_slice)
    return np.stack(masked_slices, axis=-1)

directory = "ds004650/sourcedata/5"
image_prefix = "IM-0002-"

# Load the DICOM files from the directory
dicom_files = load_dicom_files(directory, image_prefix)

# Convert the DICOM files to PNG images


# Load the PNG images
images = load_dicom_images(dicom_files)
# Slice the images to include only the range from the start of the first ear to the end of the second ear
images = images[16:194]  # We add 1 to the end index because Python indexing is exclusive at the end
for image in images:
        volume = gamma_correction(image, 0.05)

volume = np.stack(images, axis=-1)  # Stack the remaining images into a 3D volume




# Perform trilinear interpolation
new_shape = (256, 256, 256)  # Example new shape
volume = trilinear_interpolation(volume, new_shape)

# Create a scalar field
# Create a scalar field

src = mlab.pipeline.scalar_field(volume)
# Create a volume module and add it to the pipeline
vol = mlab.pipeline.volume(src)
# Define the center of the ellipsoid

mlab.show()
