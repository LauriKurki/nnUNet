import os
import tqdm
import numpy as np
import tensorflow as tf

import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk


def get_dataloader(folder: str):
    files = sorted(os.listdir(folder))
    files = [
        os.path.join(folder, f)
        for f in files
        if f.startswith("afms_")
    ]

    # Get the element spec of the first file.
    element_spec = tf.data.Dataset.load(files[0]).element_spec
    ds = tf.data.Dataset.from_tensor_slices(files)
    # Shuffle the files.
    ds = ds.interleave(
        lambda path: tf.data.Dataset.load(path, element_spec=element_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Compute the segmentation map.
    ds = ds.map(
        compute_segmentation_map,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch the data.
    ds = ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    return ds

def compute_segmentation_map(sample):
    image = sample["x"]
    xyz = sample["xyz"]
    sw = sample["sw"]

    # The image has to be transposed.
    image = tf.transpose(image, [1, 0, 2])

    z_max = tf.math.reduce_max(xyz[:, 2])
    x = tf.linspace(sw[0,0], sw[1,0], 128)
    y = tf.linspace(sw[0,1], sw[1,1], 128)
    z = tf.linspace(z_max, z_max-1.0, 10)

    # Create a meshgrid.
    X, Y, Z = tf.meshgrid(x, y, z, indexing='xy')

    # Create segmentation map.
    seg = tf.zeros_like(X)

    for atom in xyz:
        # Skip padding
        if atom[-1] == 0:
            break

        # Compute the distance to the atom.
        m = (X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2
        m = tf.where(m < 0.2, atom[-1], 0.)

        # Add the atom to the segmentation map.
        seg += m

    return {
        "x": image,
        "seg": seg,
        "sw": sw,
        "xyz": xyz
    }


def convert_tensorflow_afm(tensorflow_dir: str, dataset_id: int):
    task_name = "AFMRebias10"

    ds = get_dataloader(tensorflow_dir)

    target_dataset_name = f'Dataset{dataset_id:03}_{task_name}'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for i, sample in tqdm.tqdm(enumerate(ds)):
        image = sample["x"]
        seg = sample["seg"]

        # Cast image to float16 and seg to uint8
        image = image.astype(np.float32)
        seg = seg.astype(np.uint8)

        image_path = os.path.join(imagesTr, f"afm_{i+1:06d}_0000.nii.gz")
        seg_path = os.path.join(labelsTr, f"seg_{i+1:06d}.nii.gz")

        nib.save(nib.Nifti1Image(image, affine=None), image_path)
        nib.save(nib.Nifti1Image(seg, affine=None), seg_path)

    labels = {
        "background": 0,
        "H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    },
    n_samples = len(
        os.listdir(image_path)
    )

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'AFM'},
        labels,
        n_samples,
        '.nii.gz',
        None,
        target_dataset_name,
        #overwrite_image_reader_writer='NibabelIOWithReorient',
        #reference='https://aortaseg24.grand-challenge.org/',
        #license='see ref'
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_folder',
        type=str,
        help="The generated AFM dataset in TF format (must have afms_XXXXX subfolders)"
    )
    parser.add_argument('-d', required=False, type=int, default=300, help='nnUNet Dataset ID, default: 300')
    args = parser.parse_args()
    tf_base = args.input_folder
    convert_tensorflow_afm(tf_base, args.d)
