import os
import shutil
import tarfile
import h5py
import multiprocessing as mp
import numpy as np
import tqdm.contrib.concurrent

import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from typing import List, Tuple


def compute_segmentation_map(sample):
    image = sample["x"]
    xyz = sample["xyz"]
    sw = sample["sw"]

    # The image has to be transposed.
    image = np.transpose(image, [1, 0, 2])

    z_max = np.max(xyz[:, 2])
    x = np.linspace(sw[0,0], sw[1,0], 128)
    y = np.linspace(sw[0,1], sw[1,1], 128)
    z = np.linspace(z_max, z_max-1.0, 10)

    # Create a meshgrid.
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    # Create segmentation map.
    seg = np.zeros_like(X)

    for atom in xyz:
        # Skip padding
        if atom[-1] == 0:
            break

        # Compute the distance to the atom.
        m = (X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2
        m = np.where(m < 0.2, atom[-1], 0.)

        # Add the atom to the segmentation map.
        seg += m

    return {
        "x": image,
        "seg": seg,
        "sw": sw,
        "xyz": xyz
    }

def generate_nnunet_samples(
    input_file: str,
    output_dir: str,
    indices: List[Tuple[str, int]],
    counter_indices: List[int],
    start: int,
    end: int,
):
    """Generate a chunk of nnUNet samples from the AFM dataset."""
    image_tar_path = os.path.join(output_dir, f"image_chunk_{start:06d}_{end:06d}.tar")
    label_tar_path = os.path.join(output_dir, f"label_chunk_{start:06d}_{end:06d}.tar")

    # Create a tmp directory for the images and labels
    image_tmp_folder = os.path.join(output_dir, 'imagesTr', str(start))
    label_tmp_folder = os.path.join(output_dir, 'labelsTr', str(start))
    maybe_mkdir_p(image_tmp_folder)
    maybe_mkdir_p(label_tmp_folder)

    for i, (mode, h5_idx) in enumerate(indices[start:end]):
        with h5py.File(input_file, 'r') as f:
            sample = {
                "x": f[mode]['X'][h5_idx],
                "xyz": f[mode]['xyz'][h5_idx],
                "sw": f[mode]['sw'][h5_idx],
            }

        sample = compute_segmentation_map(sample)

        # Convert sample into nifti format
        x = sample["x"]
        seg = sample["seg"]

        x = x.astype(np.float32)
        seg = seg.astype(np.uint8)

        x_path = os.path.join(image_tmp_folder, f"afm_{i:06d}_0000.nii.gz")
        seg_path = os.path.join(label_tmp_folder, f"afm_{i:06d}.nii.gz")

        # Save the images
        nib.save(nib.Nifti1Image(x, affine=None), x_path)
        nib.save(nib.Nifti1Image(seg, affine=None), seg_path)

    # Create tar files for the images and labels
    with tarfile.open(image_tar_path, "w") as tar:
        tar.add(image_tmp_folder, arcname=os.path.basename(image_tmp_folder))
    with tarfile.open(label_tar_path, "w") as tar:
        tar.add(label_tmp_folder, arcname=os.path.basename(label_tmp_folder))

    # Remove the tmp folders
    shutil.rmtree(image_tmp_folder, ignore_errors=True)
    shutil.rmtree(label_tmp_folder, ignore_errors=True)


def generate_nnunet_samples_wrapper(args):
    """Wrapper for generate_nnunet_samples."""
    generate_nnunet_samples(*args)


def main(args):
    input_file = args.input_file
    num_workers = args.num_workers
    chunk_size = args.chunk_size
    dataset_id = args.d

    labels = {
        "background": 0,
        "H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    }

    # Get dataset sizes
    modes = ["train", "val", "test"]
    mode_sizes = {}
    for mode in modes:
        with h5py.File(input_file, 'r') as f:
            mode_sizes[mode] = len(f[mode]['X'])
    total_size = sum(mode_sizes.values())
    print(f"Dataset {dataset_id}: {total_size} samples")

    # Create a list of tuples for each sample (mode, h5_idx)
    indices = [(mode, h5_idx) for mode in modes for h5_idx in range(mode_sizes[mode])]

    # Create dataset directories
    task_name = "AFMRebias10"
    target_dataset_name = f'Dataset{dataset_id:03}_{task_name}'
    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # Create a list of arguments to pass to the wrapper function
    args_list = [
        (
            input_file,
            join(nnUNet_raw, target_dataset_name),
            indices,
            list(range(len(indices))),
            start,
            start + chunk_size
        ) for start in range(0, total_size, chunk_size)
    ]

    tqdm.contrib.concurrent.process_map(
        generate_nnunet_samples_wrapper,
        args_list,
        max_workers=num_workers,
    )

    # Finally, generate the dataset JSON
    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'AFM'},
        labels,
        total_size,
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
        'input_file',
        type=str,
        help="The generated AFM dataset in H5 format (must have afms_XXXXX subfolders)"
    )
    parser.add_argument(
        '-n', '--num_workers',
        type=int,
        default=4,
        help="Number of workers for parallel processing"
    )
    parser.add_argument(
        '-c', '--chunk_size',
        type=int,
        default=100,
        help="Chunk size for parallel processing"
    )
    parser.add_argument('-d', required=False, type=int, default=300, help='nnUNet Dataset ID, default: 300')
    args = parser.parse_args()
    input_file = args.input_file
    num_workers = args.num_workers
    chunk_size = args.chunk_size
    dataset_id = args.d

    num_workers = 4
    chunk_size = 1024

    main(args)
