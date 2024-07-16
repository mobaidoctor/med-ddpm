# +
import argparse
import glob
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import torchio as tio

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MRI datasets.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the outputs will be saved")
    return parser.parse_args()

def create_dirs(output_dir):
    dirs = {
        "t1": os.path.join(output_dir, "t1"),
        "t1ce": os.path.join(output_dir, "t1ce"),
        "t2": os.path.join(output_dir, "t2"),
        "flair": os.path.join(output_dir, "flair"),
        "seg": os.path.join(output_dir, "seg"),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def load_data_list(data_dir, modality, finalProcessing):
    if finalProcessing:
        return sorted(glob.glob(os.path.join(data_dir, modality, f"*_{modality}.nii.gz")))
    else:
        return sorted(glob.glob(os.path.join(data_dir, "*", f"*_{modality}.nii.gz")))

def preprocess_and_save(subject, output_dirs, img_names):
    for modality, img in subject.items():
        modality_key = modality.replace("_img", "")
        if modality_key != "seg":  # Skip mask for intensity rescaling
            transform = tio.RescaleIntensity((-1, 1))
            img = transform(img)
        img.save(os.path.join(output_dirs[modality_key], img_names[modality_key]))

def preprocess_seg(t1, seg_path, affine):
    img = nib.load(t1).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    seg[seg == 4] = 3
    img[img > -1] = 4.
    seg = np.where(seg == 0, img, seg)
    nib.save(nib.Nifti1Image(seg, affine), seg_path)

def main():
    args = parse_args()
    output_dirs = create_dirs(args.output_dir)

    modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    data_lists = {modality: load_data_list(args.data_dir, modality, False) for modality in modalities}
    
    # Preprocess and crop
    for idx in tqdm(range(len(data_lists["t1"]))):
        img_names = {modality: os.path.basename(data_lists[modality][idx]) for modality in modalities}
        subject = tio.Subject(
            t1_img=tio.ScalarImage(data_lists["t1"][idx]),
            t1ce_img=tio.ScalarImage(data_lists["t1ce"][idx]),
            t2_img=tio.ScalarImage(data_lists["t2"][idx]),
            flair_img=tio.ScalarImage(data_lists["flair"][idx]),
            seg=tio.LabelMap(data_lists["seg"][idx])
        )
        transform = tio.CropOrPad((192, 192, 144))
        subject = transform(subject)
        preprocess_and_save(subject, output_dirs, img_names)
    
    # Update the path to the output directory where the preprocessed images are saved
    data_lists = {modality: load_data_list(args.output_dir, modality, True) for modality in modalities}
    
    # Preprocess mask separately
    for t1_path, seg_path in tqdm(zip(data_lists["t1"], data_lists["seg"])):
        preprocess_seg(t1_path, seg_path, nib.load(seg_path).affine)

    print("COMPLETE!")
if __name__ == "__main__":
    main()

