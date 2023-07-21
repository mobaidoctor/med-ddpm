# Med-DDPM: Conditional Diffusion Models for Semantic 3D Medical Image Synthesis

[[Paper](https://arxiv.org/pdf/2305.18453.pdf)]

This repository houses the official implementation and pretrained model weights for our paper titled "Conditional Diffusion Models for Semantic 3D Medical Image Synthesis". Our work focuses on utilizing diffusion models to generate realistic and high-quality 3D medical images while preserving semantic information.

## Synthetic Samples for Given Input Mask:

| Input Mask | Real Image | Synthetic Sample 1 | Synthetic Sample 2 |
| ---------- | ---------- | ------------------ | ------------------ |
| ![Input Mask](images/img_0.gif) | ![Real Image](images/img_1.gif) | ![Synthetic Sample 1](images/img_2.gif) | ![Synthetic Sample 2](images/img_3.gif) |
| ![Synthetic Sample 3](images/img_4.gif) | ![Synthetic Sample 4](images/img_5.gif) | ![Synthetic Sample 5](images/img_6.gif) | ![Synthetic Sample 6](images/img_7.gif) |
| ![Synthetic Sample 7](images/img_8.gif) | ![Synthetic Sample 8](images/img_9.gif) | ![Synthetic Sample 9](images/img_10.gif) | ![Synthetic Sample 10](images/img_11.gif) |
| ![Synthetic Sample 11](images/img_12.gif) | ![Synthetic Sample 12](images/img_13.gif) | ![Synthetic Sample 13](images/img_14.gif) | ![Synthetic Sample 14](images/img_15.gif) |
| ![Synthetic Sample 15](images/img_16.gif) | ![Synthetic Sample 16](images/img_17.gif) | ![Synthetic Sample 17](images/img_18.gif) | ![Synthetic Sample 18](images/img_19.gif) |
| ![Synthetic Sample 19](images/img_20.gif) | ![Synthetic Sample 20](images/img_21.gif) | ![Synthetic Sample 21](images/img_22.gif) | ![Synthetic Sample 22](images/img_23.gif) |


## Setup

The following two libraries must be installed for training and generation.

- Torchio : [torchio](https://github.com/fepegar/torchio)
- Nibabel : [nibabel](https://github.com/nipy/nibabel)

## Training 

Learning can be performed from the following code. The script is executed according to the data size 64, 128. 
The path to the dataset folder is specified with `--inputfolder` in the script code.

**Size : 128x128x128**

```
$ ./scripts/train128.sh
```

## Generate Samples

To generate samples, run the following script The learned weight file is specified by `--weightfile`, and the mask file to be input is specified by `--inputfile`.

**Size : 128x128x128**

```
$ ./scripts/generate128.sh
```

## Citation

To cite our work, please use

```
@misc{,
  doi = {},
  url = {https://arxiv.org/abs/2305.18453},
  author = {Zolnamar Dorjsembe, Hsing-Kuo Pao, Sodtavilan Odonchimed, Furen Xiao},
  title = {Conditional Diffusion Models for Semantic 3D Medical Image Synthesis},
  publisher = {arXiv},
  year = {2022},
}
```
