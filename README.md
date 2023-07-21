# Med-DDPM: Conditional Diffusion Models for Semantic 3D Medical Image Synthesis

[[Paper](https://arxiv.org/pdf/2305.18453.pdf)]

This repository houses the official implementation and pretrained model weights for our paper titled "Conditional Diffusion Models for Semantic 3D Medical Image Synthesis". Our work focuses on utilizing diffusion models to generate realistic and high-quality 3D medical images while preserving semantic information.

## Synthetic Samples for Given Input Mask:

<table>
  <tr>
    <td align="center">
      <strong>Input Mask</strong><br>
      <img id="img_0" src="images/img_0.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Real Image</strong><br>
      <img id="img_1" src="images/img_1.gif" alt="Real Image" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Synthetic Sample 1</strong><br>
      <img id="img_2" src="images/img_2.gif" alt="Synthetic Sample 1" width="100%">
    </td>
    <td align="center">
      <strong>Synthetic Sample 2</strong><br>
      <img id="img_3" src="images/img_3.gif" alt="Synthetic Sample 2" width="100%">
    </td>
  </tr>
</table>


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
