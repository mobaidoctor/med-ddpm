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


## 🛠️ Setup 

Ensure you have the following libraries installed for training and generating images:

- **Torchio**: [Torchio GitHub](https://github.com/fepegar/torchio)
- **Nibabel**: [Nibabel GitHub](https://github.com/nipy/nibabel)

```
pip install -r requirements.txt
```

## 🚀 Run on Your Own Dataset

Med-DDPM is versatile. If you're working with image formats other than NIfTI (.nii.gz), modify the \`read_image\` function in \`dataset.py\`.

1. Specify the segmentation mask directory with \`--inputfolder\`.
2. Set the image directory using \`--targetfolder\`.
3. If you have more than 3 segmentation mask label classes, update channel configurations in \`train.py\`, \`datasets.py\`, and \`utils/dtypes.py\`.

## 🎓 Training 

Specify dataset paths using \`--inputfolder\` and \`--targetfolder\`:

- **Image Dimensions**: 128x128x128

```
$ ./scripts/train.sh
```

## 🧠 Model Weights

Access our optimized model weights using the link below:

[Download Model Weights](https://drive.google.com/file/d/1cy1uPjA7PEcL3FDf2-weprWvLSJoJf_n/view?usp=sharing)

After downloading, place the file under the "model" directory.

## 🎨 Generate Samples

To produce images, follow the script below:

- **Image Dimensions**: 128x128x128
- Set the learned weight file path with \`--weightfile\`.
- Determine the input mask file using \`--inputfolder\`.

```
$ ./scripts/sample.sh
```

## 📋 ToDo List

Your contributions to Med-DDPM are valuable! Here's our ongoing task list:

- [x] Main model code release
- [x] Release model weights 
- [x] Implement fast sampling feature
- [ ] Release 4 modality model code & weights
- [ ] Deploy model on HuggingFace for broader reach
- [ ] Draft & release a comprehensive tutorial blog
- [ ] Launch a Docker image

## 📜 Citation

If our work assists your research, kindly cite us:

```
@misc{https://doi.org/10.48550/arxiv.2305.18453,
  doi = {10.48550/ARXIV.2305.18453},
  url = {https://arxiv.org/abs/2305.18453},
  author = {Zolnamar Dorjsembe and Hsing-Kuo Pao and Sodtavilan Odonchimed and Furen Xiao},
  title = {Conditional Diffusion Models for Semantic 3D Medical Image Synthesis},
  publisher = {arXiv},
  year = {2023},
}
```

## 💡 Acknowledgements

Gratitude to these foundational repositories:

1. [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
2. [guided-diffusion](https://github.com/openai/guided-diffusion)


```python

```
