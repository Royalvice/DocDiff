<div align="center">

 <img width="100%" src="./demo_img/teaser.png">

</div>

<div align="center">

[简体中文](README.md) | [English](README.EN.md) | [Paper](https://arxiv.org/abs/2305.03892v1)
# DocDiff
This is the official repository for the paper [DocDiff: Document Enhancement via Residual Diffusion Models](https://arxiv.org/abs/2305.03892v1). DocDiff is a document enhancement model (please refer to the [paper](https://arxiv.org/abs/2305.03892v1)) that can be used for tasks such as document deblurring, denoising, binarization, watermark and stamp removal, etc. DocDiff is a lightweight residual prediction-based diffusion model, that can be trained on a batch size of 64 with only 12GB of VRAM at a resolution of 128*128.

Not only for document enhancement, DocDiff can also be used for other img2img tasks, such as natural scene deblurring[<sup>1</sup>](#refer-anchor-1), denoising, rain removal, super-resolution[<sup>2</sup>](#refer-anchor-2), image inpainting, as well as high-level tasks such as semantic segmentation[<sup>4</sup>](#refer-anchor-4).
</div>

# Guide

Whether it's for training or inference, you just need to modify the configuration parameters in `conf.yml` and run `main.py`. MODE=1 is for training, MODE=0 is for inference. The parameters in `conf.yml` have detailed annotations, so you can modify them as needed. Pre-trained weights for document deblurring Coarse Predictor and Denoiser can be found [here](https://drive.google.com/file/d/1F9oPOzrjZFd--yzQINZW3heWsewVdaj1/view?usp=sharing) and [here](https://drive.google.com/file/d/13vkMRRHXDGprXzX1kwGQTI6db7ecol1f/view?usp=sharing), respectively.

Please note that the default parameters in `conf.yml` work best for document scenarios. If you want to apply DocDiff to natural scenes, please first read [Notes!](#notes!) carefully. If you still have issues, welcome to submit an issue.

- Because downsampling is applied three times, the resolution of the input image must be a multiple of 8. If your image is not a multiple of 8, you can adjust the image to be a multiple of 8 using padding or cropping. Please do not directly resize, as it may cause image distortion. In particular, in the deblurring task, image distortion will increase the blur and result in poor performance. For example, the document deblurring dataset [<sup>5</sup>](#refer-anchor-5) used by DocDiff has a resolution of 300\*300, which needs to be padded to 304\*304 before inference.

## Environment

- python >= 3.7
- pytorch >= 1.7.0
- torchvision >= 0.8.0

<div align="center">

# Notes!

</div>

- The default configuration parameters and training/inference strategies of DocDiff are designed for **document images**. If you want to use it for natural scenes and achieve better results, you need to **adjust the parameters**, such as enlarging the model, adding self-attention, and modifying the **training/inference strategies**.
- **Training strategy**: As described in the paper, in document scenarios, we do not pursue diverse results and we need to minimize the inference time as much as possible. Therefore, we set the diffusion step T to 100, and predict $x_0$ instead of predicting $\epsilon$. Based on the premise of using a channel-wise concatenation conditioning scheme, this strategy can recover a fine $x_0$ in the early steps of reverse diffusion. In natural scenes, in order to better reconstruct textures and pursue diverse results, the diffusion step T should be set as large as possible, and $\epsilon$ should be predicted. You just need to modify **PRE_ORI="False"** in `conf.yml` to use the scheme of predicting $\epsilon$, and modify **TIMESTEPS=1000** to use a larger diffusion step.
- **Inference strategy**: The images generated in document scenarios should not have randomness. (short-step stochastic sampling may cause text edges to be distorted), so DocDiff performs deterministic sampling as described in DDIM[<sup>3</sup>](#refer-anchor-3). In natural scenes, stochastic sampling is essential for diverse results, so you can use stochastic sampling by modifying **PRE_ORI="False"** in `conf.yml`. In other words, the scheme of predicting $\epsilon$ is bound to stochastic sampling, while the scheme of predicting $x_0$ is bound to deterministic sampling. If you want to predict $x_0$ and use stochastic sampling, or predict $\epsilon$ and use deterministic sampling, you need to modify the code yourself. In DocDiff, deterministic sampling is performed using the method in DDIM, while stochastic sampling is performed using the method in DDPM. You can modify the code to implement other sampling strategies yourself.
- **Summary**: For tasks that do not require diverse results, such as semantic segmentation, document enhancement, predicting $x_0$ with a diffusion step of 100 is enough, and the performance is already good. For tasks that require diverse results, such as deblurring for natural scenes, super-resolution, image restoration, etc., predicting $\epsilon$ with a diffusion step of 1000 is recommended.

# To-do Lists

- [x] Add training code
- [x] Add inference code
- [x] Upload pre-trained model
- [x] Use DPM_solver to reduce inference step size (although the effect is not significant in practice)
- [ ] Train on multiple GPUs
- [ ] Jump-step sampling for DDIM
- [ ] Use depth separable convolution to compress the model
- [ ] Train the model on natural scenes and provide results and pre-trained models

# Acknowledgement

- If you find DocDiff helpful, please give us a star. Thank you! 🤞😘
- If you have any questions, please don't hesitate to open an issue. We will reply as soon as possible.
- If you want to communicate with us, please send an email to **viceyzy@foxmail.com** with the subject "**DocDiff**".
- If you want to use DocDiff as the baseline for your project, please cite our paper.
```
@articlec{yang2023docdiff,
      title={DocDiff: Document Enhancement via Residual Diffusion Models}, 
      author={Zongyuan Yang and Baolin Liu and Yongping Xiong and Lan Yi and Guibin Wu and Xiaojun Tang and Ziqi Liu and Junjie Zhou and Xing Zhang},
      journal={arXiv preprint arXiv:2305.03892},
      year={2023}
}
```

# References
<div id="refer-anchor-1"></div>

- [1] Whang J, Delbracio M, Talebi H, et al. Deblurring via stochastic refinement[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 16293-16303.

<div id="refer-anchor-2"></div>

- [2] Shang S, Shan Z, Liu G, et al. ResDiff: Combining CNN and Diffusion Model for Image Super-Resolution[J]. arXiv preprint arXiv:2303.08714, 2023.

<div id="refer-anchor-3"></div>

- [3] Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.

<div id="refer-anchor-4"></div>

- [4] Wu J, Fang H, Zhang Y, et al. MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model[J]. arXiv preprint arXiv:2211.00611, 2022.

<div id="refer-anchor-5"></div>

- [5] Michal Hradiš, Jan Kotera, Pavel Zemčík and Filip Šroubek. Convolutional Neural Networks for Direct Text Deblurring. In Xianghua Xie, Mark W. Jones, and Gary K. L. Tam, editors, Proceedings of the British Machine Vision Conference (BMVC), pages 6.1-6.13. BMVA Press, September 2015.
