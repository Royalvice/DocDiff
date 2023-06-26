<div align="center">

 <img width="100%" src="demo/teaser.png">

</div>

<div align="center">

[简体中文](README.md) | [English](README.EN.md) | [Paper](https://arxiv.org/abs/2305.03892v1)
# DocDiff
这里是论文[DocDiff: Document Enhancement via Residual Diffusion Models](https://arxiv.org/abs/2305.03892v1)的官方复现仓库。DocDiff是一个文档增强模型（详见[论文](https://arxiv.org/abs/2305.03892v1)），可以用于文档去模糊、文档去噪、文档二值化、文档去水印和印章等任务。DocDiff是一个轻量级的基于残差预测的扩散模型，在128*128分辨率上以Batchsize=64训练只需要12GB显存。
不仅文档增强，DocDiff还可以应用在其他img2img任务上，比如自然场景去模糊[<sup>1</sup>](#refer-anchor-1)，去噪，去雨，超分[<sup>2</sup>](#refer-anchor-2)，图像修复等low-level任务以及语义分割[<sup>4</sup>](#refer-anchor-4)等high-level任务。
</div>

# News

- 2023.05.08: 代码的初始版本已经上传。请查看To-do lists来获取未来的更新。
- 2023.06.13: 为了方便复现，已上传推理笔记本`demo/inference.ipynb`和预训练模型`checksave/`。

# 使用指南

无论是训练还是推理，你只需要修改conf.yml中的配置参数，然后运行main.py即可。MODE=1为训练，MODE=0为推理。conf.yml中的参数都有详细注释，你可以根据注释修改参数。文档去模糊预训练权重在`checksave/`。
**请注意**conf.yml中的默认参数在文档场景表现最好。如果你想应用DocDiff在自然场景，请先看一下[注意事项!!!](#注意事项!!!)。如果仍有问题，欢迎提issue。

- 由于要下采样3次，所以输入图像的分辨率必须是8的倍数。如果你的图像不是8的倍数，可以使用padding或者裁剪的方式将图像调整为8的倍数。请不要直接Resize，因为这样会导致图像失真。尤其在去模糊任务中，图像失真会导致模糊程度增加，效果会变得很差。例如，DocDiff使用的文档去模糊数据集[<sup>5</sup>](#refer-anchor-5)分辨率为300\*300，需要先padding到304\*304，再送入推理。

## 环境配置

- python >= 3.7
- pytorch >= 1.7.0
- torchvision >= 0.8.0


<div align="center">

# 注意事项!!!
</div>

- DocDiff的默认配置参数，训练和推理策略是为**文档图像设计**的，如果要用于自然场景，想获得更好的效果，需要**调整参数**，比如扩大模型，添加Self-Attention等，并修改**训练和推理策略**。
- **训练策略**：如论文所述，在文档场景中，因为不追求生成多样性，并且希望尽可能缩减推理时间。所以我们将扩散步长T设为100，并预测 $x_0$ 而不是预测 $\epsilon$。在使用基于通道叠加的引入条件（Coarse Predictor的输出）的方案的前提下，这种策略可以使得在逆向扩散的前几步就可以恢复出较好的 $x_0$ 。在自然场景中，为了更好地重建纹理并追求生成多样性，扩散步长T尽可能大，并要预测 $\epsilon$ 。你只需要修改**conf.yml**中的**PRE_ORI="False"**，即可使用预测 $\epsilon$ 的方案; 修改**conf.yml**中的**TIMESTEPS=1000**，即可使用更大的扩散步长。
- **推理策略**：在文档场景中生成的图像不想带有随机性（短步随机采样会导致文本边缘扭曲），DocDiff执行DDIM[<sup>3</sup>](#refer-anchor-3)中的确定采样。在自然场景中，随机采样是生成多样性的关键，修改**conf.yml**中的**PRE_ORI="False"**，即可使用随机采样。也就是说，预测 $\epsilon$ 的方案与随机采样是绑定的，而预测 $x_0$ 的方案与确定采样是绑定的。如果你想预测 $x_0$ 并随机采样或者 预测 $\epsilon$ 并确定采样，你需要自己修改代码。DocDiff中确定采样是DDIM中的确定采样，随机采样是DDPM中的随机采样，你可以自己修改代码实现其他采样策略。
- **总结**：应用于不需要生成多样性的任务，比如语义分割，文档增强，使用预测 $x_0$ 的方案，扩散步长T设为100就ok，效果已经很好了；应用于需要生成多样性的任务，比如自然场景去模糊，超分，图像修复等，使用预测 $\epsilon$ 的方案，扩散步长T设为1000。

# To-do lists

- [x] 添加训练代码
- [x] 添加推理代码
- [x] 上传预训练模型
- [x] 使用DPM_solver减少推理步长（实际用起来，效果一般）
- [x] 上传Inference notebook，方便复现
- [ ] 多GPU训练
- [ ] DDIM的跳步采样
- [ ] 使用深度可分离卷积压缩模型
- [ ] 在自然场景上训练模型并提供结果和预训练模型

# 感谢

- 如果你觉得DocDiff对你有帮助，请给个star，谢谢！🤞😘
- 如果你有任何问题，欢迎提issue，我会尽快回复。
- 如果你想交流，欢迎给我发邮件**viceyzy@foxmail.com**，备注：**DocDiff**。
- 如果你愿意将DocDiff作为你的项目的baseline，欢迎引用我们的论文。
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
