**Multi-style Generative Network for Real-time Transfer**  [[arXiv](https://arxiv.org/pdf/1703.06953.pdf)] [[project](http://computervisionrutgers.github.io/MSG-Net/)]  
  [Hang Zhang](http://hangzh.com/),  [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html)
```
@article{zhang2017multistyle,
	title={Multi-style Generative Network for Real-time Transfer},
	author={Zhang, Hang and Dana, Kristin},
	journal={arXiv preprint arXiv:1703.06953},
	year={2017}
}
```
<div><img src ="images/figure1.jpg" width="500" /></div>	

### Table of Contents
0. [Demo Video](#demo-video)
0. [Installation](#installation)
0. [Test on New Images](#test-on-new-images)
0. [Train Your Own Model](#train-your-own-model)
0. [Release Timeline](#release-timeline)
0. [Acknowledgement](#acknowledgement)
0. [Example Results](Examples.md)

### Demo Video 
[![IMAGE ALT TEXT](http://img.youtube.com/vi/oy6pWNWBt4Y/0.jpg)](http://www.youtube.com/watch?v=oy6pWNWBt4Y "Video Title")

### Installation
Please install [Torch7](http://torch.ch/) with cuda and cudnn support. The code has been tested on Ubuntu 16.04 with Titan X Pascal and Maxwell.
```bash
luarocks install https://raw.githubusercontent.com/zhanghang1989/MSG-Net/master/texture-scm-1.rockspec
```

### Test on New Images

0. Clone the repo and download pre-trained models
	```bash
	git clone git@github.com:zhanghang1989/MSG-Net.git
	cd MSG-Net/experiments
	bash models/download_models.sh 
	```
0. Test the model
	```
	th test.lua
	eog stylized
	```
0. Test on new image
	```
	th test.lua -input_image path/to/the/image
	```

### Train Your Own Model
We are working on cleaning the training code, but we may decide to released it upon paper acceptance.

### Release Timeline
- 03/20/2017 we have released the [demo video](https://www.youtube.com/watch?v=oy6pWNWBt4Y).
- 03/24/2017 We have released [ArXiv paper](https://arxiv.org/pdf/1703.06953.pdf) and demo code with pre-trained models.
- 03/31/2017 Release the code for camera or video demo.
- We are working on cleaning the training code, but we may decide to released it upon paper acceptance.

### Acknowledgement
The code benefits from outstanding prior work and their implementations including:
- [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/pdf/1603.03417.pdf) by Ulyanov *et al. ICML 2016*. ([code](https://github.com/DmitryUlyanov/texture_nets))
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf) by Johnson *et al. ECCV 2016* ([code](https://github.com/jcjohnson/fast-neural-style))
- [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys *et al. CV{R 2016* and its torch implementation [code](https://github.com/jcjohnson/neural-style) by Johnson.
