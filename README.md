# MNCAA

This repository is for the paper, "MNCAA: Balanced Style Transfer Based on Multi-level Normalized Cross Attention Alignment".

Liu, H., MI, Z., Chen, F. et al. MNCAA: Balanced Style Transfer Based on Multi-level Normalized Cross Attention Alignment.

## Datasets

 Style images: [WIKIART](https://www.wikiart.org/)  <br>  
Content images: [COCO2014](https://cocodataset.org/#home) <br>

## Results

![1](/images/1.png)

![2](/images/2.png)

![3](/images/3.png)

Stylized results: Our method achieves a balanced style transfer, which can not only maintain the details of the content image, but also transfer the style patterns to the content image effectively.

## Requirements

* Ubuntu: 18.04

* python: 3.8.5

* pytorch: 1.4.0

* CUDA: 10.1

* Other necessary libraries: PIL,numpy,scipy,tqdm

## Training && Testing

### Testing

Pretrained models: [vgg-model, decoder, MNCAF](https://pan.baidu.com/disk/home?#/all?vmode=list&path=%2Fmncaf_model).<br> 
Please download them and put them into the floder ./model/.  <br>

If you want to deal with only one image:

`CUDA_VISIBLE_DEVICES=0 python Eval.py --content input/content/avril.jpg --style input/style/asheville.jpg`

or a folder:

`CUDA_VISIBLE_DEVICES=0 python Eval.py --content_dir input/content --style_dir input/style`

### Training

`CUDA_VISIBLE_DEVICES=0 python train.py --content_dir ../../train2014 --style_dir ../../wikiart`



## Citing

If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you ^ . ^. 

`@cite`
