# pix2pix2 - Tensorflow
A Tensorflow implementation of Pix2Pix

Original paper: [Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)](https://arxiv.org/pdf/1611.07004.pdf)    
Paper Authors and Researchers: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros    
## <a href = 'https://github.com/Sxela/face2comics'>Dataset face2comics </a> - Author: <a href = 'https://github.com/Sxela'>Alex (Sxela)</a>

<p align = 'center'>
  <img src = "./image/face2comic.png">
</p>

Review training on colab: <br>
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-CJdHLsR1Nm8MXYUqchZjWlIqFKzQayJ?usp=sharing)

Review training on Kaggle: <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/dat0chin/face2comic-pix2pix)

## Author 
<ul>
    <li>Github: <a href = "https://github.com/Nguyendat-bit">Nguyendat-bit</a> </li>
    <li>Email: <a href = "nduc0231@gmai.com">nduc0231@gmail</a></li>
    <li>Linkedin: <a href = "https://www.linkedin.com/in/nguyendat4801">Đạt Nguyễn Tiến</a></li>
</ul>

## Usage
### Dependencies
- python >= 3.9
- numpy >= 1.20.3
- tensorflow >= 2.7.0
- opencv >= 4.5.4
- matplotlib >= 3.4.3
### Train your model by running this command line

Training script:


```python

python train.py --all-train ${link_to_train_A_folder} / 
    --all-train ${link_to_train_B_folder} --epochs ${epochs}
    --bone ${bone} --weights ${weights} --pretrain ${pretrain} /
    --batch-size ${batch_size} --rotation ${rotation} / 
    --random-brightness ${random_brightness} --image-size ${image-size}

```


Example:

```python

python train.py --all-train face/*.jpg  --all-train comics/*.jpg \
  --bone resunet50_unet --weights imagenet --pretrain True --epochs 10 --batch-size 8 --rotation 60 --random-brightness True --image-size 256
``` 

There are some important arguments for the script you should consider when running it:

- `all-train`: The folder of training data 
- `batch-size`: The batch size of the dataset 
- `image-size`: The image size of the dataset
## Feedback
If you meet any issues when using this library, please let us know via the issues submission tab.