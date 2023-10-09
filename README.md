MSCN: Noisy Correspondence Learning with Meta Similarity Correction (CVPR 2023, PyTorch Code)

## Requirements
- Python 3.8
- torch 1.7.0+cu110
- numpy
- scikit-learn
- pomegranate with TrueBetaDistribution (Install from https://github.com/rayleizhu/pomegranate. Note that pomegranate requires `Cython`, `NumPy`, `SciPy`, `NetworkX`, and `joblib`. Than you can run `python setup.py build` and `python setup.py install` to install it.)
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```

## Introduction

### Abstract 
Despite the success of multimodal learning in cross-modal retrieval task, the remarkable progress relies on the correct correspondence among multimedia data. However, collecting such ideal data is expensive and time-consuming. In practice, most widely used datasets are harvested from the Internet and inevitably contain mismatched pairs. Training on such noisy correspondence datasets causes performance degradation because the cross-modal retrieval methods can wrongly enforce the mismatched data to be similar. To tackle this problem, we propose a Meta Similarity Correction Network (MSCN) to provide reliable similarity scores. We view a binary classification task as the meta-process that encourages the MSCN to learn discrimination from positive and negative meta-data. To further alleviate the influence of noise, we design an effective data purification strategy using meta-data as prior knowledge to remove the noisy samples. Extensive experiments are conducted to demonstrate the strengths of our method in both synthetic and real-world noises, including Flickr30K, MS-COCO, and Conceptual Captions.


### MSCN Framework
<img src="https://github.com/hhc1997/MSCN/blob/main/meta-update.jpg"/>

## Datasets
We follow [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies. Our method needs an extra meta-data set to guide the training. 
For the Flickr30K dataset, we randomly split the meta-data from the validation set:

```
if opt.data_name == 'f30k_precomp':
    meta_len = 2900 # 2% of 145,000
    total_idsx = np.arange(0, len(images_dev)) #image length = caption length
    meta_idxs = np.random.choice(total_idsx, meta_len, False)
    captions_meta, images_meta = list(np.array(captions_dev)[meta_idxs]), images_dev[meta_idxs]
    #save...
```

  For the MS-COCO, the meta-data is split from the training set (6,328 pairs) and validation set (all 5,000 pairs):

```
if opt.data_name == 'coco_precomp':
    im_div = [0, 1, 2, 3, 4]
    sup_len = 6328 # 2%*566,435 - 5000
    total_img_idsx = np.arange(0, len(images_train))
    total_cap_idsx = np.arange(0, len(captions_train))
    sup_img_idxs = np.random.choice(total_img_idsx, sup_len, False)
    sup_0t4_idxs = np.random.choice(im_div, sup_len, True)
    sup_cap_idxs = sup_img_idxs * 5 + sup_0t4_idxs
    mask_img = np.ones(len(total_img_idsx), dtype=bool)
    mask_img[sup_img_idxs,] = False

    mask_cap = np.ones(len(total_cap_idsx), dtype=bool)
    del_cap_idxs = []
    for k in sup_img_idxs:
        del_cap_idxs.extend(list(range(k * len(im_div), k * len(im_div) + len(im_div))))
    del_cap_idxs = np.array(del_cap_idxs)
    mask_cap[del_cap_idxs,] = False
    # get meta data
    img_meta_sup = images_train[sup_img_idxs]
    cap_meta_sup = list(np.array(captions_train)[sup_cap_idxs])
    images_meta = np.vstack((images_dev, img_meta_sup))
    captions_meta = captions_dev + cap_meta_sup
    # get new train data
    images_train = images_train[mask_img]
    captions_train = list(np.array(captions_train)[mask_cap])
    #save    
```

For the CC152K, the meta-data is split from the validation set of the original Conceptual Captions. You can download the meta-data from https://drive.google.com/drive/folders/1XnGr7S-rXRfDbdeIF0QmTJV8kQFHx71-?usp=sharing.


## Training and Testing

``` 
# Flickr30K: noise_ratio = {0.2, 0.5, 0.7}
python main_MSCN.py --gpu 0 --data_name f30k_precomp --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

# MS-COCO: noise_ratio = {0.2, 0.5, 0.7}
python main_MSCN.py --gpu 0 --data_name coco_precomp --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

# Conceptual Captions
python main_MSCN.py --gpu 0 --data_name cc152k_precomp --data_path data_path --vocab_path vocab_path

```

## Cition
``` 
@InProceedings{Han_2023_CVPR,
    author    = {Han, Haochen and Miao, Kaiyao and Zheng, Qinghua and Luo, Minnan},
    title     = {Noisy Correspondence Learning With Meta Similarity Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7517-7526}
}
```

## Acknowledgements
The code is based on [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) licensed under Apache 2.0 and [MW-Net](https://github.com/xjtushujun/meta-weight-net) licensed under MIT.

