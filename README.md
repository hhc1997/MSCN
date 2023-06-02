MSCN: Noisy Correspondence Learning with Meta Similarity Correction (CVPR 2023, PyTorch Code)

## Requirements
- Python 3.8
- torch 1.7.0+cu110
- numpy
- scikit-learn
- pomegranate [Install]([https://github.com/jmschrei/pomegranate/pull/901](https://github.com/rayleizhu/pomegranate))
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
We follow [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies. Our method needs an extra meta-data set to guide the training. For Flickr30K and MS-COCO, it can be splited from training set or validation set. For Conceptual Captions contains real noise, it can only be splited from validtion set. We provide the processed features of meta-data in https://drive.google.com/drive/folders/1XnGr7S-rXRfDbdeIF0QmTJV8kQFHx71-?usp=share_link.

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

