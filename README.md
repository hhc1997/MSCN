MSCN: Noisy Correspondence Learning with Meta Similarity Correction (CVPR 2023, PyTorch Code)

## Requirements
- Python 3.7
- torch 1.7.0+cu110
- numpy
- scikit-learn
- pomegranate [Install](https://github.com/jmschrei/pomegranate/pull/901)
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```

## Introduction

### Meta Process
<img src="https://github.com/hhc1997/MSCN/blob/main/meta-process.pdf"/>

### MSCN Framework
<img src="https://github.com/hhc1997/MSCN/blob/main/meta-update.pdf"/>

## Datasets
We follow [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies. Our method needs an extra meta-data set to guide the training. For Flickr30K and MS-COCO, it can be splited from training set or validation set. For Conceptual Captions contains real noise, it can only be splited from validtion set. 

## Training and Testing

``` 
# Flickr30K: noise_ratio = {0.2, 0.5, 0.7}
python main_MSCN.py --gpu 0 --data_name f30k_precomp --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

# MS-COCO: noise_ratio = {0.2, 0.5, 0.7}
python main_MSCN.py --gpu 0 --data_name coco_precomp --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

# Conceptual Captions
python main_MSCN.py --gpu 0 --data_name cc152k_precomp --data_path data_path --vocab_path vocab_path

```

## Acknowledgements
The code is based on [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) licensed under Apache 2.0 and [MW-Net](https://github.com/xjtushujun/meta-weight-net) licensed under MIT.

