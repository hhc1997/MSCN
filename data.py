"""Dataloader"""

import os
import copy
import csv
import nltk
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        captions,
        images,
        data_split,
        noise_ratio=0,
        noise_file="",
    ):
        assert 0 <= noise_ratio < 1

        self.captions = captions
        self.images = images
        self.noise_ratio = noise_ratio
        self.data_split = data_split

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't.
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        # if data_split == "dev":
        #     self.length = 1000 * self.im_div

        # one image has five captions
        self.t2i_index = np.arange(0, self.length) // self.im_div

        # Noisy label
        if data_split == "train" or data_split == "train_all":
            split_idx = None
            self._t2i_index = copy.deepcopy(self.t2i_index)
            if noise_ratio:
                if os.path.exists(noise_file):
                    print("=> load noisy index from {}".format(noise_file))
                    self.t2i_index = np.load(noise_file)
                else:
                    idx = np.arange(self.length)
                    np.random.shuffle(idx)
                    noise_length = int(noise_ratio * self.length)

                    shuffle_index = self.t2i_index[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)
                    self.t2i_index[idx[:noise_length]] = shuffle_index

                    np.save(noise_file, self.t2i_index)
                    print("=> save noisy index to {}".format(noise_file))

            # save clean labels
            self._labels = np.ones((self.length), dtype="int")
            self._labels[self._t2i_index != self.t2i_index] = 0

            noise_label = np.ones_like(self._labels)

            if split_idx is not None:
                # self.images = self.images[split_idx]
                self.captions = [self.captions[i] for i in split_idx]
                self.t2i_index = [self.t2i_index[i] for i in split_idx]
                self._t2i_index = [self._t2i_index[i] for i in split_idx]
                self._labels = [self._labels[i] for i in split_idx]
                self.length = len(self.captions)

        print("{} data has a size of {}".format(data_split, self.length))

    def __getitem__(self, index):
        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = np.array(self.captions[index])
        text = torch.Tensor(text)
        if self.data_split == "train_all":
            label = self._labels[index]
            return image, text, index, self.t2i_index[index],label

        else:
            return image, text, index, self.t2i_index[index]


    def __len__(self):
        return self.length

class DataLoaderX(DataLoader):

    def __iter__(self,num=2):
        return BackgroundGenerator(super().__iter__(),max_prefetch = num)

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        text: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    labels = None
    if len(data[0]) == 6:
        images, captions, ids, labels, prob, _labels = zip(*data)
        # Merge
        labels = torch.stack(labels, 0).long()
        # Merge
        prob = torch.stack(prob, 0)

    elif len(data[0]) == 5:
        images, captions, ids, img_ids, _labels = zip(*data)

    else:
        images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    text = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        text[i, :end] = cap[:end]

    if len(data[0]) == 6:
        return images, text, lengths, ids, labels, prob, _labels
    elif len(data[0]) == 5:
        return images, text, lengths, ids, _labels
    else:
        return images, text, lengths, ids


def collate_fn_meta_C(data):
    images, captions,labels, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    text = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        text[i, :end] = cap[:end]


    return images, text, lengths, labels, ids

def get_dataset(data_path, data_name, data_split, vocab, return_id_caps=False):
    data_path = os.path.join(data_path, data_name)

    # Captions
    captions = []
    if data_name == "cc152k_precomp":
        img_ids = []
        with open(os.path.join(data_path, "%s_caps.tsv" % data_split)) as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                captions.append(line[1].strip())
                img_ids.append(line[0])

    elif data_name in ["coco_precomp", "f30k_precomp"]:
        with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r") as f:
            for line in f:
                captions.append(line.strip())

    else:
        raise NotImplementedError("Unsupported dataset!")

    # caption tokens
    captions_token = []
    for index in range(len(captions)):
        caption = captions[index]
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        captions_token.append(caption)

    # images
    images = np.load(os.path.join(data_path, "%s_ims.npy" % data_split))
    print(
        "load {} / {} data: {} images, {} captions".format(
            data_path, data_split, images.shape[0], len(captions)
        )
    )
    if return_id_caps:
        return captions_token, images, img_ids, captions
    else:
        return captions_token, images

def get_meta_dataset(captions_train_data, images_train_data,num_meta_total):
    '''
    :param captions_train_data:  ndarry 29000*36*2048
    :param images_train_data:    list 14500
    :param num_meta_total:
    :return: captions_train_data, images_train_data , captions_meta_data, images_meta_data
    '''

    data_length = images_train_data.shape[0] # i2t one to more
    im_div = int(len(captions_train_data)/data_length)
    i_index_total = list(range(0,data_length))
    np.random.shuffle(i_index_total)
    i_index_meta = i_index_total[:num_meta_total]
    i_index_train = i_index_total[num_meta_total:]
    t_index_meta = []
    t_index_train = []

    #meta
    for i in i_index_meta:
        t_index_meta.extend(list(range(i * im_div, i * im_div + im_div)))
    #train
    for i in i_index_train:
        t_index_train.extend(list(range(i * im_div, i * im_div + im_div)))
    captions_train_data = np.array(captions_train_data,dtype=object)

    return list(captions_train_data[t_index_train]),images_train_data[i_index_train],list(captions_train_data[t_index_meta]),images_train_data[i_index_meta]


class PrecompDataset_meta_C(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        captions,
        images,
        captions_add,
        images_add,

    ):

        self.captions = captions
        self.images = images
        self.captions_add = captions_add
        self.images_add = images_add

        self.length = len(self.captions)*2

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't.
        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1


        # one image has five captions
        self.t2i_index = np.arange(0, len(self.captions)) // self.im_div

        self.non_cor_t2i_index = None

        self.nocor_rand_t2i_idx = None
        self.rand_caption_idx = None


        self.get_add_non_correspondence_idx()



    def get_add_non_correspondence_idx(self):
        add_image_idx = np.arange(0, len(self.images_add))
        add_caption_idx = np.arange(0, len(self.captions_add))
        add_t2i = add_caption_idx // self.im_div
        rand_caption_idx = np.random.choice(add_caption_idx, len(self.captions), replace=False)
        cor_rand_t2i = add_t2i[rand_caption_idx]
        while True:
            nocor_rand_t2i_idx = np.random.choice(add_image_idx, len(self.captions), replace=True)
            if np.sum(cor_rand_t2i == nocor_rand_t2i_idx) == 0:
                self.nocor_rand_t2i_idx = nocor_rand_t2i_idx
                self.rand_caption_idx = rand_caption_idx
                break

    def __getitem__(self, index):

        if index < self.length/2:
            image = torch.Tensor(self.images[self.t2i_index[index]])
            text = np.array(self.captions[index])
            text = torch.Tensor(text)
            label = 1
            return image, text, label, index, self.t2i_index[index]
        else:
            index_map = int(index - self.length/2)
            image = torch.Tensor(self.images_add[self.nocor_rand_t2i_idx[index_map]])
            text = np.array(self.captions_add[self.rand_caption_idx[index_map]])
            text = torch.Tensor(text)
            label = 0
            return image, text,label,index, self.t2i_index[index_map]


    def __len__(self):
        return self.length

class PrecompDataset_correct(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        captions,
        images,
        data_split,
        noise_ratio=0,
        noise_file="",
        mode="",
        pred=[],
        probability=[],
    ):
        assert 0 <= noise_ratio < 1

        self.captions = captions
        self.images = images
        self.noise_ratio = noise_ratio
        self.data_split = data_split
        self.mode = mode

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't.
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1


        # one image has five captions
        self.t2i_index = np.arange(0, self.length) // self.im_div

        # Noisy label
        if data_split == "train_C":
            split_idx = None
            self._t2i_index = copy.deepcopy(self.t2i_index)
            if noise_ratio:
                if os.path.exists(noise_file):
                    print("=> load noisy index from {}".format(noise_file))
                    self.t2i_index = np.load(noise_file)
                else:
                    idx = np.arange(self.length)
                    np.random.shuffle(idx)
                    noise_length = int(noise_ratio * self.length)

                    shuffle_index = self.t2i_index[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)
                    self.t2i_index[idx[:noise_length]] = shuffle_index

                    np.save(noise_file, self.t2i_index)
                    print("=> save noisy index to {}".format(noise_file))

            # save clean labels
            self._labels = np.ones((self.length), dtype="int")
            self._labels[self._t2i_index != self.t2i_index] = 0

            noise_label = np.ones_like(self._labels)
            if self.mode == "labeled":
                split_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in split_idx]

            elif self.mode == "unlabeled":
                split_idx = (1 - pred).nonzero()[0]

            if split_idx is not None:
                # self.images = self.images[split_idx]
                self.captions = [self.captions[i] for i in split_idx]
                self.t2i_index = [self.t2i_index[i] for i in split_idx]
                self._t2i_index = [self._t2i_index[i] for i in split_idx]
                self._labels = [self._labels[i] for i in split_idx]
                self.length = len(self.captions)

        print("{} {} data has a size of {}".format(data_split, self.mode, self.length))

    def __getitem__(self, index):
        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = torch.Tensor(self.captions[index])

        if self.data_split == "train_C":
            if self.mode == "labeled":
                return (
                    image,
                    text,
                    index,
                    torch.Tensor([1]), # label (contain noise)
                    torch.Tensor([self.probability[index]]), # probs
                    self._labels[index], # real label
                )
            elif self.mode == "unlabeled":
                return image, text, index, self._labels[index], 0
            else:
                return image, text, index, self.t2i_index[index]
        else:
            return image, text, index, self.t2i_index[index]

    def __len__(self):
        return self.length


def get_loader(
    captions,
    images,
    data_split,
    batch_size,
    workers,
    noise_ratio=0,
    noise_file="",
    captions_add=None,  # only for meta_c
    images_add=None,  # only for meta_c
    samper_seq = None

):
    if data_split == "train":
        dset = PrecompDataset(captions, images, "train", noise_ratio, noise_file)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False if samper_seq else True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        return data_loader, dset.length, dset._labels

    elif data_split == "train_all":
        dset = PrecompDataset(captions, images, "train_all", noise_ratio, noise_file)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False if samper_seq else True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        return data_loader, dset.length, dset._labels


    elif data_split == "dev":
        dset = PrecompDataset(captions, images, data_split)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

    elif data_split == "meta":
        dset = PrecompDataset(captions, images, data_split)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    elif data_split == "meta_C":
        dset = PrecompDataset_meta_C(captions, images, captions_add, images_add)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn_meta_C,
            num_workers=workers,
        )
    elif data_split in ["test", "testall", "test5k"]:
        dset = PrecompDataset(captions, images, data_split)
        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    else:
        raise NotImplementedError("Not support data split!")
    return data_loader


def get_loader_correct(
    captions,
    images,
    batch_size,
    workers,
    noise_ratio=0,
    noise_file="",
    pred=[],
    prob=[],
):
    dset_c = PrecompDataset_correct(
            captions,
            images,
            "train_C",
            noise_ratio,
            noise_file,
            mode="labeled",
            pred=pred,
            probability=prob,
        )

    data_loader_c = DataLoader(
        dataset=dset_c,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=workers,
    )

    return data_loader_c