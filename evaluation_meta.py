"""Evaluation"""

from __future__ import print_function
import os
import sys
import time
import json
from itertools import chain

import torch
import numpy as np

from vocab import Vocabulary, deserialize_vocab
from models import SGRAF,Meta_Sim
from utils import AverageMeter, ProgressMeter
from data import get_dataset, get_loader


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    # switch to evaluate mode
    model.eval()
    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # max text length
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    image_ids = []
    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # image_ids.extend(img_ids)
        images, captions = images.cuda(), captions.cuda()
        # compute the embeddings
        with torch.no_grad():
            #img_emb, cap_emb, cap_len = model.module.forward_emb(images, captions, lengths)
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros(
                (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
            )
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        del images, captions
    # return img_embs, cap_embs, cap_lens, image_ids
    return img_embs, cap_embs, cap_lens

def validate(opt, val_loader, models = [],meta_nets =[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens,meta_nets[ind],opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count





    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    # r_sum = r1 + 0.5*r5 + 0.1*r10 + r1i + 0.5*r5i + 0.1*r10i

    return r1 ,r5 ,r10 , r1i ,r5i ,r10i

def encode_data_test(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    # switch to evaluate mode
    model.eval()
    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # max text length
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    image_ids = []
    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # image_ids.extend(img_ids)
        images, captions = images.cuda(), captions.cuda()
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros(
                (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
            )
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        del images, captions
    # return img_embs, cap_embs, cap_lens, image_ids
    return img_embs, cap_embs, cap_lens



def evalrank(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint["opt"]
    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # Load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

    # construct model
    net_A = SGRAF(opt).cuda()
    meta_net_A = Meta_Sim(hidden_size=opt.meta_net_hidden_size, num_layers=opt.meta_net_num_layers).cuda()
    net_B = SGRAF(opt).cuda()
    meta_net_B = Meta_Sim(hidden_size=opt.meta_net_hidden_size, num_layers=opt.meta_net_num_layers).cuda()

    # load model state
    net_A.load_state_dict(checkpoint["net_A"])
    meta_net_A.load_state_dict(checkpoint["meta_net_A"])
    net_B.load_state_dict(checkpoint["net_B"])
    meta_net_B.load_state_dict(checkpoint["meta_net_B"])


    print("Computing results...")
    validate(opt,data_loader,[net_A,net_B],[meta_net_A,meta_net_B])


def shard_attn_scores(model, img_embs, cap_embs, cap_lens,meta_net,opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    meta_net.eval()
    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                #sim,_ = model.module.forward_sim(im, ca, l,meta_net)
                sim, _ = model.forward_sim(im, ca, l, meta_net)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims


def shard_attn_scores_test(model,meta_net, img_embs, cap_embs, cap_lens, opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim,_ = model.forward_sim(im, ca, l,meta_net)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims

def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


