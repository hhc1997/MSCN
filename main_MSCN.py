import os
import numpy as np
import torch
from tqdm import tqdm
from pomegranate import GeneralMixtureModel,TrueBetaDistribution
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
from tensorboardX import SummaryWriter
from data import get_loader, get_dataset,get_loader_correct
from models import SGRAF,Meta_Sim
from opt import get_options
from vocab import deserialize_vocab
from evaluation_meta import i2t, t2i, encode_data, shard_attn_scores,evalrank
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    init_seeds,
    save_config,
    MetaAdam,
)

def warmup(opt, warm_trainloader,net,meta_net,optimizer,meta_optimizer):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(warm_trainloader), [batch_time, data_time, losses], prefix="Warmup Step"
    )
    end = time.time()
    for iteration, (images, captions, lengths, _) in enumerate(warm_trainloader):
        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break
        net.train()
        meta_net.train()
        images, captions = images.cuda(), captions.cuda()
        optimizer.zero_grad()
        meta_optimizer.zero_grad()
        loss = net(images, captions, lengths, meta_net, warm_up=True, ind=False)
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(net.parameters(), opt.grad_clip)
            clip_grad_norm_(meta_net.parameters(), opt.grad_clip)
        optimizer.step()
        meta_optimizer.step()
        losses.update(loss.item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.log_step == 0:
            progress.display(iteration)

def splite_data(opt, noisy_trainloader,meta_dataloader,net,meta_net,captions_train,images_train):
    net.eval()
    meta_net.eval()
    meta_len = len(meta_dataloader.dataset)
    # init BMM by meta data
    sims_meta = []
    ids_meta = []
    labels_meta = []
    with torch.no_grad():
        for i, (meta_images, meta_captions, meta_lengths, labels, ids) in enumerate(meta_dataloader):
            meta_images, meta_captions, labels, ids = meta_images.cuda(), meta_captions.cuda(), torch.tensor(labels).cuda(), torch.tensor(ids).cuda()
            meta_sims = net.get_sim_individual(meta_images, meta_captions, meta_lengths, meta_net)
            sims_meta.extend(meta_sims)
            ids_meta.extend(ids)
            labels_meta.extend(labels)
        sims_meta = torch.tensor(sims_meta).cuda()
        ids_meta = torch.tensor(ids_meta).cuda()
        labels_meta = torch.tensor(labels_meta).cuda()
        meta_sims_input = np.zeros(meta_len)
        meta_labels_input = np.zeros(meta_len)
        for b in range(meta_len):
            meta_sims_input[ids_meta[b]] = sims_meta[b]
            meta_labels_input[ids_meta[b]] = labels_meta[b]
        meta_noise = meta_sims_input[np.nonzero(meta_labels_input < 0.5)]
        meta_clean = meta_sims_input[np.nonzero(meta_labels_input > 0.5)]
        #clamp of sims
        meta_noise[meta_noise >= 1] = 1 - 10e-4
        meta_noise[meta_noise <= 0] = 10e-4
        meta_clean[meta_clean >= 1] = 1 - 10e-4
        meta_clean[meta_clean <= 0] = 10e-4
        BM_noise = TrueBetaDistribution.from_samples(meta_noise)
        BM_clean = TrueBetaDistribution.from_samples(meta_clean)
        BMM = GeneralMixtureModel([BM_noise, BM_clean])
        # fit by training data
        sims_all = []
        ids_all = []
        labels_all = []
        for iteration, (images, captions, lengths, ids, labels) in tqdm(enumerate(noisy_trainloader)):
            images, captions = images.cuda(), captions.cuda()
            sims_ind = net.get_sim_individual(images, captions, lengths, meta_net)
            ids = torch.tensor(ids).cuda()
            sims_all.extend(sims_ind)
            ids_all.extend(ids)
            labels_all.extend(labels)
        labels_all = np.array(labels_all) # labels only used to evaluation
        sims_all = torch.tensor(sims_all).numpy()
        #process of sims
        sims_all[sims_all >= 1] = 1 - 10e-4
        sims_all[sims_all <= 0] = 10e-4
        sims_all = sims_all.reshape(-1, 1)
        BMM.fit(sims_all,stop_threshold=1e-2, max_iterations=10)
        print(BMM)
        prob = BMM.predict_proba(sims_all)[:, 1]
        pred = prob > 0.5  # True = Correct data
        correct_labels = labels_all[pred]
        print('Correct data acc:', sum(correct_labels) / len(correct_labels))
        print('Total data acc:', sum(labels_all == pred) / len(labels_all))

    correct_trainloader = get_loader_correct(
        captions_train,
        images_train,
        opt.batch_size,
        opt.workers,
        noise_ratio=opt.noise_ratio,
        noise_file=opt.noise_file,
        pred=pred,
        prob=prob,
    )
    return correct_trainloader

def train(opt, correct_trainloader,meta_dataloader,net,meta_net,optimizer,meta_optimizer,epoch,BCE_Loss):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(correct_trainloader),
        [batch_time, data_time, losses],
        prefix="Training Step",
    )
    # reset meta data
    meta_dataloader.dataset.get_add_non_correspondence_idx()
    meta_dataloader_iter = iter(meta_dataloader)
    # adjust lr
    lr = opt.lr * (0.1 ** (epoch // opt.lr_update))
    for group in optimizer.param_groups:
        group['lr'] = lr
    meta_lr = opt.meta_lr * (0.1 ** (epoch // opt.lr_update))
    for group in meta_optimizer.param_groups:
        group['lr'] = meta_lr
    # train the network
    print('\n Training...')
    end = time.time()
    for iteration, (images, captions, lengths, _, _, _, _) in enumerate(correct_trainloader):
        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break
        net.train()
        images, captions = images.cuda(), captions.cuda()
        # W_hat(t) <-- W(t)
        with torch.backends.cudnn.flags(enabled=False):
            if (iteration + 1) % opt.meta_interval == 0:
                pseudo_net = SGRAF(opt).cuda()
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()
                meta_net.train()
                for param in meta_net.parameters():
                    param.requires_grad = True
                pseudo_loss = pseudo_net(images, captions, lengths, meta_net, warm_up=False, ind=False)
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
                pseudo_optimizer = MetaAdam(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                if opt.grad_clip > 0:
                    clip_grad_norm_(pseudo_net.parameters(), opt.grad_clip)
                pseudo_optimizer.meta_step(pseudo_grads)
                del pseudo_loss
                # theta_(t+1) <-- theta_(t)
                try:
                    meta_images, meta_captions, meta_lengths, labels, _ = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_images, meta_captions, meta_lengths, labels, _ = next(meta_dataloader_iter)
                meta_images, meta_captions, labels = meta_images.cuda(), meta_captions.cuda(), torch.Tensor(labels).unsqueeze(-1).cuda()
                meta_out = pseudo_net(meta_images, meta_captions, meta_lengths, meta_net, warm_up=False, ind=True)
                meta_optimizer.zero_grad()
                meta_loss = BCE_Loss(meta_out, labels)
                meta_loss.backward()
                meta_optimizer.step()
        # W_(t+1) <-- W_(t)
        for param in meta_net.parameters():
            param.requires_grad = False
        optimizer.zero_grad()
        loss = net(images, captions, lengths, meta_net, warm_up=False, ind=False)
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(net.parameters(), opt.grad_clip)
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.log_step == 0:
            progress.display(iteration)

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

    return r1 ,r5 ,r10 , r1i ,r5i ,r10i

def main(opt):
    print("\n*-------- Experiment Config --------*")
    print(opt)
    # Output dir
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not opt.noise_file:
        opt.noise_file = os.path.join(
            opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
        )
    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
        opt.noise_file = ""
    # save config
    save_config(opt, os.path.join(opt.output_dir, "config.json"))
    # set tensorboard
    writer = SummaryWriter(os.path.join('runs', opt.output_dir))
    # fix random seeds and cuda
    init_seeds()
    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)
    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_meta, images_meta = get_dataset(opt.data_path, opt.data_name, "meta", vocab)
    captions_dev,images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    meta_dataloader = get_loader(
        captions_meta, images_meta, "meta_C", opt.batch_size, opt.workers,
        captions_add=captions_train, images_add=images_train,
    )
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size,opt.workers,
    )
    # create models
    net_A = SGRAF(opt).cuda()
    meta_net_A = Meta_Sim().cuda()
    net_B = SGRAF(opt).cuda()
    meta_net_B = Meta_Sim().cuda()
    # load from checkpoint if existed
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            print('Load warm up model')
            checkpoint = torch.load(opt.warmup_model_path)
            net_A.load_state_dict(checkpoint["net_A"])
            meta_net_A.load_state_dict(checkpoint["meta_net_A"])
            net_B.load_state_dict(checkpoint["net_B"])
            meta_net_B.load_state_dict(checkpoint["meta_net_B"])
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    #init
    best_rsum = 0
    BCE_Loss = torch.nn.BCELoss().cuda()
    #warm up
    if opt.warmup_epoch > 0:
        warm_trainloader, _, _ = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size*2,
            opt.workers,
            noise_ratio=opt.noise_ratio,
            noise_file=opt.noise_file,
        )
        # create optimizer
        optimizer_A = torch.optim.Adam(
            net_A.parameters(),
            lr=opt.lr,  # warm up
        )
        meta_optimizer_A = torch.optim.Adam(meta_net_A.parameters(), lr=opt.lr)  # when warm up ,same lr
        # create optimizer
        optimizer_B = torch.optim.Adam(
            net_B.parameters(),
            lr=opt.lr,  # warm up
        )
        meta_optimizer_B = torch.optim.Adam(meta_net_B.parameters(), lr=opt.lr)  # when warm up ,same lr
        for epoch in range(opt.warmup_epoch):
            print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, warm_trainloader, net_A, meta_net_A, optimizer_A, meta_optimizer_A)
            print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, warm_trainloader, net_B, meta_net_B, optimizer_B, meta_optimizer_B)
        del warm_trainloader

        # val the network
        print("\n Validattion ...")
        r1 ,r5 ,r10 , r1i ,r5i ,r10i = validate(opt, val_loader, [net_A,net_B], [meta_net_A,meta_net_B])
        rsum = r1 + r5 + r10 + r1i + r5i + r10i
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "net_A": net_A.state_dict(),
                    "meta_net_A": meta_net_A.state_dict(),
                    "net_B": net_B.state_dict(),
                    "meta_net_B": meta_net_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="warm_checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )
    # train
    optimizer_A = torch.optim.Adam(
        net_A.parameters(),
        lr=opt.lr, #
    )
    meta_optimizer_A = torch.optim.Adam(meta_net_A.parameters(), lr=opt.meta_lr)  # reset lr
    optimizer_B = torch.optim.Adam(
        net_B.parameters(),
        lr=opt.lr, #
    )
    meta_optimizer_B = torch.optim.Adam(meta_net_B.parameters(), lr=opt.meta_lr)  # reset lr

    noisy_trainloader, _, _ = get_loader(
        captions_train,
        images_train,
        "train_all",
        opt.batch_size*64,
        opt.workers,
        noise_ratio = opt.noise_ratio,
        noise_file = opt.noise_file,
        samper_seq = True
    )

    for epoch in range(opt.num_epochs):
        print('Epoch', epoch, '/', opt.num_epochs)
        print("Split dataset ...")
        correct_trainloader_A = splite_data(opt, noisy_trainloader, meta_dataloader, net_A, meta_net_A, captions_train, images_train)
        print("\nModel A training ...")
        train(opt, correct_trainloader_A, meta_dataloader, net_B, meta_net_B, optimizer_B, meta_optimizer_B, epoch, BCE_Loss)
        print("Split dataset ...")
        correct_trainloader_B = splite_data(opt, noisy_trainloader, meta_dataloader, net_B, meta_net_B, captions_train, images_train)
        print("\nModel B training ...")
        train(opt, correct_trainloader_B, meta_dataloader, net_A, meta_net_A, optimizer_A, meta_optimizer_A, epoch, BCE_Loss)

        print("\n Validattion ...")
        r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [net_A,net_B], [meta_net_A,meta_net_B])
        rsum = r1 +  r5 + r10 + r1i + r5i + r10i
        writer.add_scalar('Image to Text R1', r1, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R5', r5, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R10', r10, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R1', r1i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R5', r5i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R10', r10i, global_step=epoch, walltime=None)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "net_A": net_A.state_dict(),
                    "meta_net_A": meta_net_A.state_dict(),
                    "net_B": net_B.state_dict(),
                    "meta_net_B": meta_net_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

    # test
    print("\n*-------- Testing --------*")
    if opt.data_name == "coco_precomp":
        print("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, "model_best.pth.tar"),
            split="testall",
            fold5=True,
        )
        print("full validation")
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="test")




if __name__ == "__main__":
    # load arguments
    opt = get_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # traing and evaluation
    print("\n*-------- Training & Testing --------*")
    main(opt)
