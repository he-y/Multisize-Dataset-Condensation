import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader, Data
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils 
from math import ceil
from matchloss import *
from reg_ipcx import *
import glob
import pickle

# Dream
from query_strategies import RandomSampling, KMeansSampling

import time
from aim import Run, Text

# Dream 
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "KMeansSampling":
        return KMeansSampling
    else:
        raise NotImplementedError

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        self.dream = args.slct_type == 'dream' # whether to use DREAM initialization

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)

        logger(f"\nDefine synthetic data: {self.data.shape}")

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        logger(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            logger("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            logger("Mixed initialize synset")
            for c in range(self.nclass):
                if not self.dream:  # use IDC initialization
                    img, _ = loader.class_sample(c, self.ipc * self.factor**2, start_idx=args.start_idx)
                    img = img.data.to(self.device)
                else:   # use DREAM initialization
                    assert hasattr(self, "dream_init_images"), "DREAM initialization images not found"
                    img = self.dream_init_images[c]
                    img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            indices = indices[:max_size]
            data = data[indices]
            target = target[indices]
            # [49, 123, 231, ...]
        else:
            indices = np.arange(data.shape[0])
            # [ 0, ..., 39]

        return data, target, indices

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target, indices = self.subsample(data, target, max_size=max_size)
        return data, target, indices

    def loader(self, args, augment=True, ipcx=-1, indices=None):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            if ipcx > 0:
                idx_to = self.ipc * c + ipcx
            else:
                idx_to = self.ipc * (c + 1)
            
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            if indices is not None: # use indices after decoding
                data = data[indices]
                target = target[indices]

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        logger(f"Decode condensed data: {data_dec.shape}")
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, ipcx=-1, indices=None, aim_run=None, step=None, context=None):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment, ipcx=ipcx, indices=indices)
        return test_data(args, loader, val_loader, logger=logger, aim_run=aim_run, step=step, context=context)

def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor(),
                                      download=True)
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    logger(f"Augmentataion Matching: {aug_type}")
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    logger(f"Augmentataion Net update: {aug_type}")
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand

def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut'
    folder_list = glob.glob(f'{folder_base}*')
    tag = np.random.randint(len(folder_list))
    folder = folder_list[tag]

    epoch = args.pt_from
    if args.pt_num > 1:
        epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
    ckpt = f'checkpoint{epoch}.pth.tar'

    file_dir = os.path.join(folder, ckpt)
    load_ckpt(model, file_dir, verbose=verbose)

def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, val_loader = load_resized_data(args)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # DREAM Setup
    if args.slct_type == 'dream':

        # Define model: used for dream initialization only
        model = define_model(args, nclass).to(device)
        model.train()
        optim_net = optim.SGD(model.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

        images_all = []
        labels_all = []
        images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
        labels_all = [trainset[i][1] for i in range(len(trainset))]
        
        images_all = torch.cat(images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        dataset = Data(images_all, labels_all)
        strategy_init = get_strategy('KMeansSampling')(dataset, model)
        query_list=torch.tensor(np.ones(shape=(nclass,args.batch_real)), dtype=torch.long, requires_grad=False, device=device)

        def get_init_images(c,n):
            query_idxs= strategy_init.query(c,n)
            return images_all[query_idxs]

    # Define real dataset and loader
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    resume_epoch = 0
    if args.load_checkpoint:
        logger("======= LOAD CHECKPOINT SETTING ========")
        resume_epoch = torch.load(os.path.join(args.save_dir, 'it.pt'))
        synset.data, _ = torch.load(os.path.join(args.save_dir, f'data_{resume_epoch}.pt'))
        synset.data = synset.data.cuda().requires_grad_(True)
        logger(f"RESUME FROM ITERATION: {resume_epoch}")
    else:
        if args.slct_type == 'dream':
            imgs = [[] for i in range(nclass)]
            for c in range(nclass):
                imgs[c] = get_init_images(c, synset.ipc * synset.factor**2).detach()
            synset.dream_init_images = imgs
        synset.init(loader_real, init_type=args.init)

    logger(f"init_size: {synset.data.size()}")
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    step = resume_epoch

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    if args.start_idx > 0:
        save_img(os.path.join(args.save_dir, f'aug.png'),
                aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
                unnormalize=True,
                dataname=args.dataset)

    # MDC Setup
    use_reg_flag = args.adaptive_reg
    if use_reg_flag:
        # create regularizer objects for each class
        regularizer_list = []
        for c in range(nclass):
            regularizer_list.append(Regularizer(args))
        freeze_ipc = -1

        # compute regularize index
        ipcx_index_class_dict = {}
        ipcs_list = list(range(1, args.ipc))
        args.adaptive_reg_list = ipcs_list
        for ipcx in ipcs_list:
            ipcx_num = ipcx * args.factor ** 2
            ipcx_index_class = [i for i in range(ipcx_num)]
            ipcx_index_class_dict[ipcx] = ipcx_index_class

    if not args.test:
        synset.test(args, val_loader, logger, aim_run=run, step=step, context=f"ipc{args.ipc}")
        if use_reg_flag and (args.ipc <= 10) and (args.factor < 3):  # test for the regularized ipc version
            for ipcx in set(args.adaptive_reg_list):
                ipcx_index_class = ipcx_index_class_dict[ipcx]
                synset.test(args, val_loader, logger, ipcx=-1, indices=ipcx_index_class, aim_run=run, step=step, context=f"ipc{ipcx}")

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
    period = 100
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    it_test = [i*period for i in range(args.niter//period+1)]
    it_test += [n_iter]

    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
    args.fix_iter = max(1, args.fix_iter)
    for init_loop in range(resume_epoch, n_iter):
        if init_loop % args.fix_iter == 0:
            model = define_model(args, nclass).to(device)
            model.train()
            optim_net = optim.SGD(model.parameters(),
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

            if args.pt_from >= 0:
                pretrain_sample(args, model)

            if args.early > 0:
                for _ in range(args.early):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                aug=aug_rand,
                                mixup=args.mixup_net)

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        loss_list_per_class = [[] for i in range(nclass)] # should have shape (nclass, )
        freeze_ipc_list = [-1 for i in range(nclass)]
        for model_loop in range(args.inner_loop):
            ts.set()
            if args.test:
                continue

            # Update synset
            for c in range(nclass):
                if args.slct_type == 'dream':  # use DREAM sampling
                    if model_loop % args.interval == 0:
                        strategy = get_strategy('KMeansSampling')(dataset, model)
                        query_index = strategy.query_match_sample(c,args.batch_real)
                        query_list[c] = query_index
                    img = images_all[query_list[c]]
                    lab = torch.tensor([np.ones(img.size(0))*c], dtype=torch.long, requires_grad=False, device=device).view(-1)
                else:
                    img, lab = loader_real.class_sample(c)
                img_syn, lab_syn, sampled_indices = synset.sample(c, max_size=args.batch_syn_max)
                ts.stamp("data")

                if use_reg_flag:
                    regularizer_list[c].update_status(init_loop)   # update regularizer status
                    freeze_ipc_list[c] = regularizer_list[c].get_freeze_ipc()

                    if freeze_ipc_list[c] > 0: # freeze the ipcx
                        freeze_ipc_idx = ipcx_index_class_dict[freeze_ipc_list[c]]
                        detached_img_syn = img_syn[freeze_ipc_idx].detach()
                        detached_img_syn.requires_grad = False

                        # replace the freeze ipcx with the detached version
                        img_syn[freeze_ipc_idx] = detached_img_syn

                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                # track the feature loss of each IPC
                condition1 = args.adaptive_reg
                condition2 = (init_loop == 0) or ((init_loop + 1) % args.adaptive_period == 0)
                if  condition1 and condition2:
                    # loss_list contains the loss of all ipcx in search space
                    loss_list = feat_loss_for_ipc_reg(args, img_aug[:n], img_aug[n:], model, indices=ipcx_index_class_dict)
                    loss_list_per_class[c].append(loss_list)

                if use_reg_flag:
                    reg_ipcx_list = regularizer_list[c].get_regularized_ipc()
                    loss = grad_loss_for_img_update(args, img_aug[:n], img_aug[n:], lab, lab_syn, model, ipcx_list=reg_ipcx_list, indices=ipcx_index_class_dict)
                else:
                    loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                    
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                ts.stamp("backward")

            # Net update
            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    top1, top5, model_loss = train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)
            ts.stamp("net update")

            if (model_loop + 1) % 10 == 0:
                ts.flush()

        if run and (not args.test):
            run.track(top1, name="top1", step=init_loop+1, context={"subset": "model"})
            run.track(model_loss, name="loss", step=init_loop+1, context={"subset": "model"})

        # Logging
        if (init_loop % it_log == 0) and (not args.test):
            logger(
                f"{utils.get_time()} (Iter {init_loop:3d}) loss: {loss_total/nclass/args.inner_loop:.1f}")

        # update the regularizer
        condition1 = args.adaptive_reg
        condition2 = (init_loop == 0) or ((init_loop + 1) % args.adaptive_period == 0)
        if condition1 and condition2:
            for c in range(nclass):
                if args.adaptive_class_wise:
                    cur_loss = torch.sum(tensor(loss_list_per_class[c]), axis=0)
                    cur_loss = torch.round(cur_loss/args.inner_loop, decimals=2).tolist()
                else:
                    flattened_loss_list = torch.Tensor(loss_list_per_class).reshape(-1, len(loss_list_per_class[0][0]))
                    cur_loss = torch.sum(flattened_loss_list, axis=0)
                    cur_loss = torch.round(cur_loss/args.nclass/args.inner_loop, decimals=2).tolist()

                regularizer_list[c].stats["cur_loss"] = cur_loss
                run.track(Text(f"{cur_loss}"), name="loss", step = init_loop+1, context={"subset": "cur_loss"})

                # update reg ipcx status
                ipcx_list = select_reg_ipc(args, regularizer_list[c], init_loop+1, logger=logger, aim_run=run)
                run.track(Text(f"Reg ipc: {ipcx_list}"), name="regularized_ipc", step=init_loop+1, context={"subset": f"class_{c}"})
                regularizer_list[c].update_ipc_prev_list()  # Clear prev list

                for ipcx in ipcx_list:
                    regularizer_list[c].regularize_ipcx(ipcx, prev=True)
                    regularizer_list[c].set_quit_iteration(ipcx, init_loop+1 + args.adaptive_period) # quit and freeze ipcx after adaptive_period iterations

                # update loss stats
                regularizer_list[c].history.append(cur_loss)
                regularizer_list[c].stats["prev_loss"] = cur_loss
                regularizer_list[c].stats["cur_loss"] = []

        condition1 = (init_loop + 1) in it_test
        condition2 = args.adaptive_reg and ((init_loop + 1) % args.adaptive_period == 0)
        if condition1 or condition2:
            torch.save(init_loop+1, os.path.join(args.save_dir, f'it.pt'))

            # save regularizer objectj as pickle file
            pickle.dump(regularizer_list, open(os.path.join(args.save_dir, f"regularizer_{init_loop+1}.pkl"), "wb"))

            # It is okay to clamp data to [0, 1] at here.
            # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data_{init_loop+1}.pt'))
            logger("img and data saved!")

            if not args.test:
                synset.test(args, val_loader, logger, aim_run=run, step=init_loop+1, context=f"ipc{args.ipc}")
                if args.adaptive_reg and (args.ipc <= 10) and (args.factor < 3):  # test for the regularized ipc version
                    for ipcx in set(args.adaptive_reg_list):
                        ipcx_index_class = ipcx_index_class_dict[ipcx]
                        synset.test(args, val_loader, logger, ipcx=-1, indices=ipcx_index_class, aim_run=run, step=init_loop+1, context=f"ipc{ipcx}")


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    # create time for log
    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    global run
    if args.use_aim:
        # check path exist: os.path.join(args.load_checkpoint, 'run_hash.pt')
        hash_path = os.path.join(args.save_dir, 'run_hash.pt')
        if os.path.exists(hash_path):
            run_hash = torch.load(hash_path)
            args.load_checkpoint = True
        else:
            run_hash = None
        run = Run(experiment=args.exp_name, repo=args.aim_repo, run_hash=run_hash)
        run.name = args.run_name
        if not args.test:
            torch.save(run.hash, os.path.join(args.save_dir, 'run_hash.pt'))

        hyperparams = dict()
        for key, value in vars(args).items():
            hyperparams.update({key: value})
        run["hparams"] = hyperparams

    else:
        run = None

    condense(args, logger)

    from misc.aim_export import aim_log
    if run:
        dir_name = args.save_dir
        aim_log(run, dir_name, args)
