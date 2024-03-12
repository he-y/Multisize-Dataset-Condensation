import os
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from train import define_model, train
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
import models.resnet as RN
import models.densenet_cifar as DN
from coreset import randomselect, herding
from efficientnet_pytorch import EfficientNet

DATA_PATH = "./results"

def search_file(path):
    for file in os.listdir(path):
        if file.endswith(".pt"):
            print("found file:", file)
            return file

def return_data_path(args):
    if args.factor > 1:
        init = 'mix'
    else:
        init = 'random'
    if args.dataset == 'imagenet' and args.nclass == 100:
        args.slct_type = 'idc_cat'
        args.nclass_sub = 20

    if 'idc' in args.slct_type:
        name = args.name
        if name == '':
            if args.dataset == 'cifar10':
                name = f'cifar10/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'cifar100':
                name = f'cifar100/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'imagenet':
                if args.nclass == 10:
                    name = f'imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor{args.factor}_{init}'
                elif args.nclass == 100:
                    name = f'imagenet100/resnet10apin_grad_l1_pt5_nd500_cut_nlr0.1_wd0.0001_factor{args.factor}_lr0.001_b_real128_{init}'

            elif args.dataset == 'svhn':
                name = f'svhn/conv3in_grad_mse_nd500_cut_niter2000_factor{args.factor}_lr0.005_{init}'
                if args.factor == 1 and args.ipc == 1:
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_cutout_scale_rotate'

            elif args.dataset == 'mnist':
                if args.factor == 1:
                    name = f'mnist/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'
                else:
                    name = f'mnist/conv3in_grad_l1_nd500_niter2000_factor{args.factor}_color_crop_lr0.0001_{init}'
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_scale_rotate'

            elif args.dataset == 'fashion':
                name = f'fashion/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'

        path_list = [f'{name}_ipc{args.ipc}']

    elif args.slct_type == 'dsa':
        path_list = [f'cifar10/dsa/res_DSA_CIFAR10_ConvNet_{args.ipc}ipc']
    elif args.slct_type == 'kip':
        path_list = [f'cifar10/kip/kip_ipc{args.ipc}']
    else:
        path_list = ['']

    return path_list


def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model


def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

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
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    # save_img(f"{args.save_dir}/test_dec.png", data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec


def load_data_path(args):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    print()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if 'idc' in args.slct_type:
            if args.slct_type == 'idc':
                if args.save_dir[-3:] != ".pt":
                    path = os.path.join(f'{args.save_dir}', f"data_{args.niter}.pt")
                    # check if file exists
                    if not os.path.isfile(path):
                        path = os.path.join(f'{args.save_dir}', f"data.pt")
                else:
                    path = args.save_dir
                    args.save_dir = os.path.dirname(args.save_dir)  # get the parent directory
                data, target = torch.load(path)

            elif args.slct_type == 'idc_cat':
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir, exist_ok=True)
                data_all = []
                target_all = []
                for idx in range(args.nclass // args.nclass_sub):
                    path = f'{args.save_dir}_{args.nclass_sub}_phase{idx}'
                    data, target = torch.load(os.path.join(path, 'data.pt'))
                    data_all.append(data)
                    target_all.append(target)
                    print(f"Load data from {path}")

                data = torch.cat(data_all)
                target = torch.cat(target_all)

            print("Load condensed data ", data.shape, args.save_dir)

            if args.factor > 1:
                data, target = decode(args, data, target)
            train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
            train_dataset = TensorDataset(data, target, train_transform)
        else:
            train_dataset = ImageFolder(traindir,
                                        train_transform,
                                        nclass=args.nclass,
                                        seed=args.dseed,
                                        slct_type=args.slct_type,
                                        ipc=args.ipc,
                                        load_memory=args.load_memory)
            print(f"Test {args.dataset} random selection {args.ipc} (total {len(train_dataset)})")
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)

    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        elif args.dataset == 'svhn':
            transform_fn = transform_svhn
        elif args.dataset == 'mnist':
            transform_fn = transform_mnist
        elif args.dataset == 'fashion':
            transform_fn = transform_fashion
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

        # Load condensed dataset
        if args.slct_type in ['idc', 'dream']:
            if args.save_dir[-3:] != ".pt":
                path = os.path.join(f'{args.save_dir}', f"data_{args.niter}.pt")
                # check if file exists
                if not os.path.isfile(path):
                    path = os.path.join(f'{args.save_dir}', f"data.pt")
            else:   # the path is already a pt file
                path = args.save_dir
                args.save_dir = os.path.dirname(args.save_dir)  # get the parent directory

            data, target = torch.load(path)
            print("Load condensed data ", args.save_dir, data.shape)
            # This does not make difference to the performance
            # data = torch.clamp(data, min=0., max=1.)
            if args.factor > 1:
                data, target = decode(args, data, target)

            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)

        elif args.slct_type == 'idc_cat':
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)
            data_all = []
            target_all = []
            for idx in range(args.nclass // args.nclass_sub):
                path = f'{args.save_dir}_{args.nclass_sub}_phase{idx}'
                data, target = torch.load(os.path.join(path, 'data_2000.pt'))
                data_all.append(data)
                target_all.append(target)
                print(f"Load data from {path}")

            data = torch.cat(data_all)
            target = torch.cat(target_all)

            print("Load condensed data ", data.shape, args.save_dir)
            if args.factor > 1:
                data, target = decode(args, data, target)

            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)

        elif args.slct_type in ['dc', 'dsa', 'kip', 'mtt']:
            condensed = torch.load(os.path.join(args.save_dir, search_file(args.save_dir)))

            if args.slct_type in ['dc', 'dsa']:
                condensed = condensed['data']
                # exp_idx=3 can recover 52.1 baseline of dsa for CIFAR-10 ipc10
                exp_idx=3
                data = condensed[exp_idx][0]  
                target = condensed[exp_idx][1]
            elif args.slct_type == 'kip':
                data = condensed[0].permute(0, 3, 1, 2)
                target = torch.arange(args.nclass).repeat_interleave(len(data) // args.nclass)
            elif args.slct_type == 'mtt':
                data = torch.load(os.path.join(f'{args.save_dir}', 'images_best.pt'))
                target = torch.load(os.path.join(f'{args.save_dir}', 'labels_best.pt'))
            else:
                raise NotImplementedError
            
            assert target.shape[0] == args.nclass * args.ipc * args.factor ** 2

            # These data are saved as the normalized values!
            train_transform, _ = transform_fn(augment=args.augment,
                                              from_tensor=True,
                                              normalize=False)
            train_dataset = TensorDataset(data, target, train_transform)
            print("Load condensed data ", args.save_dir, data.shape)

        else:
            if args.dataset == 'cifar10':
                train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                             train=True,
                                                             transform=train_transform)
            elif args.dataset == 'cifar100':
                train_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                              train=True,
                                                              transform=train_transform)
            elif args.dataset == 'svhn':
                train_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                          split='train',
                                                          transform=train_transform)
                train_dataset.targets = train_dataset.labels
            elif args.dataset == 'mnist':
                train_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                           train=True,
                                                           transform=train_transform)
            elif args.dataset == 'fashion':
                train_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                                  train=True,
                                                                  transform=train_transform)

            indices = randomselect(train_dataset, args.ipc, nclass=args.nclass)
            train_dataset = Subset(train_dataset, indices)
            print(f"Random select {args.ipc} data (total {len(indices)})")

        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       transform=test_transform)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        transform=test_transform)
        elif args.dataset == 'svhn':
            val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                    split='test',
                                                    transform=test_transform)
        elif args.dataset == 'mnist':
            val_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                     train=False,
                                                     transform=test_transform)
        elif args.dataset == 'fashion':
            val_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                            train=False,
                                                            transform=test_transform)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    # os.makedirs('./results', exist_ok=True)
    # save_img('./results/test.png',
    #          torch.stack([d[0] for d in train_dataset]),
    #          dataname=args.dataset)
    print()

    return train_dataset, val_dataset


def test_data(args,
              train_loader,
              val_loader,
              test_resnet=False,
              model_fn=None,
              repeat=1,
              logger=print,
              num_val=4,
              aim_run=None,
              step=None,
              context=None):
    """Train neural networks on condensed data
    """

    args.epoch_print_freq = args.epochs // num_val

    if model_fn is None:
        model_fn_ls = [define_model]
        if test_resnet:
            model_fn_ls = [resnet10_bn]
    else:
        model_fn_ls = [model_fn]

    for model_fn in model_fn_ls:
        best_acc_l = []
        acc_l = []
        list_of_result = []
        for _ in range(repeat):
            model = model_fn(args, args.nclass, logger=logger)
            best_acc, acc = train(args, model, train_loader, val_loader, logger=print)
            best_acc_l.append(best_acc)
            acc_l.append(acc)
            list_of_result.append([(model.state_dict(), best_acc, acc)])
        logger(
            f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}')
        # log standard deviation
        logger(
            f'Repeat {repeat} => Best, last std: {np.std(best_acc_l):.1f} {np.std(acc_l):.1f}\n')

        if args.eval:
            # save evaluation model
            torch.save(list_of_result, f'{args.save_dir}/{run.name}_{args.cur_time}.pt')
        
        if aim_run:
            if context is None:
                aim_run.track(np.mean(best_acc_l), name="best acc", step=step, context={"subset": model_fn.__name__})
                aim_run.track(np.std(best_acc_l), name="std", step=step, context={"subset": model_fn.__name__})
                # aim_run.track(np.mean(acc_l), name="last acc", step=step, context={"subset": model_fn.__name__})
            else:
                aim_run.track(np.mean(best_acc_l), name="best acc", step=step, context={"subset": context})
                aim_run.track(np.std(best_acc_l), name="std", step=step, context={"subset": context})
                # aim_run.track(np.mean(acc_l), name="last acc", step=step, context={"subset": context})
    return np.mean(best_acc_l), np.mean(acc_l)

def Cx_Cy(args, train_dataset=None, val_dataset=None):
    '''
    Test ipcy results on ipcx condensed dataset by extracting first 4 images from each class
    '''
    n_ipcy = args.ipcy * args.factor ** 2 
    n_ipcx = args.ipc * args.factor**2
    if train_dataset is None or val_dataset is None:
        print("Loading data from ", args.save_dir)
        train_dataset, val_dataset = load_data_path(args)

    classes = [i for i in range(args.nclass)]

    if args.dataset[:4] == 'svhn':
        train_transform, _ = transform_svhn(augment=args.augment, from_tensor=True)
    elif args.dataset[:5] == 'cifar':
        train_transform, _ = transform_cifar(augment=args.augment, from_tensor=True)
    elif args.dataset == 'imagenet':
        train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)

    data = torch.zeros((n_ipcy*args.nclass, 3, args.size, args.size), requires_grad=False)
    targets = torch.zeros((n_ipcy*args.nclass), dtype=torch.long, requires_grad=False)
    for c in classes:
        data[n_ipcy*c : n_ipcy*(c+1)] = train_dataset.images[n_ipcx*c: n_ipcx*c + n_ipcy]
        targets[n_ipcy*c : n_ipcy*(c+1)] = int(c)

    ipcy = TensorDataset(data, targets, transform=train_transform)
    
    return ipcy, val_dataset

def test_baseline_b(args):
    """
    make a `args.ipc` size dataset with numbers of SMALL_IPC datasets
    """
    SMALL_IPC=1

    data_all = []
    target_all = []
        
    for idx in range(args.ipc//SMALL_IPC):
        idx_tag = f'_idx_{idx}' if idx > 0 else ''
        path=f'{args.save_dir}{idx_tag}'
        data, target = torch.load(os.path.join(path,'data_2000.pt'))
        # before the data is decoded

        data_all.append(data)
        target_all.append(target)

    data = torch.cat(data_all)
    target = torch.cat(target_all)

    # select how many ipcs to use
    data = data[:args.nclass*args.ipcy]
    target = target[:args.nclass*args.ipcy]
    
    if args.dataset[:5] == 'cifar':
        train_transform, _ = transform_cifar(augment=args.augment, from_tensor=True)
    elif args.dataset[:4] == 'svhn':
        train_transform, _ = transform_svhn(augment=args.augment, from_tensor=True)
    elif args.dataset == 'imagenet':
        train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
    if args.factor > 1:
        data, target = decode(args, data, target)
    train_dataset = TensorDataset(data, target, transform=train_transform)

    return train_dataset, None

if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    import numpy as np
    import time
    cudnn.benchmark = True
    from aim import Run

    # set to eval mode
    args.eval = True
    # create a log file
    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    global run
    if args.exp_name != 'regularize': # default experiment name
        run = Run(experiment=args.exp_name)
    elif args.dataset == 'svhn':
        run = Run(experiment='eval-svhn')
    elif args.dataset == 'cifar10':
        run = Run(experiment='eval-cifar10')
    elif args.dataset == 'cifar100':
        run = Run(experiment='eval-cifar100')
    elif args.dataset == 'imagenet':
        run = Run(experiment='eval-imagenet')
    else:
        run = Run(experiment='eval')
    run.name = args.run_name

    if args.same_compute and args.factor > 1:
        args.epochs = int(args.epochs / args.factor**2)

    path_list = return_data_path(args)
    for p in path_list:
        args.save_dir = os.path.join(DATA_PATH, p)
        if args.test_type == 'herding':
            train_dataset, val_dataset = herding(args)
        elif args.test_type == 'cx_cy':
            args.save_dir = args.test_data_dir
            train_dataset, val_dataset = Cx_Cy(args)
            args.ipc = args.ipcy
        elif args.test_type == 'baseline_b':
            args.save_dir = args.test_data_dir
            train_dataset, _ = test_baseline_b(args)
            args.slct_type = 'none' # use original dataset
            _, val_dataset = load_data_path(args)
            args.ipc = args.ipcy
        else:
            args.save_dir = args.test_data_dir
            train_dataset, val_dataset = load_data_path(args)

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        val_loader = MultiEpochsDataLoader(val_dataset,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)

        test_data(args, train_loader, val_loader, repeat=args.repeat, test_resnet=False, aim_run=run)
        # if args.dataset[:5] == 'cifar':
        #     test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=resnet10_bn)
        #     if (not args.same_compute) and (args.ipc >= 50 and args.factor > 1):
        #         args.epochs = 400
        #     test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=densenet)
        # elif args.dataset == 'imagenet':
        #     test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=resnet18_bn)
        #     test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=efficientnet)

    from misc.aim_export import aim_log
    if run:
        dir_name = args.save_dir
        aim_log(run, dir_name, args)