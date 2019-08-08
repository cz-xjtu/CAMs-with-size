import torch
import torch.nn as nn
import argparse
from network import resnet
from loadData import dataProstate
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from functools import partial
from tqdm import tqdm


def do_epoch(phase, net, device, loader, epoch, optimizer):
    if phase == 'train':
        net.train()
        desc = f">> Training ({epoch})"
    elif phase == 'val':
        net.eval()
        desc = f">> Validating ({epoch})"

    total_iter = len(loader)
    total_imgs = len(loader.dataset)
    tqdm_iter = tqdm(total=total_iter, desc=desc, ncols=100, leave=False)
    for i, (names, images, labels) in enumerate(loader):
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # Reset gradients
        if optimizer:
            optimizer.zero_grad()
        # forward
        pred_logits, pred_cams = net(images)

        # compute loss
        loss_ce = bce_loss(pred_logits, labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of CAMs-with-size")
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')
    parser.add_argument('--val_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--network', type=str, default='resnet_50', help='')
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--model_save_path', type=str, default="res50_model")
    args = parser.parse_args()

    # setting up the network
    print('\n>>> Setting up ...')
    # default use cpu
    use_gpu = False
    device = torch.device('cpu')
    if args.gpu_ids:
        use_gpu = True
        device = torch.device('cuda:{}'.format(args.gpu_ids[0]))
    # if use weights pre-trained
    if args.resume:
        if use_gpu:
            net = torch.load(args.resume)
        else:
            net = torch.load(args.resume, map_location='cpu')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print(f'>> Restored weights from {args.resume} successfully.')
    else:
        net_class = getattr(resnet, args.network)
        net = net_class(num_classes=args.num_class)
        # multi-gpu
        if len(args.gpu_ids) > 1:
            net.to(device)
            net = torch.nn.parallel.DataParallel(net, device_ids=args.gpu_ids)

    # define loss function
    bce_loss = nn.BCELoss()
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # get data loader
    myTransforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.203, 0.203, 0.203], [0.191, 0.191, 0.191])
    ])
    train_dataset = dataProstate.ClsDataset(args.train_list, data_root=args.data_root, transform=myTransforms)
    val_dataset = dataProstate.ClsDataset(args.val_list, data_root=args.data_root, transform=myTransforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # train and val
    for i in range(args.num_epochs):
        # train and val alternatively
        do_epoch()







