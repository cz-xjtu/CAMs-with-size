import torch
import torch.nn as nn
import argparse
from network import resnet
from loadData import dataProstate
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from tqdm import tqdm
from losses import SizeConstraintLoss
import os


def calculate_hit(pred, gt):
    # batch_num = pred.shape[0]
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    hits = (pred == gt).sum()
    return hits


def do_epoch(phase, net, device, loader, epoch, optimizer=None):
    if phase == 'train':
        net.train()
        desc = f">> Training ({epoch})"
    elif phase == 'val':
        net.eval()
        desc = f">> Validating ({epoch})"

    total_iter = len(loader)
    total_imgs = len(loader.dataset)
    epoch_hit = 0.
    current_loss = 0.
    tqdm_iter = tqdm(total=total_iter, desc=desc, ncols=140, leave=False)
    for i, (names, images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        if optimizer:
            optimizer.zero_grad()
        # forward
        pred_logits, conv5_3_3, pred_cams = net.forward_cam(images)
        # pred_logits = net(images)

        # compute cams
        '''parm = {}
        bz, nc, h, w = conv5_3_3.shape
        for name, parameters in net.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        weight = parm['fc_cz.weight']
        cam = weight[0].dot(conv5_3_3.reshape(nc, h * w))
        cam = cam.reshape(h, w)'''

        pred_logits = torch.sigmoid(pred_logits)

        # compute loss
        loss_ce = bce_loss(pred_logits, labels)
        loss_size = size_loss(pred_cams, labels)
        loss = loss_ce + 0.01 * loss_size
        # loss = loss_ce
        current_loss += loss.item()

        # backward
        if optimizer:
            loss.backward()
            optimizer.step()

        # statistics logs
        batch_size = images.shape[0]
        batch_hits = calculate_hit(pred_logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
        epoch_hit += batch_hits
        current_acc = epoch_hit.__float__() / ((i+1) * batch_size)
        current_loss = current_loss / (i+1)

        # logging

        tqdm_iter.set_postfix({'loss': f"{current_loss:.4f}", 'loss_ce': f"{loss_ce.item():.4f}",
                               'loss_size': f"{loss_size.item():.4f}", 'accuracy': f"{current_acc:.4f}"})
        # tqdm_iter.set_postfix({'loss': f"{current_loss:.4f}", 'loss_ce': f"{loss_ce.item():.4f}",
        #                        'accuracy': f"{current_acc:.4f}"})
        tqdm_iter.update(1)
    tqdm_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in {'loss': f"{current_loss:.4f}", 'accuracy': f"{current_acc:.4f}"}.items()))
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of CAMs-with-size")
    parser.add_argument('--data_root', type=str, default='/data/cz/dataset/prostate_MR/DL_Image')
    parser.add_argument('--train_list', type=str, default='prostate_MR/train_all_img.txt')
    parser.add_argument('--val_list', type=str, default='prostate_MR/val_all_img.txt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--network', type=str, default='resnet_50', help='')
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_ids', type=list, default=[1])
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--model_save_path', type=str, default="res50_model/constraint")
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
    net.to(device)

    # define loss function
    bce_loss = nn.BCELoss()
    size_loss = SizeConstraintLoss(idc=[0], values={0: [240, 36000]})
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
        do_epoch(phase='train', net=net, device=device, loader=train_loader, epoch=i, optimizer=optimizer_ft)
        with torch.no_grad():
            do_epoch(phase='val', net=net, device=device, loader=val_loader, epoch=i)
        torch.save(net, os.path.join(args.model_save_path, "best.pkl"))

    print('Bingo!')








