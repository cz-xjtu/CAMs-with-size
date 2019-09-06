import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from loadData import dataProstate
import numpy as np
import cv2
import os


def infer_cam(args, model, device):
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))
    # read data
    infer_dataset = dataProstate.ClsDataset(args.val_list, data_root=args.data_root, transform=transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.203, 0.203, 0.203], [0.191, 0.191, 0.191])
    ]))
    infer_dataloader = DataLoader(infer_dataset, shuffle=False)
    acc_pos_num = 0.
    pos_num = 0.
    for i, (img_name, inputs, labels) in enumerate(infer_dataloader):
        img_name = img_name[0]
        label = labels[0]
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.to(device)
        else:
            inputs = inputs
        # label = label[0]
        # img_path = os.path.join(args.data_root, img_name + '.png')
        # orig_img = np.asarray(Image.open(img_path))
        # orig_img_size = orig_img.shape[:2]
        # get the features of final conv layer
        outputs, conv5_3_3, pred_cams = model.forward_cam(inputs)
        print(img_name)
        img = cv2.imread(os.path.join(args.data_root, img_name + '.png'))
        height, width, _ = img.shape
        for j in range(pred_cams.shape[0]):
            cams = cv2.resize(pred_cams[j][0].detach().cpu().numpy(), (width, height))
            heatmap = cv2.applyColorMap(np.uint8(cams * 255), cv2.COLORMAP_JET)
            result = heatmap * 0.5 + img * 0.5
            cv2.imwrite(os.path.join(args.save_root, img_name + '_' + str(int(label)) + '.png'), result)

        '''bz, nc, h, w = conv5_3_3.shape
        conv5_3_3 = conv5_3_3.detach().cpu().numpy()
        # get the weight of final fc layer
        parm = {}
        for name, parameters in model.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        weight = parm['fc_cz.weight']
        # generate the class activation maps upsample to size(256,256)
        upsample_size = (512, 512)
        output_cam = []
        for idx in range(args.num_class):
            cam = weight[idx].dot(conv5_3_3.reshape(nc, h*w))
            #cam = np.sum(conv5_3_3.reshape(nc, h*w), 0)
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, upsample_size))

        # idx = idx.cpu().numpy()
        for cn in range(1):
            if True:
                print(img_name)
                img = cv2.imread(os.path.join(args.data_root, img_name + '.png'))
                height, width, _ = img.shape
                cams = cv2.resize(output_cam[0], (width, height))
                heatmap = cv2.applyColorMap(cams, cv2.COLORMAP_JET)
                result = heatmap * 0.5 + img * 0.5
                cv2.imwrite(os.path.join(args.save_root, img_name + '_' + str(int(label)) + '.png'), result)
                #cv2.imwrite(os.path.join(args.save_root, img_name + '_' + str(int(labels)) + '.png'), cams)'''

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of CAMs-with-size")
    parser.add_argument('--data_root', type=str, default='/data/cz/dataset/prostate_MR/DL_Image')
    parser.add_argument('--train_list', type=str, default='prostate_MR/train_all_img.txt')
    parser.add_argument('--val_list', type=str, default='prostate_MR/train_all_img.txt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--network', type=str, default='resnet_50', help='')
    parser.add_argument('--resume', type=str, default="res50_model/constraint_wait_1e-3/best.pkl", help="For training from one checkpoint")
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_ids', type=list, default=[1])
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--model_save_path', type=str, default="res50_model/")
    parser.add_argument('--save_root', type=str, default="cams/constraint_wait_1e-3/", help="path for saving cams")
    args = parser.parse_args()

    net = torch.load(args.resume)
    net.eval()
    device = torch.device('cuda:{}'.format(args.gpu_ids[0]))
    net.to(device)

    infer_cam(args, net, device)
