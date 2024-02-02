import os
import time
import argparse
import datetime
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image
from model import get_net
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei'] # Support displaying Chinese
#plt.rcParams['axes.unicode_minus'] = False

seed = 12345

import torch.nn.functional as F

def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().detach().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels, classes):
    preds, probs = images_to_probs(net, images)

    fig = plt.figure(figsize=(12, 48))

    n_images = images.shape[0]

    for idx in np.arange(n_images):
        ax = fig.add_subplot(1, n_images, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx].cpu().detach(), one_channel=False)

        clazz_pred = classes[preds[idx]]
        clazz_true = classes[labels[idx].cpu().detach().numpy().item()]
        prob_pred = probs[idx] * 100.0
        color = 'green' if preds[idx]==labels[idx].item() else 'red'

        ax.set_title(f'pred={clazz_pred}, {prob_pred:.2f}\ntrue={clazz_true}', color=color)

    return fig

def initialize_dataset(image_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    classes = []
    image_files = []
    image_labels = []

    trainfilepath = os.path.join(args.savedir, "imagelist.txt")
    labelfilepath = os.path.join(args.savedir, "labels.txt")

    if not os.path.exists(trainfilepath) or not os.path.exists(labelfilepath):
        classes = os.listdir(image_dir)

        for i, classitem in enumerate(classes):
            classitemdir = os.path.join(image_dir, classitem)
            filepaths = [os.path.join(classitem, filepath) for filepath in os.listdir(classitemdir)]
            image_files.extend(filepaths)
            image_labels.extend([i]*len(filepaths))

        f = open(trainfilepath, "w")
        for image_file, image_label in zip(image_files, image_labels):
            print(f"{image_label} {image_file}", file=f)
        f.close()

        f = open(labelfilepath, "w")
        for i, classitem in enumerate(classes):
            print(f"{classitem} {i}", file=f)
        f.close()
    else:
        f = open(trainfilepath, "r")
        lines = [line.strip() for line in f.readlines()]
        f.close()

        for line in lines:
            blankidx = line.find(' ')
            image_files.append(line[blankidx+1:])
            image_labels.append(int(line[:blankidx]))

        f = open(labelfilepath, "r")
        lines = [line.strip() for line in f.readlines()]
        f.close()

        for line in lines:
            blankidx = line.find(' ')
            classes.append(line[:blankidx])

    return classes, image_files, image_labels

class ImageFolderEx(Dataset):
    def __init__(self, image_dir, image_files, image_labels, classnum=1000, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.image_labels = image_labels
        self.classnum = classnum
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = os.path.join(self.image_dir, self.image_files[index])  
        image = cv2.imread(image_name)
        if image is None:
            print(image_name)
            image = np.zeros((224,224,3), dtype=np.uint8)
        image = image[:,:,::-1]
        image = Image.fromarray(image)
        label = self.image_labels[index]
        if self.transform:
            image = self.transform(image)
        onehot = [0]*self.classnum
        onehot[label] = 1
        return (image, np.array(onehot).astype(np.float32))   

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(net, classes, trainloader, valloader, lr=0.001, epochs=1, backbone='resnet101'):
    print(f"Training on {device.type}")

    logging = SummaryWriter(os.path.join(args.savedir, f"runs/{backbone}"))

    cutmix = v2.CutMix(num_classes=len(classes))
    mixup = v2.MixUp(num_classes=len(classes))
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        print("Using Adam optimizer by default...")
        optimizer = optim.Adam([{'params': net.parameters(), 'lr': lr}])

    if args.lrsched == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs, eta_min=0.00001, last_epoch=-1
            )
    else:
        print("Using StepLR scheduler by default...")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,  gamma=0.1)

    if args.losstype == 'focalloss':
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    step_count = 0

    for epoch in range(epochs): 
        print(f'Epoch {epoch}/{epochs}')

        # Train Stage
        net.train()
        net = net.to(device)
        train_running_loss = 0.0
        train_running_corrects = 0.
        val_running_loss = 0.0
        val_running_corrects = 0.

        progress_bar = tqdm(range(0, len(trainloader)))
        progress_bar.set_description("Train steps")

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_aug,labels_aug = inputs,labels
            if random.random() < args.mixratio:
                inputs_aug, labels_aug = cutmix_or_mixup(inputs, torch.max(labels,1)[1])
            optimizer.zero_grad()
            outputs = net(inputs_aug)
            _, preds = torch.max(outputs, 1)
            train_loss = criterion(outputs, labels_aug)
            train_loss.backward()
            optimizer.step()
            step_count += 1
            train_running_loss += train_loss.item()
            logging.add_scalar('Training running loss',
                            train_loss.item(),
                            step_count)
            train_running_corrects += torch.sum(preds == torch.max(labels_aug,1)[1])

            progress_bar.update(1)
            progress_bar.set_postfix(**{'train_loss': train_loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        scheduler.step()

        train_loss = train_running_loss / len(trainloader.dataset)
        train_acc = train_running_corrects.item() / len(trainloader.dataset)

        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")

        # Validation Stage
        net.eval()

        progress_bar = tqdm(range(0, len(valloader)))
        progress_bar.set_description("Validate steps")

        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(preds == torch.max(labels,1)[1])

            progress_bar.update(1)
            progress_bar.set_postfix(**{'val_loss': val_loss.item()})
        
        val_loss = val_running_loss / len(valloader.dataset)
        val_acc = val_running_corrects.item() / len(valloader.dataset)

        logging.add_figure(
                'Predictions vs. Actuals',
                 plot_classes_preds(net, inputs[:5], torch.max(labels,1)[1][:5], classes),
                 global_step=epoch
                 )

        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            PATH = os.path.join(ckptdir, f"{backbone}-epoch{str(epoch).zfill(3)}.pth")
            torch.save(net.state_dict(), PATH)

    print(f'Training complete - model saved to {PATH}')

    logging.close()

def main(device, backbone, lr=0.001, epochs=1, num_workers=1, ckptpath=None):
    classes, image_files, image_labels = initialize_dataset(args.datadir)
    result = list(zip(image_files, image_labels))
    np.random.shuffle(result)
    image_files, image_labels = zip(*result)
    trainlen = len(image_files)*6//7
    classnum = len(classes)

    print(f"Using backbone: {backbone}, classnum: {classnum}")

    net = get_net(backbone, 3, classnum)

    if ckptpath is not None:
        print(f"finetuing from {args.ckptpath}...")
        net.load_state_dict(torch.load(args.ckptpath))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = ImageFolderEx(
            args.datadir,
            image_files[:trainlen],
            image_labels[:trainlen],
            classnum,
            transforms.Compose([
                transforms.RandomRotation([-13,13]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    nn.ModuleList([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        transforms.RandomGrayscale(),
                        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                        transforms.RandomAffine(degrees=0, shear=(-10, 10)),
                        ]),
                    p=0.5
                    ),
                transforms.Resize(args.imgsz+32*(args.imgsz//256+1)),
                transforms.RandomCrop(args.imgsz),
                transforms.RandomApply(nn.ModuleList([
                        transforms.AugMix(severity= 6, mixture_width=2),
                    ]),
                    p=0.5
                    ),
                transforms.ToTensor(),
                normalize
            ]))

    val_data = ImageFolderEx(
            args.datadir,
            image_files[trainlen:],
            image_labels[trainlen:],
            classnum,
            transforms.Compose([
                transforms.Resize(args.imgsz),
                transforms.CenterCrop(args.imgsz),
                transforms.ToTensor(),
                normalize
            ]))

    train_loader = DataLoader(
            train_data, batch_size=args.train_bs, shuffle=True, num_workers=num_workers
            )
    val_loader = DataLoader(val_data, batch_size=args.val_bs, shuffle=False, num_workers=num_workers)

    train(net, classes, train_loader, val_loader, lr=lr, epochs=epochs, backbone=backbone)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet101", type=str, help="Backbone, options: resnet50, resnet101, efficientnet-b4...")
    parser.add_argument("--datadir", default=None, type=str, help="Dataset path")
    parser.add_argument("--imgsz", default=224, type=int, help="Input image size")
    parser.add_argument("--train_bs", default=32, type=int, help="Train batch size")
    parser.add_argument("--val_bs", default=32, type=int, help="Val batch size")
    parser.add_argument("--workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--ckptpath", default=None, type=str, help="Finetune from ckptpath")
    parser.add_argument("--savedir", default="./exp", type=str, help="Saving directory")
    parser.add_argument("--mixratio", default=0.3, type=float, help="Mix ratio for CutMix and MixUp")
    parser.add_argument("--losstype", default='cross entropy', type=str, help="Loss type")
    parser.add_argument("--optim", default='SGD', type=str, help="Optimizer type, options: SGD, Adam")
    parser.add_argument("--lrsched", default='CosineAnnealingLR', type=str, help='Learning rate scheduler, options: CosineAnnealingLR, StepLR')

    args = parser.parse_args()
 
    ckptdir = os.path.join(args.savedir, "ckpts")

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(
        device, args.backbone, args.lr, args.epochs, args.workers, args.ckptpath
        )
