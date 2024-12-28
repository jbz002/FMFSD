from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                   "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default="imagenet", type=str, help="cifar100|cifar10")
parser.add_argument('--epoch', default=100, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.1, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--autoaugment', default=False, type=bool)
parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
args = parser.parse_args(args=[])
print(args)

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


if args.autoaugment:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    transform_train = transforms.Compose([transforms.RandomRotation(20),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])


data_dir = '../input/tiny-imagenet/tiny-imagenet-200'
dataset_train = TinyImageNet(data_dir, train=True)
dataset_val = TinyImageNet(data_dir, train=False)


if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )
elif args.dataset == "imagenet":
    trainset = TinyImageNet(data_dir, train=True, transform=transform_train)
    testset = TinyImageNet(data_dir, train=False, transform=transform_test)
    
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=2
)

if args.model == "resnet18":
    net = resnet18()
if args.model == "resnet34":
    net = resnet34()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()
if args.model == "wideresnet50":
    net = wide_resnet50_2()
if args.model == "wideresnet101":
    net = wide_resnet101_2()
if args.model == "resnext50_32x4d":
    net = resnet18()
if args.model == "resnext101_32x8d":
    net = resnext101_32x8d()

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
init = False

if __name__ == "__main__":
    best_acc = 0
    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0
    for epoch in range(args.epoch):
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        if epoch in [args.epoch // 3, args.epoch * 2 // 3, args.epoch - 10]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, outputs_feature = net(inputs)
            ensemble = sum(outputs[:-1])/len(outputs)
            ensemble.detach_()

            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = outputs_feature[0].size(1)
                for index in range(1, len(outputs_feature)):
                    student_feature_size = outputs_feature[index].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                net.adaptation_layers = nn.ModuleList(layer_list)
                net.adaptation_layers.cuda()
                optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
                #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   for deepest classifier
            loss += criterion(outputs[0], labels)

            teacher_output = outputs[0].detach()
            teacher_feature = outputs_feature[0].detach()

            #   for shallow classifiers
            for index in range(1, len(outputs)):
                #   logits distillation
                loss += CrossEntropy(outputs[index], teacher_output) * args.loss_coefficient
                loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
                #   feature distillation
                if index != 1:
                    loss += torch.dist(net.adaptation_layers[index-1](outputs_feature[index]), teacher_feature) * \
                            args.feature_loss_coefficient
                    #   the feature distillation loss will not be applied to the shallowest classifier

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            outputs.append(ensemble)

            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
#             print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
#                   ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
#                                           100 * correct[0] / total, 100 * correct[1] / total,
#                                           100 * correct[2] / total, 100 * correct[3] / total,
#                                           100 * correct[4] / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct = [0 for _ in range(5)]
            predicted = [0 for _ in range(5)]
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, outputs_feature = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.append(ensemble)
                for classifier_index in range(len(outputs)):
                    _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                  ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                         100 * correct[2] / total, 100 * correct[3] / total,
                                         100 * correct[4] / total))
            
            if correct[3] / total > best_acc3:
                best_acc3 = correct[3]/total
                print("bran1 Best Accuracy Updated: ", best_acc3 * 100)
            if correct[2] / total > best_acc2:
                best_acc2 = correct[2]/total
                print("bran2 Accuracy Updated: ", best_acc2 * 100)
            if correct[1] / total > best_acc1:
                best_acc1 = correct[1]/total
                print("bran3 Accuracy Updated: ", best_acc1 * 100)
            if correct[0] / total > best_acc:
                best_acc = correct[0]/total
                print("Best Accuracy Updated: ", best_acc * 100)
                torch.save(net.state_dict(), "./"+str(args.model)+".pth")

    print("Training Finished, TotalEPOCH=%d,main Best Accuracy=%.3f,bran3 Best Accuracy=%.3f,bran2 Best Accuracy=%.3f,bran1 Best Accuracy=%.3f" % (args.epoch, best_acc,best_acc1,best_acc2,best_acc3))