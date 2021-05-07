import os
import time, math
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as TF
from torchvision.transforms import FiveCrop, Compose, ToTensor, Lambda

import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IMAGE_SIZE = 256
NUM_CLASSES = 10
NUM_FEATURES = 100

def label2onehot(y_train):
    y_onehot = torch.zeros((y_train.shape[0], NUM_CLASSES), dtype=torch.long)
    y_onehot[torch.arange(y_train.shape[0]),torch.squeeze(y_train)] = 1
    return y_onehot

class ImageWoofDataset(Dataset):
    """Class to store ImageWoof dataset"""
    def __init__(self, all_images):
        self.all_images = all_images
        label = 0
        self.files = []
        self.labels = []
        for image_dir in os.listdir(all_images):
            for file in os.listdir(os.path.join(all_images, image_dir)):
                self.files.append(os.path.join(image_dir, file).strip())
                self.labels.append(label)
            label += 1
        self.transform = TF.Compose([TF.RandomCrop(IMAGE_SIZE), TF.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = torch.LongTensor([self.labels[idx]])
        image = Image.open(os.path.join(self.all_images, fname), "r").convert('RGB')

        ## RAND CROP
        if image.size[0] < IMAGE_SIZE or image.size[1] < IMAGE_SIZE:
            image = TF.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)
        image = self.transform(image)

        return image, label

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.rnns = nn.ModuleList([nn.RNN(256, 256, num_layers=2, bidirectional=True)
                                   for i in range(4)])
        self.linear = nn.Sequential(nn.Linear(82944, NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x_quad = self.image_to_quad(x)
        rnn_out = []
        for i in range(len(x_quad)):
            out, _ = self.rnns[i](x_quad[i].reshape(-1, x.shape[0], x.shape[1]))
            rnn_out.append(out.reshape(x.shape[0], -1))
        rnn_out = torch.cat(rnn_out, 1)
        out = torch.cat((rnn_out, x.reshape(x.shape[0], -1)), 1)
        return self.linear(out)

    def image_to_quad(self, x: torch.Tensor):
        """Reshapes image to get four flattened arrays representing
        different directions of traversal
        inputs:
            x: shape (batch size, channels, width, height)
        """
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3], -1)
        x0 = torch.flatten(x, 1, 2)
        x1 = torch.flatten(torch.flip(x, [1,2]), 1, 2)
        x = torch.transpose(x, 1, 2)
        x2 = torch.flatten(x, 1, 2)
        x3 = torch.flatten(torch.flip(x, [1,2]), 1, 2)
        return x0, x1, x2, x3


def train_cnn(model, train_batches, test_batches, num_epochs = 30):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    all_train_acc, all_test_acc = [], []
    all_train_loss, all_test_loss = [], []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_acc = []
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            images = images.cuda()
            labels = torch.squeeze(labels).cuda()
            if len(labels.shape) == 0:
                labels = torch.unsqueeze(labels, dim=0)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_acc.append(accuracy(output, labels))
            total_loss += loss.cpu().item()
        print("Training loss after epoch {}: {}".format(epoch, total_loss/len(train_batches)))
        print("Training accuracy after epoch {}: {}".format(epoch, np.mean(epoch_acc, axis=0)))
        all_train_acc.append(np.mean(epoch_acc, axis=0))
        all_train_loss.append(total_loss/len(train_batches))
        test_loss, test_acc = test_cnn(model, test_batches, criterion)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

    return all_train_loss, np.array(all_train_acc), all_test_loss, np.array(all_test_acc)

def accuracy(output, target, topk=(1,2,3)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def test_cnn(model, test_batches, criterion):
    with torch.no_grad():
        all_accuracy = []
        total = 0.0
        total_loss = 0.0
        for i, batch in enumerate(test_batches):
            images, labels = batch
            images = images.cuda()
            labels = torch.squeeze(labels).cuda()
            if len(labels.shape) == 0:
                labels = torch.unsqueeze(labels, dim=0)
            output = model(images)
            loss = criterion(output, labels)
            all_accuracy.append(accuracy(output, labels))
            total_loss += loss.item()
    return total_loss/len(test_batches), np.mean(all_accuracy, axis=0)

def get_plots(train_a, test_a, train_l, test_l, epochs = 30):
    x = np.arange(1, epochs+1)
    plt.plot(x, train_a[:,0])
    plt.plot(x, train_a[:,1])
    plt.plot(x, train_a[:,2])
    plt.plot(x, test_a[:,0])
    plt.plot(x, test_a[:,1])
    plt.plot(x, test_a[:,2])
    plt.legend(["Top-1 Train", "Top-2 Train", "Top-3 Train", "Top-1 Test", "Top-2 Test", "Top-3 Test"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("rnn_cnn_accuracy.png")
    plt.close()

    plt.plot(x, train_l)
    plt.plot(x, test_l)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("rnn_cnn_loss.png")
    plt.close()

if __name__ == "__main__":
    seed = 10
    batch_size = 16
    num_epochs = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    path_to_train = "imagewoof2/train"
    path_to_val = "imagewoof2/val"

    def crop_collate(batch): # used for 5 crop
        imgs,targets = zip(*batch)
        return torch.cat(imgs),torch.cat(targets)

    train_dataset = ImageWoofDataset(path_to_train)
    test_dataset = ImageWoofDataset(path_to_val)
    train_batches = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True)#, collate_fn=crop_collate)
    test_batches = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True)#, collate_fn=crop_collate)

    cnn = AlexNet()
    cnn.cuda()
    train_l, train_a, test_l, test_a = train_cnn(cnn, train_batches, test_batches, num_epochs)
    np.save("dump/rnn_cnn_trainl.npy", train_l)
    np.save("dump/rnn_cnn_traina.npy", train_a)
    np.save("dump/rnn_cnn_testl.npy", test_l)
    np.save("dump/rnn_cnn_testa.npy", test_a)
    get_plots(train_a, test_a, train_l, test_l, epochs = num_epochs)

    torch.save(cnn, "rnn_cnn.pt")
