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

from cnn import AlexNet

IMAGE_SIZE = 256
NUM_CLASSES = 10
NUM_FEATURES = 100
BATCH_SIZE = 8

torch.autograd.set_detect_anomaly(True)

def label2onehot(y_train):
    # One hot encoding buffer that you create out of the loop and just keep reusing
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

class ConvRNN(nn.Module):
    def __init__(self, in_f, out_f, kernel, stride, padding, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size=kernel,
                              stride=stride, padding=padding)
        self.rnns = nn.ModuleList([nn.RNN(out_f, out_f//4, num_layers=1, bidirectional=True)
                                   for i in range(4)])
        self.relu = nn.ReLU(inplace=False)
        self.pool_bool = pool
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        rnn_out = []
        x_quad = self.image_to_quad(x)
        for i in range(len(self.rnns)):
            rnn_in = x_quad[i].reshape(-1, x.shape[0], x.shape[1])
            _, out = self.rnns[i](rnn_in)
            out = out.reshape(x.shape[0], -1)
            rnn_out.append(out)
        x = self.relu(x)
        if self.pool_bool:
            x = self.pool(x)
        out = torch.cat(rnn_out, dim=-1)
        return x, out

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

class AlexRNN(nn.Module):
    """Class defining AlexNet layers used for the convolutional network"""

    def __init__(self):
        super(AlexRNN, self).__init__()
        ## Define params of AlexNet
        in_f = [3, 64, 192, 384, 256]
        out_f = [64, 192, 384, 256, 256]
        kernel_sizes = [11, 5, 3, 3, 3]
        strides = [2, 1, 1, 1, 1]
        paddings = [4, 2, 1, 1, 1]
        pool_bool = [True, True, False, False, True]

        params = np.stack((in_f, out_f, kernel_sizes, strides, paddings, pool_bool), axis=-1)
        self.convrnn_blocks = nn.ModuleList([ConvRNN(in_f, out_f, k, s, p, pb) for
                                             in_f, out_f, k, s, p, pb in params])
        self.classifier = nn.Linear(2304, NUM_CLASSES)

    def forward(self, x):
        rnn_outs = []
        for block in self.convrnn_blocks:
            x, out = block(x)
            rnn_outs.append(out)
        rnn_out = torch.cat(rnn_outs, dim=-1)
        result = self.classifier(rnn_out)
        return result

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
    plt.savefig("rnn_all_accuracy.png")
    plt.close()

    plt.plot(x, train_l)
    plt.plot(x, test_l)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("rnn_all_loss.png")
    plt.close()

if __name__ == "__main__":
    seed = 10
    num_epochs = 5
    np.random.seed(seed)
    torch.manual_seed(seed)
    path_to_train = "imagewoof2/train"
    path_to_val = "imagewoof2/val"

    cnn_weights = torch.load("models/cnn.pt")
    train_dataset = ImageWoofDataset(path_to_train)
    test_dataset = ImageWoofDataset(path_to_val)
    train_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)

    cnn = AlexRNN()
    print(cnn)
    cnn.cuda()
    train_l, train_a, test_l, test_a = train_cnn(cnn, train_batches, test_batches, num_epochs)
    np.save("dump/rnn_all_trainl.npy", train_l)
    np.save("dump/rnn_all_traina.npy", train_a)
    np.save("dump/rnn_all_testl.npy", test_l)
    np.save("dump/rnn_all_testa.npy", test_a)
    get_plots(train_a, test_a, train_l, test_l, epochs = num_epochs)
