from __future__ import print_function
import argparse, os, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.transform import rotate as skrotate
from scipy.linalg import circulant
from tqdm import tqdm

from PIL import Image
import torchvision.transforms as TF
from torchvision.transforms import FiveCrop, Compose, ToTensor, Lambda

import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .hypercolumns import Hypercolumns

IMAGE_SIZE = 256
NUM_CLASSES = 10
NUM_FEATURES = 100

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
    def __init__(self, hypercolumns = False, num_classes: int = 10) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x

class GraphNet(nn.Module):
    def __init__(self, img_size=6, num_features = 256, filter = "gaussian", hypercolumns=False):
        super(GraphNet, self).__init__()
        self.pred_edge = False
        self.num_features = num_features
        self.N = img_size ** 2
        self.classifier = nn.Linear(9216, NUM_CLASSES)
        self.backbone = AlexNet(hypercolumns)
        if filter == "pred":
            self.pred_edge = True
            col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
            coord = np.stack((col, row), axis=2).reshape(-1, 2)
            coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)
            coord = torch.from_numpy(coord).float()  # 784,2
            coord = torch.cat((coord.unsqueeze(0).repeat(self.N, 1,  1),
                                    coord.unsqueeze(1).repeat(1, self.N, 1)), dim=2)
            self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                              nn.Linear(64, 1))
            self.register_buffer('coord', coord)
        elif filter == "gaussian":
            # precompute adjacency matrix for each channel before training
            A = self.precompute_adjacency_images(img_size) # 36 x 36
            self.register_buffer('A', A)
        elif filter == "gabor":
            A = self.precompute_Gabor(img_size)
            self.register_buffer('A', A)


    @staticmethod
    def precompute_adjacency_images(img_size):
        col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / img_size
        dist = cdist(coord, coord)
        sigma = 0.05 * np.pi

        A = np.exp(- dist**2 / sigma ** 2)
        A[A < 0.01] = 0
        A = torch.from_numpy(A).float()

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2
        return A_hat

    @staticmethod
    def precompute_Gabor(img_size):
        # Create Gabor filter
        N = img_size
        x, y = np.meshgrid(np.arange(-float(N), N), np.arange(-float(N), N))
        y = skrotate(y, 35)
        x2 = skrotate(x, -35)
        sigma = 0.75 * np.pi
        lmbda = 1.5 * sigma
        gamma = 1.3
        gabor = np.exp(-(x**2 + gamma*y**2)/(2*sigma**2))*np.cos(2*np.pi*x2/lmbda)

        # Create the adjacency matrix based on the Gabor filter without any zero-padding
        A = np.zeros((N ** 2, N ** 2))
        for i in range(N):
            for j in range(N):
                A[i*N + j, :] = gabor[N - i:N - i + N, N - j:N - j + N].flatten()

        A[A < 0.01] = 0
        A = torch.from_numpy(A).float()

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2
        return A_hat

    def forward(self, x):
        x = self.backbone(x)
        B = x.size(0)
        if self.pred_edge:
            self.A = self.pred_edge_fc(self.coord).squeeze() # 36 x 36
        mat1 = self.A.unsqueeze(0).expand(B, self.num_features,-1,-1)
        mat2 = x.view(B, self.num_features, -1, 1)
        avg_neighbor_features = torch.matmul(mat1, mat2).reshape(B, -1)
        ## consider flattening array
        logits = self.classifier(avg_neighbor_features)
        return logits


def train_cnn(model, train_batches, test_batches, num_epochs = 30):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
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

def get_plots(train_a, test_a, train_l, test_l, filter, epochs = 30):
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
    plt.savefig("gnnlast_{}_accuracy.png".format(filter))
    plt.close()

    plt.plot(x, train_l)
    plt.plot(x, test_l)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("gnnlast_{}_loss.png".format(filter))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN training args')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    ## options: 'gaussian', 'gabor', 'pred'
    parser.add_argument('--filter', type=str, default='gaussian',
                        help='specifices the type of filter applied to create the adj matrix (default: gaussian)')
    parser.add_argument('--hypercolumns', action='store_true', default=False,
                        help='use hypercolumns as features')

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    num_epochs = args.epochs
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

    cnn = GraphNet(filter=args.filter, hypercolumns=args.hypercolumns)
    cnn.cuda()
    train_l, train_a, test_l, test_a = train_cnn(cnn, train_batches, test_batches, num_epochs)
    np.save("dump/gnnlast_{}_trainl.npy".format(args.filter), train_l)
    np.save("dump/gnnlast_{}_traina.npy".format(args.filter), train_a)
    np.save("dump/gnnlast_{}_testl.npy".format(args.filter), test_l)
    np.save("dump/gnnlast_{}_testa.npy".format(args.filter), test_a)
    get_plots(train_a, test_a, train_l, test_l, args.filter, epochs = num_epochs)

    torch.save(cnn, "gnn_{}.pt".format(args.filter))
