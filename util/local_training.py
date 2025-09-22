import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 
from util.util import generate

def slerp(lam, p1, p2):
    """Spherical interpolation between low and high vectors."""
    dot = torch.sum(p1 * p2, dim=1).clamp(-1, 1)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # Avoid division by zero for vectors that are nearly identical
    so = torch.where(so == 0, torch.ones_like(so), so)

    # Compute spherical interpolation
    part1 = (torch.sin(lam * omega) / so).unsqueeze(1) * p1
    part2 = (torch.sin((1.0 - lam) * omega) / so).unsqueeze(1) * p2
    res = part1 + part2

    # Handle vectors that are nearly identical
    close_mask = (omega.abs() < 1e-6).unsqueeze(1)  
    res = torch.where(close_mask, p1, res)  
    return res

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class DotRegressionLoss(nn.Module):
    def __init__(self):
        super(DotRegressionLoss, self).__init__()

    def forward(self, outputs, targets):
        # Normalize the outputs and targets
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        
        # Compute the dot product
        dot_product = torch.sum(outputs_norm * targets_norm, dim=1)
        
        loss = 0.5 * torch.mean((dot_product - 1) ** 2)
        return loss

def globaltest(net, test_dataset, args):
    """Accuracy using softmax predictions from classifier logits"""
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc

def globaltest_cos_similarity(net, test_dataset, args):
    """Accuracy using cosine similarity predictions"""
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        if args.model == 'resnet50':
            prototypes = generate(2048, args.num_classes, seed=args.seed)
        else:
            prototypes = generate(512, args.num_classes, seed=args.seed)
        prototypes = torch.tensor(prototypes).to(args.device)

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs, features = net(images, latent_output=True)
            
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
            
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(features.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
            predicted = torch.argmax(cos_sim, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()  
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, w_g, epoch, lr=None):
        net_glob = w_g
        net_glob.eval()

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        if self.args.model == 'resnet50':
            prototypes = generate(2048, self.args.num_classes, seed=self.args.seed)
        else:
            prototypes = generate(512, self.args.num_classes, seed=self.args.seed)
        prototypes = torch.tensor(prototypes).to(self.args.device)
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.mixup:
                    batch_size = images.size()[0] 
                    idx = torch.randperm(batch_size).cuda()
                    input_a, input_b = images, images[idx]
                    batch_prototypes = prototypes[labels]
                    batch_prototypes_a, batch_prototypes_b = batch_prototypes, batch_prototypes[idx]
                    lam = np.random.beta(self.args.alpha,self.args.alpha)
                    mixed_input = lam * input_a + (1 - lam) * input_b        
                    mixed_prototypes = slerp(lam, batch_prototypes_a, batch_prototypes_b)
                    net.zero_grad()
                    outputs_mixed, feature_mixed = net(mixed_input, latent_output=True)
                    loss_dotregression = DotRegressionLoss()

        
                    loss_alignement = loss_dotregression(feature_mixed, mixed_prototypes)
                    loss_mixed_crossentropy = lam * self.loss_func(outputs_mixed, labels) + (1 - lam) * self.loss_func(outputs_mixed, labels[idx])


                    p_mixed = torch.softmax(outputs_mixed, dim=1)  
                    h_c = p_mixed.mean(dim=0)  # Shape: (C,)
                    u_c = torch.full_like(h_c, 1 / self.args.num_classes)
                    loss_reg = (u_c * torch.log(u_c / h_c)).sum()
                    
                    loss = loss_mixed_crossentropy + loss_alignement + loss_reg

                else:
                    batch_size = images.size()[0] 
                    idx = torch.randperm(batch_size).cuda()
                    input_a, input_b = images, images[idx]
                    batch_prototypes = prototypes[labels]
                    net.zero_grad()
                    outputs, features = net(images, latent_output=True)
                    loss_dotregression = DotRegressionLoss()

        
                    loss_alignement = loss_dotregression(features, batch_prototypes)
                    loss_crossentropy = self.loss_func(outputs, labels) 

                    loss = loss_crossentropy + loss_alignement
                    

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)