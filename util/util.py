import numpy as np
import torch
import torch.nn.functional as F
import copy
import random
import math
from typing import Optional

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * (1 - 1/args.num_classes), noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent_output=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent_output == False:
                outputs, features = net(images, latent_output=True)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs, features = net(images, True)
                outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
                if latent_output:
                    features_whole = np.array(features.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
                if latent_output:
                    features_whole = np.concatenate((features_whole, features.cpu()), axis=0)
    if criterion is not None:
        if latent_output == False:
            return output_whole, loss_whole
        else:
            return output_whole, loss_whole, features_whole
    else:
        return output_whole
    
def pedcc_generation(
    n: int, k: int = None, seed: Optional[int] = None
) -> np.ndarray:
    def pedcc_frame(n: int, k: int = None) -> np.ndarray:
        assert 0 < k <= n + 1
        zero = [0] * (n - k + 1)
        u0 = [-1][:0] + zero + [-1][0:]
        u1 = [1][:0] + zero + [1][0:]
        u = np.stack((u0, u1)).tolist()
        for i in range(k - 2):
            c = np.insert(u[len(u) - 1], 0, 0)
            for j in range(len(u)):
                p = np.append(u[j], 0).tolist()
                s = len(u) + 1
                u[j] = math.sqrt(s * (s - 2)) / (s - 1) * np.array(p) - 1 / (
                    s - 1
                ) * np.array(c)
            u.append(c)
        return np.array(u)

    U = pedcc_frame(n=n, k=k)
    r = np.random.RandomState(seed)
    while True:
        try:
            noise = r.rand(n, n) # [0, 1)   
            V, _ = np.linalg.qr(noise)
            break
        except np.linalg.LinAlgError:
            continue

    points = np.dot(U, V)
    return points

def generate(
    n: int,
    k: int,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate k evenly distributed R^n points in a unit (n-1)-hypersphere 
    Args:
        n (int): dimension of the Euclidean space
        k (int): number of points to generate
        method (str): method to generate the points. Defaults to "simplex".
        filename (str, optional): filename to save the points. Defaults to None.
        seed (int, optional): seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: k evenly distributed points in a unit (n-1)-hypersphere

    >>> generate(2, 3, method="simplex")
    array([[ 0.        ,  0.        ],
           [ 0.70710678,  0.70710678],
    """
    if seed is None or not (isinstance(seed, int) and 0 <= seed < 2 ** 32):
        print("[pedcc.generate] seed must be an integer between 0 and 2**32-1")
        seed = random.randrange(2 ** 32)
        print("[pedcc.generate] seed set to", seed)

    points = pedcc_generation(n, k, seed=seed)

    if filename:
        path = f"{filename}_n{n}_k{k}_seed{seed}.npy"
        np.save(path, points)

    return points


def extract_features(args, model, dataset):
        test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for images, label_batch in test_loader:
                images = images.to(args.device)
                output, feature = model(images, latent_output=True)  # Extract features
                feature = F.normalize(feature, p=2, dim=1) # Normalize the extracted features
                features.append(feature.cpu())
                labels.append(label_batch.cpu())

        features = torch.cat(features, dim=0).numpy()  # Convert to numpy array
        labels = torch.cat(labels, dim=0).numpy()
        return features, labels