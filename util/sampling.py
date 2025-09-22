import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)]) 
    return dict_users


def non_iid_dirichlet_sampling_returnphi(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])   
    return dict_users, Phi

def partition_test_data(y_test, y_train, dict_users_train, Phi, seed):
    np.random.seed(seed)
    num_users = Phi.shape[0]
    num_classes = Phi.shape[1]
    dict_users_test = {i: set() for i in range(num_users)}

    # Compute training class proportions per client
    train_class_counts = {i: {} for i in range(num_users)}
    for i in range(num_users):
        labels = y_train[list(dict_users_train[i])]
        for cls in np.unique(labels):
            train_class_counts[i][cls] = np.sum(labels == cls)
        total = sum(train_class_counts[i].values())
        for cls in train_class_counts[i]:
            train_class_counts[i][cls] /= total  # normalize

    # Distribute test samples
    for cls in range(num_classes):
        test_idxs = np.where(y_test == cls)[0]
        clients_with_cls = np.where(Phi[:, cls] == 1)[0]
        proportions = np.array([train_class_counts[i].get(cls, 0) for i in clients_with_cls])
        if proportions.sum() == 0:
            continue
        proportions /= proportions.sum()
        assignment = np.random.choice(clients_with_cls, size=len(test_idxs), p=proportions)
        for client in clients_with_cls:
            dict_users_test[client].update(test_idxs[assignment == client])

    return dict_users_test

 
