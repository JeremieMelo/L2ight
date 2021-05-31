#!/usr/bin/env python3
# coding=UTF-8

import torch
import os
import numpy as np


def loadDataset(path, dataset, n_label=8):
    data = []
    labels = []
    label_remap = [0, 5, 1, 6, 2, 7, 3, 4, 8, 9, 10]
    with open(f'{path}/{dataset}.data', 'r')as f:
        for line in f:

            line = line.strip().split()[3:]

            label = int(line[-1])
            label = label_remap[label]
            if(label >= n_label):
                continue


            labels.append(label),
            example = [float(i) for i in line[:-1]]
            data.append(example)

    data = torch.Tensor(data)
    labels = torch.LongTensor(labels)
    # data = torch.rfft(data, signal_ndim=1, normalized=True, onesided=True).norm(p=2, dim=-1)

    # perm = torch.randperm(labels.size(0))
    # data = data[perm]
    # labels = labels[perm]

    return data, labels


# def splitDataset(data, labels, train_per=0.7):
#     test_per = 1 - train_per
#     n_example = data.size(0)
#     n_train, n_test = int(n_example * train_per), int(n_example * test_per)

#     data_train, labels_train = data[:n_train, :], labels[:n_train]
#     data_test, labels_test = data[n_train:, :], labels[n_train:]
#     print(f'train: {n_train}, test: {n_test}')

#     return data_train, labels_train, data_test, labels_test


# def PCA(data_train, data_test):
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=4)
#     data_train_reduced = pca.fit_transform(data_train.numpy())
#     data_test_reduced = pca.transform(data_test.numpy())
#     print(data_test_reduced)
#     return data_train_reduced, data_test_reduced

def saveDataset(path, dataset, data_train, labels_train, data_test, labels_test):
    try:
        os.mkdir(path)
    except:
        pass
    with open(f'{path}/train.pt', 'wb') as f:
        torch.save((data_train, labels_train), f)

    with open(f'{path}/test.pt', 'wb') as f:
        torch.save((data_test, labels_test), f)
    print(f'Dataset {dataset} saved')


def splitDataset(data, labels, train_per=0.7):
    from sklearn.model_selection import train_test_split
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=train_per, random_state=42)
    # test_per = 1 - train_per
    # n_example = data.size(0)
    # n_train, n_test = int(n_example * train_per), int(n_example * test_per)

    # data_train, labels_train = data[:n_train, :], labels[:n_train]
    # data_test, labels_test = data[n_train:, :], labels[n_train:]
    print(f'train: {data_train.shape[0]}, test: {data_test.shape[0]}')

    return data_train, labels_train, data_test, labels_test

def select_topk_features(train_X, test_X, train_y, test_y, n_input=4):
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import RobustScaler
    rs = RobustScaler(quantile_range=(5, 95)).fit(np.concatenate([train_X.numpy(), test_X.numpy()], 0))
    train_X = torch.from_numpy(rs.transform(train_X.numpy()))
    test_X = torch.from_numpy(rs.transform(test_X.numpy()))

    clf = AdaBoostClassifier(
                base_estimator=ExtraTreesClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=5),
                n_estimators=10,
                random_state=42)
    clf = clf.fit(train_X.numpy(), train_y.numpy())
    predicted = clf.predict(train_X)
    acc = accuracy_score(train_y, predicted)
    print(f"[I] training accuracy: {acc}")
    imp = torch.tensor(permutation_importance(clf, test_X.numpy(), test_y.numpy(), n_repeats=10,random_state=42, n_jobs=4).importances_mean)
    values, indices = imp.sort(descending=True)
    print(imp)
    print(indices)
    selected_indices = indices[:n_input].sort()[0]
    train_X = train_X[:, selected_indices]
    test_X = test_X[:, selected_indices]
    # print(res, mask)


    return train_X, test_X, train_y, test_y

def PCA(data_train, data_test, n_input=4):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    pca = PCA(n_components=n_input)
    data_train_reduced = pca.fit_transform(data_train)
    data_test_reduced = pca.transform(data_test)

    # print(data_test_reduced)
    rs = RobustScaler(quantile_range=(10,90)).fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
    data_train_reduced = rs.transform(data_train_reduced)
    data_test_reduced = rs.transform(data_test_reduced)
    mms = MinMaxScaler()
    mms.fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
    data_train_reduced = mms.transform(data_train_reduced)
    data_test_reduced = mms.transform(data_test_reduced)
    # print(data_test_reduced)

    return torch.from_numpy(data_train_reduced).float(), torch.from_numpy(data_test_reduced).float()

def feature_extraction(data_train, data_test, labels_train, labels_test, n_feat, n_label):
    # data_train, data_test, labels_train, labels_test = torch.from_numpy(data_train), torch.from_numpy(data_test), torch.from_numpy(labels_train), torch.from_numpy(labels_test)
    torch.manual_seed(0)
    np.random.seed(0)
    if(torch.cuda.is_available()):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(data_train.size(1), 64)
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.dropout1 = torch.nn.Dropout(0.3)
            self.fc2 = torch.nn.Linear(64, n_feat)
            self.bn2 = torch.nn.BatchNorm1d(n_feat)
            self.dropout2 = torch.nn.Dropout(0.2)
            self.fc3 = torch.nn.Linear(n_feat, 32)
            self.fc4 = torch.nn.Linear(32, n_label)

        def forward(self, x):
            x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            y = self.fc4(self.dropout2(torch.relu(self.fc3(torch.relu(x)))))
            return x, y
    model = Model().cuda()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    bs = 16
    model.train()
    from tqdm import tqdm
    for epoch in tqdm(range(0, 400)):
        step = 0
        correct = 0
        while step < data_train.size(0):
            data = data_train[step:step+bs].cuda()
            target = labels_train[step:step+bs].cuda()
            step += bs
            optimizer.zero_grad()
            model.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss.backward()
            optimizer.step()
        accuracy = 100. * correct.to(torch.float32) / data_train.data.size(0)
    print(f"loss={loss.item()}, acc={accuracy}%")
    model.eval()
    data_train_reduced, _ = model(data_train.cuda())
    data_test_reduced, _ = model(data_test.cuda())
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    rs = RobustScaler(quantile_range=(5,95)).fit(np.concatenate([data_train_reduced.data.cpu().numpy(), data_test_reduced.data.cpu().numpy()], 0))
    print(data_train_reduced.size())
    data_train = rs.transform(data_train_reduced.data.cpu().numpy())
    data_test = rs.transform(data_test_reduced.data.cpu().numpy())

    return torch.from_numpy(data_train), torch.from_numpy(data_test), labels_train, labels_test



if __name__ == '__main__':
    dataset = 'vowel-context'
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_feat", type=int, default=4, help="Number of Features")
    argparser.add_argument("--n_label", type=int, default=4, help="Number of Labels")
    args = argparser.parse_args()
    raw_path = os.path.join(os.path.dirname(__file__), "raw")
    proc_path = os.path.join(os.path.dirname(__file__), "processed")
    data, labels = loadDataset(raw_path, dataset, n_label=args.n_label)
    data_train, labels_train, data_test, labels_test = splitDataset(data, labels)
    # data_train, data_test = PCA(data_train, data_test, n_input=4)
    # data_train, data_test, labels_train, labels_test = select_topk_features(data_train, data_test, labels_train, labels_test, n_input=5)
    # data_train, data_test = PCA(data_train, data_test, n_input=4)
    data_train, data_test, labels_train, labels_test = feature_extraction(data_train, data_test, labels_train, labels_test, n_feat=args.n_feat, n_label=args.n_label)
    saveDataset(proc_path, dataset, data_train, labels_train, data_test, labels_test)
