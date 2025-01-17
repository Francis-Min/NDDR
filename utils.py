import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from args import args


device = args.device


def normalize(data, select_dim=0):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # [0, 1]
    data_normalized = (data - min_vals) / (max_vals - min_vals)
    return data_normalized


def session_merge(data, label):
    data_15 = []
    sum_time = 3
    use_time = [0, 1, 2]
    for i in range(1, len(label)):
        if i < len(label):
            pre_inx = label[i-1]
            now_inx = label[i]
            data_temp = data[:, pre_inx:now_inx, :]
        else:
            pre_inx = label[i-1]
            data_temp = data[:, pre_inx:, :]
        [domain_size, sample_size, feature_size] = data_temp.shape
        res = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
        for i in range(domain_size // sum_time):
            inx = [i * sum_time + t for t in use_time]
            temp = np.array(data_temp[inx])
            temp = temp.reshape(-1, temp.shape[-1])
            res[i] = temp
        data_15.append(res)
    return data_15


def process_data(path, fold, batch, batch_size):
    all_data = np.load(path[0])
    all_label = np.load(path[1])
    domain_size, session_size, sample_size, d_num, feature_size = np.shape(all_data)
    fold_list = list(range(9, 15))
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for i in range(fold):
        start_session = 0
        end_session = 1
        data = all_data[i, start_session:end_session]
        label = all_label[i, start_session:end_session]

        # d = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], -1)
        d = data.reshape(data.shape[0]*data.shape[1], -1)
        min_max_scaler = preprocessing.MinMaxScaler()
        # d = min_max_scaler.fit_transform(d)  # 归一化
        data = d.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

        test_inx = fold_list
        train_inx = [index for index in all_inx if index not in test_inx]
        train_num = 2010
        f1 = 0
        f2 = 5
        train_data = data[:, :train_num, :, f1:f2]
        train_label = label[:, :train_num]
        test_data = data[:, train_num:, :, f1:f2]
        test_label = label[:, train_num:]

        X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
        Y_train = train_label.reshape(-1, train_label.shape[-1])
        X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
        Y_test = test_label.reshape(-1, test_label.shape[-1])

        # random n
        if batch:
            n = batch_size*4
        else:
            n = X_train.shape[0]//(fold-1)
        # n = 128
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)

        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]

        # random_indices = np.random.choice(X_test.shape[0], n, replace=False)
        # X_test = X_test[random_indices]
        # Y_test = Y_test[random_indices]
        #
        # random_indices = np.random.choice(X_train.shape[0], n, replace=False)
        # X_train = X_train[random_indices]
        # Y_train = Y_train[random_indices]

        # XY = np.hstack((X_train, Y_train))
        # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def my_process_data(path, fold, batch, batch_size):
    all_data = np.load(path[0])
    all_adj = np.load(path[1])
    all_label = np.load(path[2])
    domain_size, session_size, sample_size, d_num, feature_size = np.shape(all_data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]

    x_train_list = []
    a_train_list = []
    y_train_list = []
    x_val_list = []
    a_val_list = []
    y_val_list = []
    x_test_list = []
    a_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for i in range(fold):
        session = 0
        if session < 2:
            data = all_data[i, session:session + 1]
            adj = all_adj[i, session:session + 1]
            label = all_label[i, session:session + 1]
        else:
            data = all_data[i, session:]
            adj = all_adj[i, session:]
            label = all_label[i, session:]

        # d = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], -1)
        d = data.reshape(data.shape[0]*data.shape[1], -1)
        min_max_scaler = preprocessing.MinMaxScaler()
        # d = min_max_scaler.fit_transform(d)  # 归一化
        data = d.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

        train_num = 2010
        f1 = 0
        f2 = 5
        train_data = data[:, :train_num, :, f1:f2]
        train_adj = adj[:, :train_num]
        train_label = label[:, :train_num]
        test_data = data[:, train_num:, :, f1:f2]
        test_adj = adj[:, train_num:]
        test_label = label[:, train_num:]

        X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
        A_train = train_adj.reshape(-1, train_adj.shape[-2], train_adj.shape[-1])
        Y_train = train_label.reshape(-1, train_label.shape[-1])
        X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
        A_test = test_adj.reshape(-1, test_adj.shape[-2], test_adj.shape[-1])
        Y_test = test_label.reshape(-1, test_label.shape[-1])

        if batch:
            n = batch_size*4
        else:
            n = X_train.shape[0]//(fold-1)
        # n = 128
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引
        X_val = X_train[random_indices]
        A_val = A_train[random_indices]
        Y_val = Y_train[random_indices]

        # XY = np.hstack((X_train, Y_train))
        # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        a_train_list.append(A_train)
        a_val_list.append(A_val)
        a_test_list.append(A_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, a_train_list, y_train_list, x_val_list, a_val_list, y_val_list, x_test_list, a_test_list, y_test_list


def my_process_data_noa(path, fold, batch, batch_size):
    all_data = np.load(path[0])
    all_label = np.load(path[1])

    domain_size, session_size, sample_size, d_num, feature_size = np.shape(all_data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for i in range(fold):
        session = 0
        if session < 2:
            data = all_data[i, session:session + 1]
            label = all_label[i, session:session + 1]
        else:
            data = all_data[i, session:]
            label = all_label[i, session:]

        # d = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], -1)
        d = data.reshape(data.shape[0]*data.shape[1], -1)
        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(d)  # 归一化
        data = d.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

        train_num = 2010
        f1 = 0
        f2 = 5
        train_data = data[:, :train_num, :, f1:f2]
        train_label = label[:, :train_num]
        test_data = data[:, train_num:, :, f1:f2]
        test_label = label[:, train_num:]

        X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
        Y_train = train_label.reshape(-1, train_label.shape[-1])
        X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
        Y_test = test_label.reshape(-1, test_label.shape[-1])

        # XY = np.hstack((X_train, Y_train))
        # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_cross_data(path, fold, batch, batch_size):
    all_data = np.load(path[0])
    all_label = np.load(path[1])
    domain_label = np.ones(all_label.shape)
    for i in range(domain_label.shape[0]):
        domain_label[i:(i+1)] = i
    all_label = np.concatenate((all_label, domain_label), axis=3)

    domain_size, session_size, sample_size, d_num, feature_size = np.shape(all_data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for session in range(3):
        for i in range(fold):
            if session < 2:
                data = all_data[:, session:session + 1]
                label = all_label[:, session:session + 1]
            else:
                data = all_data[:, session:]
                label = all_label[:, session:]

            # d = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], -1)
            # d = data.reshape(data.shape[0]*data.shape[1], -1)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # d = min_max_scaler.fit_transform(d)  # 归一化
            # data = d.reshape(data.shape)

            for sub in range(data.shape[0]):
                d = data[sub]
                temp = d.reshape(d.shape[0]*d.shape[1], -1)
                temp = normalize(temp)
                d = temp.reshape(d.shape)
                data[sub] = d

            test_index = i
            train_index = [index for index in all_inx if index != test_index]
            f1 = 0
            f2 = 5
            train_data = data[train_index, :, :, :, f1:f2]
            train_label = label[train_index, :]
            test_data = data[test_index, :, :, :, f1:f2]
            test_label = label[test_index, :]

            X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
            Y_train = train_label.reshape(-1, train_label.shape[-1])
            X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
            Y_test = test_label.reshape(-1, test_label.shape[-1])

            # XY = np.hstack((X_train, Y_train))
            # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
            # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
            # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(Y_train)
            y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_cross_data_IV(path, fold, batch, batch_size):
    data_path = ['SEED_IV-s1.npy', 'SEED_IV-s2.npy', 'SEED_IV-s3.npy']
    d1_num = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27,
              54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]
    d2_num = [55, 25, 34, 36, 53, 27, 34, 46, 34, 20, 60, 12,
              36, 27, 44, 15, 46, 49, 45, 10, 37, 44, 24, 19]
    d3_num = [42, 32, 23, 45, 48, 26, 64, 23, 26, 16, 51, 41,
              39, 19, 28, 44, 14, 17, 45, 22, 39, 38, 41, 39]
    label_path = ['SEED_IV-s1_label.npy', 'SEED_IV-s2_label.npy', 'SEED_IV-s3_label.npy']

    d1 = np.load(path + data_path[0])
    d2 = np.load(path + data_path[1])
    d3 = np.load(path + data_path[2])
    l1 = np.load(path + label_path[0])
    l2 = np.load(path + label_path[1])
    l3 = np.load(path + label_path[2])

    domain_label = np.ones(l1.shape)
    for i in range(domain_label.shape[0]):
        domain_label[i:(i+1)] = i
    l1 = np.concatenate((l1, domain_label), axis=2)

    domain_label = np.ones(l2.shape)
    for i in range(domain_label.shape[0]):
        domain_label[i:(i+1)] = i
    l2 = np.concatenate((l2, domain_label), axis=2)

    domain_label = np.ones(l3.shape)
    for i in range(domain_label.shape[0]):
        domain_label[i:(i+1)] = i
    l3 = np.concatenate((l3, domain_label), axis=2)

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for session in range(3):
        for i in range(fold):
            if session == 0:
                data = d1
                label = l1
            elif session == 1:
                data = d2
                label = l2
            elif session == 2:
                data = d3
                label = l3

            # d = data.reshape(data.shape[0]*data.shape[1], -1)
            # # d = data.reshape(data.shape[0], -1)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # d = min_max_scaler.fit_transform(d)  # 归一化
            # data = d.reshape(data.shape)
            for sub in range(data.shape[0]):
                d = data[sub]
                temp = d.reshape(d.shape[0], -1)
                temp = normalize(temp)
                d = temp.reshape(d.shape)
                data[sub] = d

            test_index = i
            train_index = [index for index in all_inx if index != test_index]
            f1 = 0
            f2 = 5
            train_data = data[train_index, :, :, f1:f2]
            train_label = label[train_index, :]
            test_data = data[test_index, :, :, f1:f2]
            test_label = label[test_index, :]

            X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
            Y_train = train_label.reshape(-1, train_label.shape[-1])
            X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
            Y_test = test_label.reshape(-1, test_label.shape[-1])

            # XY = np.hstack((X_train, Y_train))
            # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
            # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
            # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(Y_train)
            y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_cross_data_IV_noa2(path, fold, batch, batch_size):
    data_path = ['SEED_IV-s1.npy', 'SEED_IV-s2.npy', 'SEED_IV-s3.npy']
    d1_num = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27,
              54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]
    d2_num = [55, 25, 34, 36, 53, 27, 34, 46, 34, 20, 60, 12,
              36, 27, 44, 15, 46, 49, 45, 10, 37, 44, 24, 19]
    d3_num = [42, 32, 23, 45, 48, 26, 64, 23, 26, 16, 51, 41,
              39, 19, 28, 44, 14, 17, 45, 22, 39, 38, 41, 39]
    label_path = ['SEED_IV-s1_label.npy', 'SEED_IV-s2_label.npy', 'SEED_IV-s3_label.npy']

    d1 = np.load(path + data_path[0])
    d2 = np.load(path + data_path[1])
    d3 = np.load(path + data_path[2])
    l1 = np.load(path + label_path[0])
    l2 = np.load(path + label_path[1])
    l3 = np.load(path + label_path[2])

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for session in range(3):
        for i in range(fold):
            if session == 0:
                data = d1
                label = l1
            elif session == 1:
                data = d2
                label = l2
            elif session == 2:
                data = d3
                label = l3

            test_index = i
            train_index = [index for index in all_inx if index != test_index]
            f1 = 0
            f2 = 5
            train_data = data[train_index, :, :, f1:f2]
            train_label = label[train_index, :]
            test_data = data[test_index, :, :, f1:f2]
            test_label = label[test_index, :]

            X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
            Y_train = train_label.reshape(-1, train_label.shape[-1])
            X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
            Y_test = test_label.reshape(-1, test_label.shape[-1])

            # XY = np.hstack((X_train, Y_train))
            # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
            # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
            # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(Y_train)
            y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_data_noa2(path, fold, batch, batch_size, fre=0):
    all_data = np.load(path[0])
    all_label = np.load(path[1])
    domain_size, session_size, sample_size, d_num, feature_size = np.shape(all_data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(45))
    for session in range(3):
        for i in range(fold):
            if session < 2:
                data = all_data[i, session:session + 1]
                label = all_label[i, session:session + 1]
            else:
                data = all_data[i, session:]
                label = all_label[i, session:]

            # d = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], -1)
            d = data.reshape(data.shape[0]*data.shape[1], -1)
            min_max_scaler = preprocessing.MinMaxScaler()
            # d = min_max_scaler.fit_transform(d)  # 归一化
            data = d.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

            train_num = 2010
            f1 = 0
            f2 = 5
            if fre != 0:
                f1 = fre - 1
                f2 = fre
            train_data = data[:, :train_num, :, f1:f2]
            train_label = label[:, :train_num]
            test_data = data[:, train_num:, :, f1:f2]
            test_label = label[:, train_num:]

            X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
            Y_train = train_label.reshape(-1, train_label.shape[-1])
            X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
            Y_test = test_label.reshape(-1, test_label.shape[-1])

            # XY = np.hstack((X_train, Y_train))
            # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
            # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
            # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(Y_train)
            y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_data_IV_noa2(path, fold, batch, batch_size, fre=0):
    data_path = ['SEED_IV-s1.npy', 'SEED_IV-s2.npy', 'SEED_IV-s3.npy']
    d1_num = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27,
              54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]
    d2_num = [55, 25, 34, 36, 53, 27, 34, 46, 34, 20, 60, 12,
              36, 27, 44, 15, 46, 49, 45, 10, 37, 44, 24, 19]
    d3_num = [42, 32, 23, 45, 48, 26, 64, 23, 26, 16, 51, 41,
              39, 19, 28, 44, 14, 17, 45, 22, 39, 38, 41, 39]
    label_path = ['SEED_IV-s1_label.npy', 'SEED_IV-s2_label.npy', 'SEED_IV-s3_label.npy']

    d1 = np.load(path + data_path[0])
    d2 = np.load(path + data_path[1])
    d3 = np.load(path + data_path[2])
    l1 = np.load(path + label_path[0])
    l2 = np.load(path + label_path[1])
    l3 = np.load(path + label_path[2])

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for session in range(3):
        for i in range(fold):
            if session == 0:
                data = d1[i]
                label = l1[i]
                train_num = sum(d1_num[:16])
            elif session == 1:
                data = d2[i]
                label = l2[i]
                train_num = sum(d2_num[:16])
            elif session == 2:
                data = d3[i]
                label = l3[i]
                train_num = sum(d3_num[:16])

            f1 = 0
            f2 = 5
            if fre != 0:
                f1 = fre - 1
                f2 = fre
            train_data = data[:train_num, :, f1:f2]
            train_label = label[:train_num]
            if session == 0:
                test_data = [data[train_num:, :, f1:f2], data[train_num-150:train_num-145, :, f1:f2]]
                test_data = np.concatenate(test_data, axis=0)
                test_label = [label[train_num:], label[train_num - 150:train_num - 145]]
                test_label = np.concatenate(test_label, axis=0)
            else:
                test_data = data[train_num:, :, f1:f2]
                test_label = label[train_num:]

            X_train = train_data.reshape(-1, train_data.shape[-2], train_data.shape[-1])
            Y_train = train_label.reshape(-1, train_label.shape[-1])
            X_test = test_data.reshape(-1, test_data.shape[-2], test_data.shape[-1])
            Y_test = test_label.reshape(-1, test_label.shape[-1])

            # XY = np.hstack((X_train, Y_train))
            # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
            # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
            # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(Y_train)
            y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def my_process_data_IV(path, fold, batch, batch_size):
    data_path = ['SEED_IV-s1.npy', 'SEED_IV-s2.npy', 'SEED_IV-s3.npy']
    d1_num = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27,
              54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]
    d2_num = [55, 25, 34, 36, 53, 27, 34, 46, 34, 20, 60, 12,
              36, 27, 44, 15, 46, 49, 45, 10, 37, 44, 24, 19]
    d3_num = [42, 32, 23, 45, 48, 26, 64, 23, 26, 16, 51, 41,
              39, 19, 28, 44, 14, 17, 45, 22, 39, 38, 41, 39]
    adj_path = ['SEED_IV-s1_adj.npy', 'SEED_IV-s2_adj.npy', 'SEED_IV-s3_adj.npy']
    label_path = ['SEED_IV-s1_label.npy', 'SEED_IV-s2_label.npy', 'SEED_IV-s3_label.npy']

    d1 = np.load(path + data_path[0])
    d2 = np.load(path + data_path[1])
    d3 = np.load(path + data_path[2])
    a1 = np.load(path + adj_path[0])
    a2 = np.load(path + adj_path[1])
    a3 = np.load(path + adj_path[2])
    l1 = np.load(path + label_path[0])
    l2 = np.load(path + label_path[1])
    l3 = np.load(path + label_path[2])

    x_train_list = []
    a_train_list = []
    y_train_list = []
    x_val_list = []
    a_val_list = []
    y_val_list = []
    x_test_list = []
    a_test_list = []
    y_test_list = []

    all_inx = list(range(15))
    for sub in range(fold):
        data = [d1[sub], d2[sub], d3[sub]]
        adj = [a1[sub], a2[sub], a3[sub]]
        label = [l1[sub], l2[sub], l3[sub]]

        d = np.concatenate(data, axis=0)
        reshape_d = d.reshape(d.shape[0], -1)
        min_max_scaler = preprocessing.MinMaxScaler()
        reshape_d = min_max_scaler.fit_transform(reshape_d)  # 归一化
        d = reshape_d.reshape(d.shape)
        data = [d[:d1.shape[1]], d[d1.shape[1]:d1.shape[1]+d2.shape[1]], d[d1.shape[1]+d2.shape[1]:d1.shape[1]+d2.shape[1]+d3.shape[1]]]

        train_num = [sum(d1_num[:16]), sum(d2_num[:16]), sum(d3_num[:16])]
        f1 = 0
        f2 = 5
        train_data = [data[s][:train_num[s], :, f1:f2] for s in range(len(data))]
        X_train = np.concatenate(train_data, axis=0)
        test_data = [data[s][train_num[s]:, :, f1:f2] for s in range(len(data))]
        X_test = np.concatenate(test_data, axis=0)

        train_adj = [adj[s][:train_num[s]] for s in range(len(adj))]
        A_train = np.concatenate(train_adj, axis=0)
        test_adj = [adj[s][train_num[s]:] for s in range(len(adj))]
        A_test = np.concatenate(test_adj, axis=0)

        train_label = [label[s][:train_num[s]] for s in range(len(label))]
        Y_train = np.concatenate(train_label, axis=0)
        test_label = [label[s][train_num[s]:] for s in range(len(label))]
        Y_test = np.concatenate(test_label, axis=0)

        if batch:
            n = batch_size*4  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        # n = 128
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)
        X_val = X_train[random_indices]
        A_val = A_train[random_indices]
        Y_val = Y_train[random_indices]
        # print(X_train.shape)
        # print(X_val.shape)
        # print(X_test.shape)
        # print(A_train.shape)
        # print(A_val.shape)
        # print(A_test.shape)
        # print(Y_train.shape)
        # print(Y_val.shape)
        # print(Y_test.shape)

        # XY = np.hstack((X_train, Y_train))
        # sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        # X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        # Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        a_train_list.append(A_train)
        a_val_list.append(A_val)
        a_test_list.append(A_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, a_train_list, y_train_list, x_val_list, a_val_list, y_val_list, x_test_list, a_test_list, y_test_list


def getfold(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    xy_tr = list(zip(X_train, Y_train))
    xy_val = list(zip(X_val, Y_val))
    xy_te = list(zip(X_test, Y_test))
    return xy_tr, xy_val, xy_te


def load_data(x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list, i, batch, batch_size=1024):
    xy_tr, xy_val, xy_te = getfold(x_train_list[i], y_train_list[i],
                                   x_val_list[i], y_val_list[i],
                                   x_test_list[i], y_test_list[i])
    sample_num = batch_size
    train_loader = DataLoader(
        xy_tr, batch_size=sample_num, shuffle=True)

    val_loader = DataLoader(
        xy_val, batch_size=sample_num, shuffle=True)

    test_loader = DataLoader(
        xy_te, batch_size=sample_num, shuffle=True)

    return train_loader, val_loader, test_loader, sample_num


def load_subgraph_data_noa(x_train_list, y_train_list,
                           x_test_list, y_test_list,
                           i, batch, train_batch_size, test_batch_size):
    x_tr = x_train_list[i]
    y_tr = y_train_list[i]
    x_te = x_test_list[i]
    y_te = y_test_list[i]

    xy_tr = list(zip(x_tr, y_tr))
    xy_te = list(zip(x_te, y_te))

    # Create a DataLoader to handle batching
    train_loader = DataLoader(xy_tr, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(xy_te, batch_size=test_batch_size, shuffle=True, drop_last=True)

    # print("Finish take data loader...")
    return train_loader, test_loader, [train_batch_size, test_batch_size]


if __name__ == '__main__':
    fold = 15
    batch_size = 32
    batch = True
    data_path = './data/SEED_IV/'
    path = data_path
    x_train_fold, y_train_fold, x_test_fold, y_test_fold = my_process_cross_data_IV(path, fold, batch=batch, batch_size=batch_size)



