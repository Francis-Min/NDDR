import statistics
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from valid import *
from test import *
from models.loss_function import *
from utils import *
from eval import *
from models.Model import NDDR
from args import args


res = []
device = args.device


def train(model, train_loader, test_loader, epochs, lr, decay):
    printOK = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()
    rec_criterion = nn.MSELoss()
    criterion_mmd = MMDLoss()
    max_acc_val = 0
    max_acc_te = 0

    for epoch in range(1, epochs + 1):
        y_list = []
        fy_tr_list = []
        loss_epoch_train = 0
        train_loss = [0, 0, 0, 0, 0, 0, 0, 0]
        test_loss = [0, 0, 0, 0, 0, 0, 0, 0]
        for batch_index, (x, y) in enumerate(train_loader):
            source_data = x.double().to(device)
            source_label = y[:, :1].long().to(device)
            source_domain = y[:, 1:2].long().to(device)

            test_iter = iter(test_loader)
            target_x, target_y = next(test_iter)
            target_data = target_x.double().to(device)
            target_label = target_y[:, :1].long().to(device)
            target_domain = target_y[:, 1:2].long().to(device)

            model = model.double().to(device)
            model.train()

            ([class_embedding1, mutual_embedding1, domain_embedding1, rec_embedding1], [class_cls1, domain_cls1],
             [loss_pre1, loss_IIF1, loss_MIF1, loss_rec1, loss_class1]) = model(source_data, source_label)
            ([class_embedding2, mutual_embedding2, domain_embedding2, rec_embedding2], [class_cls2, domain_cls2],
             [loss_pre2, loss_IIF2, loss_MIF2, loss_rec2, loss_class2]) = model(target_data, target_label)

            loss_cls1 = criterion(class_cls1, source_label.squeeze())
            loss_cls2 = criterion(class_cls2, target_label.squeeze())
            loss_cls = loss_cls1
            loss_domain1 = criterion(domain_cls1, source_domain.squeeze())
            loss_domain2 = criterion(domain_cls2, target_domain.squeeze())
            loss_domain = loss_domain1 + loss_domain2

            loss_IIF = loss_IIF1
            loss_MIF1 = criterion_mmd(mutual_embedding1, mutual_embedding2) + loss_MIF1
            loss_MIF2 = criterion_mmd(mutual_embedding1, mutual_embedding2) + loss_MIF2

            loss_MIF = loss_MIF1
            loss_pre = (loss_pre1 + loss_pre2) * 0.5
            loss_rec = (loss_rec1 + loss_rec2) * 0.5
            loss_mmd = criterion_mmd(class_embedding1, class_embedding2)
            loss_class = loss_class1
            loss = 2 * loss_cls + loss_domain + loss_IIF + loss_MIF + loss_pre + loss_rec + loss_mmd + loss_class
            # loss = 2 * loss_cls + 0.5 * loss_domain + loss_IIF + loss_MIF + loss_pre + loss_rec + loss_mmd + loss_class

            tr = [loss_cls1.item(), loss_domain1.item(), loss_IIF1.item(), loss_MIF1.item(), loss_pre1.item(), loss_rec1.item(), loss_mmd.item(), loss_class1.item()]
            te = [loss_cls2.item(), loss_domain2.item(), loss_IIF2.item(), loss_MIF2.item(), loss_pre2.item(), loss_rec2.item(), loss_mmd.item(), loss_class2.item()]
            for n in range(len(train_loss)):
                train_loss[n] += tr[n]
                test_loss[n] += te[n]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()
            y_list.append(y)

        if test_loader != None:
            y_tr_list_emo, y_tr_list_sub = valid_cross(train_loader, model)
            acc_tr, sen_tr, spe_tr, auc_tr, pre_tr, f1_tr, std_tr, _, _ = myEval(y_tr_list_emo)
            acc_tr2 = myEval(y_tr_list_sub, type='sub')
            y_te_list_emo, y_te_list_sub = valid_cross(test_loader, model)
            acc_te, sen, spe, auc, pre, f1, std, precision, recall = myEval(y_te_list_emo)
            acc_te2 = myEval(y_te_list_sub, type='sub')
            if epoch % 10 == 0:
                # 创建损失名称列表
                loss_names = ["loss_cls", "loss_domain", "loss_IIF", "loss_MIF", "loss_pre", "loss_rec", "loss_mmd", "loss_class"]

                # 格式化输出
                train_losses_str = ", ".join(
                    f"{name}={loss:.4f}" for name, loss in zip(loss_names, train_loss))
                train_losses_str += f", total_loss={sum(train_loss):.4f}"
                test_losses_str = ", ".join(
                    f"{name}={loss:.4f}" for name, loss in zip(loss_names, test_loss))
                test_losses_str += f", total_loss={sum(test_loss):.4f}"

                print(f"epoch {epoch}: Emotion", "acc_tr: {:.4f}, acc_test: {:.4f}".format(acc_tr, acc_te))
                print(f"epoch {epoch}: Subject", "acc_tr: {:.4f}, acc_test: {:.4f}".format(acc_tr2, acc_te2))
                print(f"     train---{train_losses_str}")
                print(f"     test----{test_losses_str}")

            if acc_tr >= max_acc_val:
                max_acc_val = acc_tr
            if acc_te >= max_acc_te:
                max_fy_tr = fy_tr_list
                max_acc_te = acc_te
                torch.save(model.state_dict(), checkpoint)

    return max_fy_tr, max_acc_val


def myEval(y_list, type='emo', save=False):
    if save:
        res.append(y_list.cpu().detach().numpy())
    y_list = y_list.cpu().clone().detach()

    if type == 'emo':
        metrics, precision, recall = eval(y_list[:, :-1], y_list[:, -1], average='macro')
        acc = metrics['Accuracy']
        auc = metrics['ROC AUC']
        sen = metrics['Sensitivity']
        spe = metrics['Specificity']
        pre = metrics['Precision']
        f1 = metrics['F1 Score']
        std = metrics['STD']

        return acc, auc, sen, spe, pre, f1, std, precision, recall
    if type == 'sub':
        acc = eval_acc(y_list[:, :-1], y_list[:, -1], average='macro')
        return acc


if __name__ == '__main__':
    fold = 15
    batch_size = 64
    batch = True
    epochs = 200
    data_path = 'data/SEED/SEED_1s.npy'
    label_path = 'data/SEED/SEED_1s_label.npy'
    path = [data_path, label_path]
    checkpoint = 'checkpoint-mycross-1.model'

    modalities = [0, 310, 310 + 33]

    weight_coef = 1
    weight_selfExp = 0.2
    weight_block = 1
    show_freq = 1
    lr = 1e-3
    svmc = 0.5
    mom_list = 0.9
    decay_list = 5e-5

    acc_tr_list = list()
    acc_te_list = list()
    auc_list = list()
    spe_list = list()
    sen_list = list()
    pre_list = list()
    F1_list = list()
    pre_list1 = list()
    pre_list2 = list()
    pre_list3 = list()
    recall_list1 = list()
    recall_list2 = list()
    recall_list3 = list()

    fold_acc_tr_list = np.empty((0, epochs))
    fold_acc_te_list = np.empty((0, epochs))
    fold_auc_list = np.empty((0, epochs))
    fold_spe_list = np.empty((0, epochs))
    fold_sen_list = np.empty((0, epochs))
    fold_pre_list = np.empty((0, epochs))
    fold_F1_list = np.empty((0, epochs))
    fold_pre_list1 = np.empty((0, epochs))
    fold_pre_list2 = np.empty((0, epochs))
    fold_pre_list3 = np.empty((0, epochs))
    fold_recall_list1 = np.empty((0, epochs))
    fold_recall_list2 = np.empty((0, epochs))
    fold_recall_list3 = np.empty((0, epochs))

    x_train_fold, y_train_fold, x_test_fold, y_test_fold = my_process_cross_data(path, fold, batch=batch,
                                                                                 batch_size=batch_size)

    for i in range(11, 12):
        seed = 7
        print('seed is {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        train_loader, test_loader, sample_num = load_subgraph_data_noa(x_train_fold, y_train_fold,
                                                                       x_test_fold, y_test_fold, i, batch,
                                                                       train_batch_size=args.train_batch_size,
                                                                       test_batch_size=args.test_batch_size)

        print("Finish data load.")
        model = NDDR(args).to(device)
        print("Finish model load.")
        y_tr_list, acc_val = train(model, train_loader, test_loader, epochs, lr=lr, decay=decay_list)

        checkpoint_ = torch.load(checkpoint)
        model.load_state_dict(checkpoint_)
        if args.save:
            sub = i % fold + 1
            torch.save(model.state_dict(), 'save/rec/SEED/trained_model_sub' + str(i) + '.model')
            model = NDDR(args).to(device)
            print("Finish model load.")

            try:
                checkpoint_ = torch.load(checkpoint, map_location=torch.device('cpu'))  # 根据需要调整 map_location
            except Exception as e:
                print(f"Wrong: {e}")

            checkpoint_ = torch.load('save/rec/SEED/trained_model_sub' + str(i) + '.model')
            model.load_state_dict(checkpoint_)

        y_tr_list_emo, y_tr_list_sub = valid_cross(train_loader, model)
        acc_tr, sen_tr, spe_tr, auc_tr, pre_tr, f1_tr, std_tr, _, _ = myEval(y_tr_list_emo)
        acc_tr2 = myEval(y_tr_list_sub, type='sub')
        y_te_list_emo, y_te_list_sub = valid_cross(test_loader, model)
        acc_te, sen, spe, auc, pre, f1, std, precision, recall = myEval(y_te_list_emo)
        acc_te2 = myEval(y_te_list_sub, type='sub')

        print('fold = ', i, 'acc_tr = ', acc_tr, 'acc_te =', acc_te, 'acc_tr_sub = ', acc_tr2, 'acc_te_sub =', acc_te2,
              'std = ', std, 'sen = ', sen, 'spe = ', spe, 'auc = ', auc, 'pre = ', pre, 'F1:', f1)
        print('sad:     precision = ', precision[0], 'recall = ', recall[0])
        print('neutral: precision = ', precision[1], 'recall = ', recall[1])
        print('happy:   precision = ', precision[2], 'recall = ', recall[2])

        acc_tr_list.append(acc_tr)
        acc_te_list.append(acc_te)
        auc_list.append(auc)
        spe_list.append(spe)
        sen_list.append(sen)
        pre_list.append(pre)
        F1_list.append(f1)
        pre_list1.append(precision[0])
        pre_list2.append(precision[1])
        pre_list3.append(precision[2])
        recall_list1.append(recall[0])
        recall_list2.append(recall[1])
        recall_list3.append(recall[2])

    print('acc_tr:', sum(acc_tr_list) / fold)
    print('acc_te:', sum(acc_te_list) / fold)
    print('auc:', sum(auc_list) / fold)
    print('spe:', sum(spe_list) / fold)
    print('sen:', sum(sen_list) / fold)
    print('pre:', sum(pre_list) / fold)
    print('F1:', sum(F1_list) / fold)
    print('sad:     precision = ', sum(pre_list1) / fold, 'recall = ', sum(recall_list1) / fold)
    print('neutral: precision = ', sum(pre_list2) / fold, 'recall = ', sum(recall_list2) / fold)
    print('happy:   precision = ', sum(pre_list3) / fold, 'recall = ', sum(recall_list3) / fold)

    print('acc_tr:', sum(acc_tr_list) / fold, '+-', statistics.stdev(acc_tr_list))
    print('acc_te:', sum(acc_te_list) / fold, '+-', statistics.stdev(acc_te_list))
    print('auc:', sum(auc_list) / fold, '+-', statistics.stdev(auc_list))
    # print('std:', sum(std_list) / fold, '+-', statistics.stdev(std_list))
    print('spe:', sum(spe_list) / fold, '+-', statistics.stdev(spe_list))
    print('sen:', sum(sen_list) / fold, '+-', statistics.stdev(sen_list))
    print('pre:', sum(pre_list) / fold, '+-', statistics.stdev(pre_list))
    print('F1:', sum(F1_list) / fold, '+-', statistics.stdev(F1_list))
    print('-' * 100)
