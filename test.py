import torch
from utils import *
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(test_loader, model, modalities):
    model.eval()
    f_te_list = []
    fy_te_list = []
    with torch.no_grad():
        for batch_index, (x_te, y_te) in enumerate(test_loader):
            f_te_list = []
            x_te.to(device)
            y_te.to(device)
            if model.multi_modal:
                x_te_list = divide(x_te, modalities)
            else:
                x_te_list = [x_te]

            _, _, _, _, _, _, _, _, _, cls, _ = model(x_te_list, valid=True)
            fy_te_list.append(torch.hstack((cls, y_te.to(device))))
        fy_te_list = torch.cat(fy_te_list, dim=0)
    return fy_te_list


def test2(test_loader, model, modalities):
    model.eval()
    f_te_list = []
    fy_te_list = []
    with torch.no_grad():
        for batch_index, (x_te, y_te) in enumerate(test_loader):
            f_te_list = []
            x_te.to(device)
            y_te.to(device)
            if model.multi_modal:
                x_te_list = divide(x_te, modalities)
            else:
                x_te_list = [x_te]

            f_list, res_feature, cls = model(x_te_list, valid=True)
            fy_te_list.append(torch.hstack((cls, y_te.to(device))))
        fy_te_list = torch.cat(fy_te_list, dim=0)
    return fy_te_list


def test_SVM(test_loader, model, modalities):
    model.eval()
    f_te_list = []
    fy_te_list = []
    loss_epoch_test = 0
    with torch.no_grad():
        for batch_index, (x_te, y_te) in enumerate(test_loader):
            f_te_list = []
            x_te.to(device)
            y_te.to(device)
            if model.multi_modal:
                x_te_list = divide(x_te, modalities)
            else:
                x_te_list = [x_te]

            _, _, _, _, _, _, _, res, _ = model(x_te_list, valid=True)
            fy_te_list.append(torch.hstack((res, y_te.to(device))))
        fy_te_list = torch.cat(fy_te_list, dim=0)
    return fy_te_list
