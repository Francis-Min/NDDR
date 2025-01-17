import torch
from args import args

device = args.device


def valid(data_loader, model):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_index, data in enumerate(data_loader):
            data = data.to(device)
            data.y = torch.tensor(data.y, dtype=torch.long).to(device)
            model = model.to(device)
            outputs = model(data)

            outputs_list.append(torch.hstack((outputs, data.y.to(device))))
        outputs_list = torch.cat(outputs_list, dim=0)

    return outputs_list


def valid_noa(data_loader, model):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            x = x.double().to(device)
            y = y.long().to(device)
            model = model.double().to(device)
            outputs, _, _ = model(x)

            outputs_list.append(torch.hstack((outputs, y)))
        outputs_list = torch.cat(outputs_list, dim=0)

    return outputs_list


def valid_noa2(data_loader, model):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            x = x.to(device).float()
            y = y.to(device)
            # x = torch.reshape(x, (-1, x.shape[-1] * x.shape[-2]))
            y = y.long().to(device)
            model.to(device)
            outputs = model(x)

            outputs_list.append(torch.hstack((outputs, y)))
        outputs_list = torch.cat(outputs_list, dim=0)

    return outputs_list


def valid_cross(data_loader, model):
    model.eval()
    outputs_list_emo = []
    outputs_list_sub = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            x = x.double().to(device)
            y = y[:, :1].long().to(device)
            d = y[:, 1:2].long().to(device)
            model = model.double().to(device)
            class_embedding, class_cls, domain_embedding, domain_cls = model.evaluate(x)

            outputs_list_emo.append(torch.hstack((class_cls, y)))
            outputs_list_sub.append(torch.hstack((domain_cls, d)))
        outputs_list_emo = torch.cat(outputs_list_emo, dim=0)
        outputs_list_sub = torch.cat(outputs_list_sub, dim=0)

    return outputs_list_emo, outputs_list_sub


def valid_cross_noa2(data_loader, model):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            x = x.float().to(device)
            y = y[:, :1].long().to(device)
            model = model.float().to(device)
            outputs = model(x)

            outputs_list.append(torch.hstack((outputs, y)))
        outputs_list = torch.cat(outputs_list, dim=0)

    return outputs_list


def valid3(data_loader, model):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            x = x.to(device).float()
            y = y.to(device)
            # y = y.squeeze()
            y = y.long()
            model.to(device)
            outputs = model(x)

            outputs_list.append(torch.hstack((outputs, y)))
        outputs_list = torch.cat(outputs_list, dim=0)

    return outputs_list