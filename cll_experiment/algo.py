import torch
import torch.nn.functional as F
import numpy as np

def ga_loss(outputs, labels, class_prior, T, num_classes):
    device = labels.device
    if torch.det(T) != 0:
        Tinv = torch.inverse(T)
    else:
        Tinv = torch.pinverse(T)
    batch_size = outputs.shape[0]
    outputs = -F.log_softmax(outputs, dim=1)
    loss_mat = torch.zeros([num_classes, num_classes], device=device)
    for k in range(num_classes):
        mask = k == labels
        indexes = torch.arange(batch_size).to(device)
        indexes = torch.masked_select(indexes, mask)
        if indexes.shape[0] > 0:
            outputs_k = outputs[indexes]
            # outputs_k = torch.gather(outputs, 0, indexes.view(-1, 1).repeat(1,num_classes))
            loss_mat[k] = class_prior[k] * outputs_k.mean(0)
    loss_vec = torch.zeros(num_classes, device=device)
    for k in range(num_classes):
        loss_vec[k] = torch.inner(Tinv[k], loss_mat[k])
    return loss_vec

def cpe_decode(model, dataloader, num_classes):
    device = model.device

    total = 0
    correct = 0
    U = torch.ones((num_classes, num_classes), device=device) * 1/9
    for i in range(num_classes):
        U[i][i] = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, num_classes, 1).repeat(1, 1, num_classes) - U.expand(outputs.shape[0], num_classes, num_classes)
            predicted = torch.argmin(outputs.norm(dim=1), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def robust_ga_loss(outputs, labels, class_prior, T, num_classes):
    device = labels.device
    def l_mae(y, output):
        return 2 - 2 * F.softmax(output, dim=1)[:, y].mean()
    if torch.det(T) != 0:
        Tinv = torch.inverse(T)
    else:
        Tinv = torch.pinverse(T)
        
    loss_vec = torch.zeros(num_classes, device=device)
    for k in range(num_classes):
        for j in range(num_classes):
            mask = j == labels
            indexes = torch.arange(outputs.shape[0]).to(device)
            indexes = torch.masked_select(indexes, mask)
            if indexes.shape[0] > 0:
                loss_vec[k] += class_prior[j] * Tinv[j][k] * l_mae(k, outputs[indexes])
    return loss_vec