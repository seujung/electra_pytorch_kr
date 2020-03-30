import torch


def calc_mask_accuracy(pred_value, label_lm, mask_idx):
    pred = list()
    label = list()
    for i in range(mask_idx.shape[0]):
        tmp_idx = mask_idx[i]
        for j in tmp_idx:
            if j > 0:
                pred.append(pred_value[i][j].cpu())
                label.append(label_lm[i][j].cpu())
    pred = torch.tensor(pred)
    label = torch.tensor(label)
    
    accuracy = pred.eq(label).sum().item() / label.shape[0]
    return accuracy
    

def calc_gan_accuracy(active_logits, active_labels):
    pred = active_logits >=0.5
    label = active_labels >= 0.5
    accuracy = pred.eq(label).sum().item() / label.shape[0]
    return accuracy