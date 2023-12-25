import torch
import torchvision
from functools import partial
from tqdm.auto import tqdm#_notebook as tqdm
from datetime import datetime
import os 

def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def criterion(prediction, mask, regr, weight=0.4, mask_loss_type="default", label_smoothing=0.0, size_average=True):
    # Binary mask loss
    if mask_loss_type == "default":
        pred_mask = torch.sigmoid(prediction[:, 0])
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()
    elif mask_loss_type == "focal":
        # mask_loss = modified_focal_loss(prediction[:, 0], mask)
        mask_loss = torchvision.ops.sigmoid_focal_loss(prediction[:, 0], mask, reduction="mean")
    elif mask_loss_type == "bce":
        mask_loss = torch.nn.functional.cross_entropy(prediction[:, 0], mask, label_smoothing=label_smoothing)
    else:
        raise ValueError("mask_loss_type must be either default or focal")
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    # loss = weight * mask_loss + (1 - weight) * regr_loss
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
        mask_loss *= prediction.shape[0]
        regr_loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss

def train_model(epoch, n_epochs, model, train_loader, optimizer, scheduler, criterion, switch_loss_epoch=0, history=None, device='cuda', tqdm_disabled=False):
    model.train()

    start = datetime.now()
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader, disable=tqdm_disabled)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        if epoch < switch_loss_epoch:
            loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, weight=1)
        else:
            loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, weight=0.5)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_mask_loss'] = mask_loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_regr_loss'] = regr_loss.data.cpu().numpy()
            # also save lr
            history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()
    end = datetime.now()

    print('Train Epoch: {}/{} \tLR: {:.6f}\tTime: {}\tSLE: {}\nTrn Loss: {:.6f}\tMask Loss: {:.6f}\tRegr Loss: {:.6f}'.format(
        epoch,
        n_epochs,
        optimizer.state_dict()['param_groups'][0]['lr'],
        end - start,
        switch_loss_epoch,
        loss.data,
        mask_loss.data,
        regr_loss.data,
        ))

def evaluate_model(epoch, model, dev_loader, criterion, switch_loss_epoch=0, history=None, device='cuda'):
    model.eval()
    total_loss = 0
    total_mask_loss = 0
    total_regr_loss = 0

    start = datetime.now()
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)
            
            if epoch < switch_loss_epoch:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, weight=1, size_average=False)
            else:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, weight=0.5, size_average=False)
            total_loss += loss.data
            total_mask_loss += mask_loss.data
            total_regr_loss += regr_loss.data
    end = datetime.now()

    total_loss /= len(dev_loader.dataset)
    total_mask_loss /= len(dev_loader.dataset)
    total_regr_loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch+1, 'dev_loss'] = total_loss.cpu().numpy()
        history.loc[epoch+1, 'dev_mask_loss'] = total_mask_loss.cpu().numpy()
        history.loc[epoch+1, 'dev_regr_loss'] = total_regr_loss.cpu().numpy()
    
    print('Dev loss: {:.4f}\tMask Loss: {:.4f}\tRegr Loss: {:.4f}'.format(
        total_loss, 
        total_mask_loss, 
        total_regr_loss
        ))
    
def get_save_folder(output_path, ctime_str):
    if output_path is None:
        save_folder = f"./{ctime_str}-None"
    else:
        basename = os.path.basename(output_path)
        folder_name, _ = os.path.splitext(basename)
        # time info is already in basename
        save_folder = f"./{folder_name}"

    return save_folder