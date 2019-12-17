import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff
from losses_pytorch import dice_loss as losses

def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.module.n_classes > 1:
                    tot += losses.dice_coeff(pred, true_mask)
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
