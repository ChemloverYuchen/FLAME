import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from FLAME.flsf.args import TrainArgs
from FLAME.dataprocess.flsf.data import MoleculeDataLoader, MoleculeDataset
from FLAME.flsf.models import MoleculeModel
from FLAME.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch. (epoch = iterations x batch)

    :param model: A :class:`~flsf.models.model.MoleculeModel`.
    :param data_loader: A :class:`~flsf.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~flsf.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """

    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.adj_features(), \
            batch.dist_features(), batch.clb_features()
#         print(target_batch[10])
#         print(mol_clb_batch[10])
        mask = torch.Tensor([[x is not None for x in tb]
                            for tb in target_batch])
        targets = torch.Tensor(
            [[0 if x is None else x for x in tb] for tb in target_batch])

        # Run model
        model.zero_grad()  # hignlights
        preds = model(mol_batch, features_batch, mol_adj_batch,
                      mol_dist_batch, mol_clb_batch)  # hignlights

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(
                1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()  # hignlights
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()  # hignlights update weights

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        # n_iter: Iterations is the number of batches needed to complete one epoch. (n_iter == num_batch ;dataset = batch_size x n_iter)
        # print the result every 10 iterations
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            # Computes the norm of the parameters of a model.
            pnorm = compute_pnorm(model)
            # Computes the norm of the gradients of a model.
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0

            lrs_str = ', '.join(
                f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(
                f'Loss = {loss_avg:.4e}, Par_Norm = {pnorm:.4f}, Grad_Norm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
