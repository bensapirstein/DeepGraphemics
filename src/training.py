import os
import abc
import sys
import tqdm
import torch
import numpy as np
from typing import Any, Callable
from torch.utils.data import DataLoader

from .train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_epoch_result = self.train_epoch(dl_train, **kw)
            train_loss.extend(train_epoch_result.losses)
            train_acc.append(train_epoch_result.accuracy)
            
            test_epoch_result = self.test_epoch(dl_test, **kw)
            test_loss.extend(test_epoch_result.losses)
            test_acc.append(test_epoch_result.accuracy)

            if early_stopping is not None:
                if best_acc is None or test_epoch_result.accuracy > best_acc:
                    best_acc = test_epoch_result.accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        break

            if checkpoints is not None:
                torch.save(self.model, checkpoints)

            actual_num_epochs += 1

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=[avg_loss], accuracy=accuracy)


class VAETrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # Forward pass
        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = self.model(X)

        # reconstruction error
        loss = self.loss_fn(image_batch_recon, X, latent_mu, latent_logvar)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        # TODO: if model is also a classifier - calculate accuracy
        num_correct = 0

        return BatchResult(loss.item(), num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # Forward pass
            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = self.model(X)

            # reconstruction error
            loss = self.loss_fn(image_batch_recon, X, latent_mu, latent_logvar)

            num_correct = 0

        return BatchResult(loss.item(), num_correct)

class CapsNetTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)
        self.n_classes = model.dc_num_capsules

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        y = torch.sparse.torch.eye(self.n_classes).index_select(dim=0, index=y)

        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # Forward pass
        # vae reconstruction
        output, reconstructions, masked = self.model(X)

        # reconstruction error
        loss = self.loss_fn(X, output, y, reconstructions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        num_correct = sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                          np.argmax(y.data.cpu().numpy(), 1))

        return BatchResult(loss.item(), num_correct)
    
    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        y = torch.sparse.torch.eye(self.n_classes).index_select(dim=0, index=y)

        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # Forward pass
            output, reconstructions, masked = self.model(X)
            loss = self.loss_fn(X, output, y, reconstructions)

            num_correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                              np.argmax(y.data.cpu().numpy(), 1))
            
        return BatchResult(loss.item(), num_correct)


