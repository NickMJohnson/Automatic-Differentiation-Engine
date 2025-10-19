#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import torch
import torchvision
## you may wish to import other things like torch.nn

import time
from itertools import product
import csv
import math
import random

### hyperparameter settings and other constants
batch_size = 100
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.001
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
	train_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = True)
	test_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = False)
	return (train_dataset, test_dataset)

# construct dataloaders for the MNIST dataset
#
# train_dataset        input train dataset (output of load_MNIST_dataset)
# test_dataset         input test dataset (output of load_MNIST_dataset)
# batch_size           batch size for training
# shuffle_train        boolean: whether to shuffle the training dataset
#
# returns              tuple of (train_dataloader, test_dataloader)
#     each component of the tuple should be a torch.utils.data.DataLoader object
#     for the corresponding training set;
#     use the specified batch_size and shuffle_train values for the training DataLoader;
#     use a batch size of 100 and no shuffling for the test data loader
def construct_dataloaders(train_dataset, test_dataset, batch_size, shuffle_train=True):
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle_train
	)
	test_dataloader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=100,
		shuffle=False
	)
	return (train_dataloader, test_dataloader)


# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
	model.eval()
	total_loss, correct, total = 0.0, 0, 0

	for _, (data, target) in enumerate(dataloader):
		data, target = data.to(device), target.to(device)
		output = model(data)
		total_loss += loss_fn(output, target).item()		
		correct += (output.argmax(dim=1) == target).sum().item()
		total += target.size(0)
	
	average_loss, accuracy = total_loss / len(dataloader), correct / total
	return (average_loss, accuracy)

# build a fully connected two-hidden-layer neural network for MNIST data, as in Part 1.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_1():
	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(28 * 28, d1),
		torch.nn.ReLU(),
		torch.nn.Linear(d1, d2),
		torch.nn.ReLU(),
		torch.nn.Linear(d2, num_classes)
	)
	return model

# build a fully connected two-hidden-layer neural network with Batch Norm, as in Part 1.4
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_4():
	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(28 * 28, d1),  # d1 = 1024
		torch.nn.BatchNorm1d(d1),
		torch.nn.ReLU(),
		torch.nn.Linear(d1, d2),       # d2 = 256
		torch.nn.BatchNorm1d(d2),
		torch.nn.ReLU(),
		torch.nn.Linear(d2, num_classes)  # c = 10
	)
	return model

# build a convolutional neural network, as in Part 3.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_cnn_model_part3_1():
	model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, stride=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, stride=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, stride=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=512, out_features=128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=num_classes)
    )
	return model

# build a convolutional neural network that improves upon the model from 3.1
#
# returns   a new model of type torch.nn.Sequential


# train a neural network on MNIST data
#     be sure to call model.train() before training and model.eval() before evaluating!
#
# train_dataloader   training dataloader
# test_dataloader    test dataloader
# model              dnn model to be trained (training should mutate this)
# loss_fn            loss function
# optimizer          an optimizer that inherits from torch.optim.Optimizer
# epochs             number of epochs to run
# eval_train_stats   boolean; whether to evaluate statistics on training set each epoch
# eval_test_stats    boolean; whether to evaluate statistics on test set each epoch
#
# returns   a tuple of
#   train_loss       an array of length `epochs` containing the training loss after each epoch, or [] if eval_train_stats == False
#   train_acc        an array of length `epochs` containing the training accuracy after each epoch, or [] if eval_train_stats == False
#   test_loss        an array of length `epochs` containing the test loss after each epoch, or [] if eval_test_stats == False
#   test_acc         an array of length `epochs` containing the test accuracy after each epoch, or [] if eval_test_stats == False
#   approx_tr_loss   an array of length `epochs` containing the average training loss of examples processed in this epoch
#   approx_tr_acc    an array of length `epochs` containing the average training accuracy of examples processed in this epoch
def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, eval_train_stats=True, eval_test_stats=True):
	train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc = [], [], [], [], [], []
	
	for _ in range(epochs):
		model.train()
		epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
		for _, (data, target) in enumerate(train_dataloader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = loss_fn(output, target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			epoch_correct += (output.argmax(dim=1) == target).sum().item()
			epoch_total += target.size(0)
		
		avg_epoch_loss, avg_epoch_acc = epoch_loss / len(train_dataloader), epoch_correct / epoch_total
		approx_tr_loss.append(avg_epoch_loss)
		approx_tr_acc.append(avg_epoch_acc)
		
		if eval_train_stats:
			train_loss_val, train_acc_val = evaluate_model(train_dataloader, model, loss_fn)
			train_loss.append(train_loss_val)
			train_acc.append(train_acc_val)
		if eval_test_stats:
			test_loss_val, test_acc_val = evaluate_model(test_dataloader, model, loss_fn)
			test_loss.append(test_loss_val)
			test_acc.append(test_acc_val)
	
	return (train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc)

############################################
# PART 2: Hyperparameter Search

def _make_train_val_loaders(train_dataset, batch_size, val_fraction=0.1, seed=42):
    """
    Splitting the original training set into train/val subsets.
    Returns train_loader, val_loader.
    Test loader remains the same.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(train_dataset)
    n_val = int(round(val_fraction * n))
    n_train = n - n_val
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=g)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def _evaluate_on(dataloader, model, loss_fn):
    
    loss, acc = evaluate_model(dataloader, model, loss_fn)
    return {"loss": loss, "acc": acc}

def _run_once(model_fn, optimizer_fn, loss_fn, train_loader, val_loader, test_loader, epochs):
    """
    Single experiment run:
      - build model
      - build optimizer
      - train for `epochs`
      - get val/test metrics at the end
    Returns a result dict including per-epoch histories from the `train`.
    """
    model = model_fn().to(device)
    opt   = optimizer_fn(model.parameters())
    trL, trA, teL, teA, approxL, approxA = train(train_loader, test_loader, model, loss_fn, opt, epochs, eval_train_stats=False, eval_test_stats=True)
    val_metrics  = _evaluate_on(val_loader,  model, loss_fn)
    test_metrics = {"loss": teL[-1], "acc": teA[-1]} if len(teL) > 0 else _evaluate_on(test_loader, model, loss_fn)
    return {
        "val_loss":  val_metrics["loss"],
        "val_acc":   val_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "test_acc":  test_metrics["acc"],
        "history":   {
            "approx_tr_loss": approxL, "approx_tr_acc": approxA,
            "test_loss": teL, "test_acc": teA
        }
    }

def _save_csv(rows, path):
    """Write list of dicts to CSV (keys from the first row)."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

##################################################
# 2.1 Grid over step size alpha for Momentum-SGD
##################################################

def part2_1_grid_alpha_momentum(train_dataset, test_dataset, base_model_fn, momentum=0.9, epochs=5, batch_size_for_search=100, seed=42):
    """
    Grid search alpha for SGD+momentum over:
      {1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001}
    Reports validation metrics and best alpha.
    """
    alphas = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    loss_fn = torch.nn.CrossEntropyLoss()

    # creating loaders (train split into train/val)
    train_loader, val_loader = _make_train_val_loaders(train_dataset, batch_size_for_search, val_fraction=0.1, seed=seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_for_search, shuffle=False)

    results = []
    for a in alphas:
        def _opt_fn(params):
            return torch.optim.SGD(params, lr=a, momentum=momentum)
        res = _run_once(base_model_fn, _opt_fn, loss_fn, train_loader, val_loader, test_loader, epochs)
        results.append({
            "alpha": a,
            "momentum": momentum,
            "val_acc": round(res["val_acc"], 6),
            "val_loss": round(res["val_loss"], 6),
            "test_acc": round(res["test_acc"], 6),
            "test_loss": round(res["test_loss"], 6)
        })
        print(f"[2.1] alpha={a:<7}  val_acc={results[-1]['val_acc']:.4f}  test_acc={results[-1]['test_acc']:.4f}")

    # pick best by validation accuracy
    best = max(results, key=lambda r: r["val_acc"])
    print(f"[2.1] Best alpha={best['alpha']} (val_acc={best['val_acc']:.4f}, test_acc={best['test_acc']:.4f})")
    _save_csv(results, "part2_1_alpha_grid.csv")
    return results, best

################################################################
# 2.2 General Grid Search (picking 3 hyperparams form part 1)
################################################################

def part2_2_grid_search(train_dataset, test_dataset, base_model_fn, epochs=5, seed=42):
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # defining grids
    grid_alpha = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    grid_beta  = [0.8, 0.9, 0.95]
    grid_wd    = [0.0, 1e-4, 1e-3]

    # loaders
    batch_size_for_search = 100
    train_loader, val_loader = _make_train_val_loaders(train_dataset, batch_size_for_search, val_fraction=0.1, seed=seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_for_search, shuffle=False)

    results = []
    for a, b, wd in product(grid_alpha, grid_beta, grid_wd):
        def _opt_fn(params):
            return torch.optim.SGD(params, lr=a, momentum=b, weight_decay=wd)
        res = _run_once(base_model_fn, _opt_fn, loss_fn, train_loader, val_loader, test_loader, epochs)
        row = {
            "alpha": a, "momentum": b, "weight_decay": wd,
            "val_acc": round(res["val_acc"], 6),
            "val_loss": round(res["val_loss"], 6),
            "test_acc": round(res["test_acc"], 6),
            "test_loss": round(res["test_loss"], 6)
        }
        results.append(row)
        print(f"[2.2] Alpha={a:<.3g} Beta={b:<.2f} wd={wd:<.0e} | val_acc={row['val_acc']:.4f} test_acc={row['test_acc']:.4f}")

    best = max(results, key=lambda r: r["val_acc"])
    print(f"[2.2] Best: Alpha={best['alpha']}, Beta={best['momentum']}, wd={best['weight_decay']} (val_acc={best['val_acc']:.4f}, test_acc={best['test_acc']:.4f})")
    _save_csv(results, "part2_2_grid.csv")
    return results, best

##########################################
# 2.3 Random search over the same space
##########################################

def _sample_log_uniform(low, high):
    """Sampling from log-uniform in [low, high]."""
    return math.exp(random.uniform(math.log(low), math.log(high)))

def part2_3_random_search(train_dataset, test_dataset, base_model_fn, epochs=5, n_trials=15, seed=123):
    """
    Random search over the same 3 params as in part 2.2:
      - alpha ~ log-uniform[1e-4, 1e-1]
      - beta ~ uniform[0.8, 0.99]
      - weight_decay ~ {0.0, 1e-5, 1e-4, 1e-3}
    """
    random.seed(seed)
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size_for_search = 100
    train_loader, val_loader = _make_train_val_loaders(train_dataset, batch_size_for_search, val_fraction=0.1, seed=seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_for_search, shuffle=False)

    wd_choices = [0.0, 1e-5, 1e-4, 1e-3]
    results = []
    for t in range(n_trials):
        a  = _sample_log_uniform(1e-4, 1e-1)
        b  = random.uniform(0.8, 0.99)
        wd = random.choice(wd_choices)
        def _opt_fn(params):
            return torch.optim.SGD(params, lr=a, momentum=b, weight_decay=wd)
        res = _run_once(base_model_fn, _opt_fn, loss_fn, train_loader, val_loader, test_loader, epochs)
        row = {
            "trial": t+1,
            "alpha": a, "momentum": b, "weight_decay": wd,
            "val_acc": round(res["val_acc"], 6),
            "val_loss": round(res["val_loss"], 6),
            "test_acc": round(res["test_acc"], 6),
            "test_loss": round(res["test_loss"], 6)
        }
        results.append(row)
        print(f"[2.3] trial={t+1:02d} Alpha≈{a:.4g} Beta={b:.3f} wd={wd:<.0e} | val_acc={row['val_acc']:.4f} test_acc={row['test_acc']:.4f}")

    best = max(results, key=lambda r: r["val_acc"])
    print(f"[2.3] Best random: Alpha={best['alpha']:.5f}, Beta={best['momentum']:.3f}, wd={best['weight_decay']} (val_acc={best['val_acc']:.4f}, test_acc={best['test_acc']:.4f})")
    _save_csv(results, "part2_3_random.csv")
    return results, best

def run_part2_all(train_dataset, test_dataset, epochs_each=5):
    """
    Runs 2.1, 2.2, 2.3 in order using the Part 1.1 MLP builder.
    """
    base_model_fn = make_fully_connected_model_part1_1  # choosing base model func

    print("\n=== Part 2.1: Grid over learning rate (SGD + momentum) ===")
    _, best_21 = part2_1_grid_alpha_momentum(train_dataset, test_dataset, base_model_fn, momentum=0.9, epochs=epochs_each)

    print("\n=== Part 2.2: Grid search over (alpha, momentum, weight_decay) ===")
    _, best_22 = part2_2_grid_search(train_dataset, test_dataset, base_model_fn, epochs=epochs_each)

    print("\n=== Part 2.3: Random search over same space (>=10 trials) ===")
    _, best_23 = part2_3_random_search(train_dataset, test_dataset, base_model_fn, epochs=epochs_each, n_trials=15)

    print("\nSummary (best val):")
    print(f"  2.1 best: Alpha={best_21['alpha']}  Beta=0.9  test_acc={best_21['test_acc']:.4f}")
    print(f"  2.2 best: Alpha={best_22['alpha']}  Beta={best_22['momentum']}  wd={best_22['weight_decay']}  test_acc={best_22['test_acc']:.4f}")
    print(f"  2.3 best: Alpha={best_23['alpha']:.5f}  Beta={best_23['momentum']:.3f}  wd={best_23['weight_decay']}  test_acc={best_23['test_acc']:.4f}")

if __name__ == "__main__":
    (train_dataset, test_dataset) = load_MNIST_dataset()
    (train_dataloader, test_dataloader) = construct_dataloaders(train_dataset, test_dataset, batch_size)

    torch.manual_seed(42)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Prefer CUDA, otherwise use Apple MPS if available, or else CPU.
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    print("Training with SGD:")
    model_sgd = make_fully_connected_model_part1_1().to(device)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=alpha)
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        start_time.record()
    import time
    start_wall_time = time.time()

    train_loss_sgd, train_acc_sgd, test_loss_sgd, test_acc_sgd, approx_tr_loss_sgd, approx_tr_acc_sgd = train(
        train_dataloader, test_dataloader, model_sgd, loss_fn, optimizer_sgd, epochs
    )

    end_wall_time = time.time()
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time) / 1000.0
        print(f"SGD GPU time: {gpu_time:.2f} seconds")
    print(f"SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")

    print("Training with Momentum SGD:")
    model_momentum = make_fully_connected_model_part1_1().to(device)
    optimizer_momentum = torch.optim.SGD(model_momentum.parameters(), lr=alpha, momentum=beta)

    start_wall_time = time.time()
    train_loss_momentum, train_acc_momentum, test_loss_momentum, test_acc_momentum, approx_tr_loss_momentum, approx_tr_acc_momentum = train(
        train_dataloader, test_dataloader, model_momentum, loss_fn, optimizer_momentum, epochs
    )
    end_wall_time = time.time()
    print(f"Momentum SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")

    print("Training with Adam:")
    model_adam = make_fully_connected_model_part1_1().to(device)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=alpha_adam, betas=(rho1, rho2))

    start_wall_time = time.time()
    train_loss_adam, train_acc_adam, test_loss_adam, test_acc_adam, approx_tr_loss_adam, approx_tr_acc_adam = train(
        train_dataloader, test_dataloader, model_adam, loss_fn, optimizer_adam, epochs
    )
    end_wall_time = time.time()
    print(f"Adam Wall time: {end_wall_time - start_wall_time:.2f} seconds")

    print("Training with Batch Normalization + SGD:")
    model_bn = make_fully_connected_model_part1_4().to(device)
    optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=alpha_adam, momentum=beta)

    start_wall_time = time.time()
    train_loss_bn, train_acc_bn, test_loss_bn, test_acc_bn, approx_tr_loss_bn, approx_tr_acc_bn = train(
        train_dataloader, test_dataloader, model_bn, loss_fn, optimizer_bn, epochs
    )
    end_wall_time = time.time()
    print(f"Batch Norm SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")

    (train_dataloader, test_dataloader) = construct_dataloaders(train_dataset, test_dataset, batch_size)
    print("Training with CNN + Adam")
    model_cnn = make_cnn_model_part3_1()
    model_cnn.to(device)
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=alpha, betas=(rho1, rho2))
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        start_time.record()
    start_wall_time = time.time()

    train_loss_cnn, train_acc_cnn, test_loss_cnn, test_acc_cnn, approx_tr_loss_cnn, approx_tr_acc_cnn = train(
        train_dataloader, test_dataloader, model_cnn, loss_fn, optimizer_cnn, epochs
    )

    end_wall_time = time.time()
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time) / 1000.0
        print(f"CNN + Adam GPU time: {gpu_time:.2f} seconds")
    else:
        print(f"CNN + Adam Wall time: {end_wall_time - start_wall_time:.2f} seconds")

    epochs_range = range(1, epochs + 1)

    def plot_metrics(algorithm_name, approx_tr_loss, train_loss, test_loss,
                     approx_tr_acc, train_acc, test_acc, epochs_range):
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(epochs_range, approx_tr_loss, 'b-', label='Approx Training Loss', linewidth=2)
        pyplot.plot(epochs_range, train_loss, 'r-', label='End-of-Epoch Training Loss', linewidth=2)
        pyplot.plot(epochs_range, test_loss, 'g-', label='Test Loss', linewidth=2)
        pyplot.title(f'{algorithm_name} - Loss vs Epoch')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(f'{algorithm_name.lower().replace(" ", "_")}_loss.png', dpi=300, bbox_inches='tight')

        pyplot.figure(figsize=(10, 6))
        pyplot.plot(epochs_range, approx_tr_acc, 'b-', label='Approx Training Accuracy', linewidth=2)
        pyplot.plot(epochs_range, train_acc, 'r-', label='End-of-Epoch Training Accuracy', linewidth=2)
        pyplot.plot(epochs_range, test_acc, 'g-', label='Test Accuracy', linewidth=2)
        pyplot.title(f'{algorithm_name} - Accuracy vs Epoch')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(f'{algorithm_name.lower().replace(" ", "_")}_accuracy.png', dpi=300, bbox_inches='tight')

    def plot_metrics2(algorithm_name, approx_tr_loss, test_loss,
                      approx_tr_acc, test_acc, epochs_range):
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(epochs_range, approx_tr_loss, 'b-', label='Approx Training Loss (Minibatches)', linewidth=2)
        pyplot.plot(epochs_range, test_loss, 'g-', label='Test Loss', linewidth=2)
        pyplot.title(f'{algorithm_name} - Loss vs Epoch')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(f'{algorithm_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_loss.png', dpi=300, bbox_inches='tight')

        pyplot.figure(figsize=(10, 6))
        pyplot.plot(epochs_range, approx_tr_acc, 'b-', label='Approx Training Accuracy (Minibatches)', linewidth=2)
        pyplot.plot(epochs_range, test_acc, 'g-', label='Test Accuracy', linewidth=2)
        pyplot.title(f'{algorithm_name} - Accuracy vs Epoch')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(f'{algorithm_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_accuracy.png', dpi=300, bbox_inches='tight')

    plot_metrics('SGD', approx_tr_loss_sgd, train_loss_sgd, test_loss_sgd,
                 approx_tr_acc_sgd, train_acc_sgd, test_acc_sgd, epochs_range)

    plot_metrics('Momentum SGD', approx_tr_loss_momentum, train_loss_momentum, test_loss_momentum,
                 approx_tr_acc_momentum, train_acc_momentum, test_acc_momentum, epochs_range)

    plot_metrics('Adam', approx_tr_loss_adam, train_loss_adam, test_loss_adam,
                 approx_tr_acc_adam, train_acc_adam, test_acc_adam, epochs_range)

    plot_metrics('Batch Norm + SGD', approx_tr_loss_bn, train_loss_bn, test_loss_bn,
                 approx_tr_acc_bn, train_acc_bn, test_acc_bn, epochs_range)

    plot_metrics2('CNN + Adam', approx_tr_loss_cnn, test_loss_cnn,
                  approx_tr_acc_cnn, test_acc_cnn, epochs_range)

    print("Final Test Accuracies:")
    print(f"SGD: {test_acc_sgd[-1]:.4f}")
    print(f"Momentum SGD: {test_acc_momentum[-1]:.4f}")
    print(f"Adam: {test_acc_adam[-1]:.4f}")
    print(f"Batch Norm + SGD: {test_acc_bn[-1]:.4f}")
    print(f"CNN + Adam: {test_acc_cnn[-1]:.4f}")

    # Running Part 2
    run_part2_all(train_dataset, test_dataset, epochs_each=5)
