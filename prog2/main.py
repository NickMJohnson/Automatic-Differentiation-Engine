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

### hyperparameter settings and other constants
batch_size = 100
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
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
	pass
	# TODO students should implement this

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



if __name__ == "__main__":
	(train_dataset, test_dataset) = load_MNIST_dataset()
	(train_dataloader, test_dataloader) = construct_dataloaders(train_dataset, test_dataset, batch_size)
	
	torch.manual_seed(42)
	loss_fn = torch.nn.CrossEntropyLoss()

	print("Training with SGD:")
	model_sgd = make_fully_connected_model_part1_1()
	optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=alpha)
	start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
	end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
	
	if torch.cuda.is_available():
		start_time.record()
	import time
	start_wall_time = time.time()
	
	train_loss_sgd, train_acc_sgd, test_loss_sgd, test_acc_sgd, approx_tr_loss_sgd, approx_tr_acc_sgd = train(
		train_dataloader, test_dataloader, model_sgd, loss_fn, optimizer_sgd, epochs)
	
	end_wall_time = time.time()
	if torch.cuda.is_available():
		end_time.record()
		torch.cuda.synchronize()
		gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
		print(f"SGD GPU time: {gpu_time:.2f} seconds")
	print(f"SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")
	
	print("Training with Momentum SGD:")
	model_momentum = make_fully_connected_model_part1_1()
	optimizer_momentum = torch.optim.SGD(model_momentum.parameters(), lr=alpha, momentum=beta)
	
	start_wall_time = time.time()
	train_loss_momentum, train_acc_momentum, test_loss_momentum, test_acc_momentum, approx_tr_loss_momentum, approx_tr_acc_momentum = train(
		train_dataloader, test_dataloader, model_momentum, loss_fn, optimizer_momentum, epochs)
	end_wall_time = time.time()
	print(f"Momentum SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")
	
	print("Training with Adam:")
	model_adam = make_fully_connected_model_part1_1()
	optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=alpha_adam, betas=(rho1, rho2))
	
	start_wall_time = time.time()
	train_loss_adam, train_acc_adam, test_loss_adam, test_acc_adam, approx_tr_loss_adam, approx_tr_acc_adam = train(
		train_dataloader, test_dataloader, model_adam, loss_fn, optimizer_adam, epochs)
	end_wall_time = time.time()
	print(f"Adam Wall time: {end_wall_time - start_wall_time:.2f} seconds")
	
	print("Training with Batch Normalization + SGD:")
	model_bn = make_fully_connected_model_part1_4()
	optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=alpha_adam, momentum=beta)
	
	start_wall_time = time.time()
	train_loss_bn, train_acc_bn, test_loss_bn, test_acc_bn, approx_tr_loss_bn, approx_tr_acc_bn = train(
		train_dataloader, test_dataloader, model_bn, loss_fn, optimizer_bn, epochs)
	end_wall_time = time.time()
	print(f"Batch Norm SGD Wall time: {end_wall_time - start_wall_time:.2f} seconds")
	
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
	
	plot_metrics('SGD', approx_tr_loss_sgd, train_loss_sgd, test_loss_sgd, 
	             approx_tr_acc_sgd, train_acc_sgd, test_acc_sgd, epochs_range)
	
	plot_metrics('Momentum SGD', approx_tr_loss_momentum, train_loss_momentum, test_loss_momentum,
	             approx_tr_acc_momentum, train_acc_momentum, test_acc_momentum, epochs_range)
	
	plot_metrics('Adam', approx_tr_loss_adam, train_loss_adam, test_loss_adam,
	             approx_tr_acc_adam, train_acc_adam, test_acc_adam, epochs_range)
	
	plot_metrics('Batch Norm + SGD', approx_tr_loss_bn, train_loss_bn, test_loss_bn,
	             approx_tr_acc_bn, train_acc_bn, test_acc_bn, epochs_range)
	
	print(f"Final Test Accuracies:")
	print(f"SGD: {test_acc_sgd[-1]:.4f}")
	print(f"Momentum SGD: {test_acc_momentum[-1]:.4f}")
	print(f"Adam: {test_acc_adam[-1]:.4f}")
	print(f"Batch Norm + SGD: {test_acc_bn[-1]:.4f}")
