import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass 
from torchvision import datasets, transforms
from typing import NamedTuple
import chex
import jax.numpy as jnp    
import jax
import numpy as np
import multiprocessing as mp 


def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)
        
def jax_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.array(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)

def subsample_dataset(dataset, num_points):
    targets = dataset.targets.numpy()
    unique_labels = np.unique(targets)
    indices = []
    for label in unique_labels:
        label_indices = np.where(targets == label)[0]
        selected_indices = np.random.choice(label_indices, num_points, replace=False)
        indices.extend(selected_indices)
    
    indices = np.array(indices)
    #np.random.shuffle(indices)  # Shuffle to mix different classes
    return torch.utils.data.Subset(dataset, indices)

def subsample_dataset_jax(dataset, num_points):
    targets = dataset.targets.numpy()
    images = dataset.data.numpy()
    unique_labels = np.unique(targets)
    
    pool = mp.Pool(mp.cpu_count())
    all_indices = jnp.zeros((unique_labels.shape[0], num_points))
    rng = jax.random.PRNGKey(0)
    batches = []
    with pool as p:
        for label in unique_labels:
            rng, _rng = jax.random.split(rng)
            indices = jnp.where(targets == label)[0]
            #print(indices.shape)
            selected_indices = jax.random.choice(_rng, indices, (num_points,), replace=False)
            batch = DataBatch(
                images=images[selected_indices, :, :],
                labels=targets[selected_indices]
            )
            batches.append(batch)
    return jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *batches)

def fast_subsample_dataset(rng, dataset, num_points):
    targets = dataset.targets.numpy()
    unique_labels = jnp.unique(targets)
    
    def select_indices(carry, label): 
        label_indices = jnp.where(targets == label)[0] 
        selected_indices = jnp.random.choice(rng, label_indices, num_points, replace=False)
        carry = carry.at[label, :].set(selected_indices)
        return carry, None 
    
    carry = jnp.zeros((unique_labels.shape[0], num_points))
    carry, batches = jax.lax.scan(
        f = select_indices,
        init = carry,
        xs = unique_labels
    )
    indices = carry.flatten() 
    return torch.utils.data.Subset(dataset, indices)

@dataclass
class MNISTData:
    train_loader = DataLoader
    test_loader = DataLoader
    batch_size: int

    def __post_init__(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        new_train_dataset = subsample_dataset(train_dataset, 1000)
        self.train_loader = DataLoader(new_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=numpy_collate)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=numpy_collate)


#@chex.dataclass
class DataBatch(NamedTuple):
    images: chex.Array
    labels: chex.Array

def jax_collate(batch):
    return jnp.array(batch)

        


@chex.dataclass 
class MNISTDataJAX:
    data_points_per_class: int

    def __post_init__(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.MEAN = jnp.array([0.1307]).reshape((1, 1, 1, 1))
        self.STD = jnp.array([0.3081]).reshape((1, 1, 1, 1))
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        train_data = subsample_dataset_jax(train_dataset, self.data_points_per_class)
        self.train_set = DataBatch(
            images=(train_data.images.reshape((-1, train_data.images.shape[-2], train_data.images.shape[-1], 1))/255 - self.MEAN)/self.STD,
            labels=train_data.labels
        )
        test_images = jnp.array(test_dataset.data.numpy())
        test_labels = jnp.array(test_dataset.targets.numpy())
        self.validation_set = DataBatch(
            images=(test_images.reshape(-1, test_images.shape[-2], test_images.shape[-1], 1)/255 - self.MEAN)/self.STD,
            labels=test_labels
        )

@chex.dataclass 
class FashionMNISTDataJAX:
    data_points_per_class: int
    standardize: bool = True

    def __post_init__(self):
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True)
        test_dataset = datasets.FashionMNIST('../data', train=False)
        if self.standardize:
            self.MEAN = jnp.array(jnp.mean(train_dataset.data.numpy()/255, axis=(0, 1, 2))).reshape((1, 1, 1, 1))
            self.STD = jnp.array(jnp.std(train_dataset.data.numpy()/255, axis=(0, 1, 2))).reshape((1, 1, 1, 1))
        else:
            self.MEAN = jnp.array([0]).reshape((1, 1, 1, 1))
            self.STD = jnp.array([1]).reshape((1, 1, 1, 1))
        train_data = subsample_dataset_jax(train_dataset, self.data_points_per_class)
        self.train_set = DataBatch(
            images=(train_data.images.reshape((-1, train_data.images.shape[-2], train_data.images.shape[-1], 1))/255 - self.MEAN)/self.STD,
            labels=train_data.labels
        )
        test_images = jnp.array(test_dataset.data.numpy())
        test_labels = jnp.array(test_dataset.targets.numpy())
        self.validation_set = DataBatch(
            images=(test_images.reshape((-1, test_images.shape[-2], test_images.shape[-1], 1))/255 - self.MEAN)/self.STD,
            labels=test_labels
        )

@chex.dataclass
class CIFAR10DataJAX:
    data_points_per_class: int

    def __post_init__(self):
        self.MEAN = jnp.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 1, 3))
        self.STD = jnp.array([0.247, 0.243, 0.261]).reshape((1, 1, 1, 3))
        train_dataset = datasets.CIFAR10('../data', train=True, download=True)
        test_dataset = datasets.CIFAR10('../data', train=False)
        train_images = jnp.array(train_dataset.data)
        train_labels = jnp.array(train_dataset.targets)
        self.train_set = DataBatch(
            images=(train_images/255 - self.MEAN)/self.STD,
            labels=train_labels
        )
        test_images = jnp.array(test_dataset.data)
        test_labels = jnp.array(test_dataset.targets)
        self.validation_set = DataBatch(
            images=(test_images/255 - self.MEAN)/self.STD,
            labels=test_labels
        )
        
@chex.dataclass
class CIFAR100DataJAX:
    data_points_per_class: int

    def __post_init__(self):
        self.MEAN = jnp.array([0.5071, 0.4867, 0.4408]).reshape((1, 1, 1, 3))
        self.STD = jnp.array([0.2675, 0.2565, 0.2761]).reshape((1, 1, 1, 3))
        train_dataset = datasets.CIFAR100('../data', train=True, download=True)
        test_dataset = datasets.CIFAR100('../data', train=False)
        train_images = jnp.array(train_dataset.data) 
        train_labels = jnp.array(train_dataset.targets)
        self.train_set = DataBatch(
            images=(train_images/255 - self.MEAN)/self.STD,
            labels=train_labels
        )
        test_images = jnp.array(test_dataset.data)
        test_labels = jnp.array(test_dataset.targets)
        self.validation_set = DataBatch(
            images=(test_images/255 - self.MEAN)/self.STD,
            labels=test_labels
        )
