import jax 
import jax.numpy as jnp 


import training 
import models 
import data_handling 
import datasets 
import adversaries 

import optax 
import augmax 
import chex 
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Literal, Any, NamedTuple, Callable
import os 
import pickle 
import wandb 

class TrainingConfig(BaseModel):
    dataset: Literal[
        "mnist",
        "cifar10",
        "cifar100"
    ] = "mnist"
    model: Literal[
        "resnet1",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnet200"
    ] = "resnet1"
    seed: int = 123456
    lr: float = 0.1
    batch_size: int = 512
    eval_batch_size: int = 1024
    eval_interval: int = 10
    total_updates: int = 101
    dropout: bool = False
    batch_norm: bool = False
    optimizer: Literal[
        "adam",
        "sgd",
        "rmsprop",
        "adamw"
    ] = "adam"
    wandb: bool = True 
    save_model: bool = True

    class Config:
        extra = "forbid"


class EvalLogMetrics(NamedTuple): 
    validation_loss: chex.Array 
    validation_accuracy: chex.Array 
    adv_loss: chex.Array 
    adv_accuracy: chex.Array 


args = TrainingConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

if args.dropout: 
    args.model = f"dropout{args.model}"

wandb_project = f"constant-updates" 
wandb_name = f"adversarial-{args.dataset}-{args.model}-lr={args.lr}-batch_size={args.batch_size}"


# LOAD DATASET        
if args.dataset == "mnist": 
    dataset = datasets.MNISTDataJAX(data_points_per_class=1024*5)
    num_classes = 10
elif args.dataset == "cifar10": 
    dataset = datasets.CIFAR10DataJAX(data_points_per_class=1024*5)
    num_classes = 10
elif args.dataset == "cifar100": 
    dataset = datasets.CIFAR100DataJAX(data_points_per_class=1024*5)
    num_classes = 100

data_points_total = 1024*5*10 
updates_per_epoch = data_points_total // args.batch_size
epochs = args.total_updates // updates_per_epoch

train_set = dataset.train_set 
validation_set = dataset.validation_set
key = jax.random.PRNGKey(args.seed)
key, shuffle_key = jax.random.split(key)
batched_validation_set = data_handling.shuffle_and_batch_tree(shuffle_key, validation_set, args.eval_batch_size)

model = models.RESNET_CONSTRUCTOR[args.model]
initial_conv_config = {"kernel_size": (3, 3), "strides": 1, "padding": "SAME"}
model_fn = model(num_classes=num_classes, num_filters=32, initial_conv_config=initial_conv_config)

# Augmentation
list_of_transforms = []
list_of_transforms.append(augmax.Resize(40, 40))
list_of_transforms.append(augmax.RandomCrop(32, 32))
list_of_transforms.append(augmax.HorizontalFlip())
list_of_transforms.append(augmax.RandomBrightness())
list_of_transforms.append(augmax.RandomContrast())

transform = augmax.Chain(*list_of_transforms)
batch_transform = jax.jit(jax.vmap(transform))



key, init_key = jax.random.split(key)
variables = model_fn.init(init_key, jnp.ones((1,) + train_set.images.shape[1:]))
if args.batch_norm: 
    params, batch_stats = variables["params"], variables["batch_stats"] 
    #batch_stats = model_fn.init_batch_stats(jax.random.PRNGKey(args.seed), jnp.ones((1,) + train_set.images.shape[1:]))
    variables = {"params": params, "batch_stats": batch_stats}
else: 
    params = variables
    
train_loss_fn, eval_loss_fn = training.build_loss_fn(model_fn, distance_metric=training.cross_entropy, regularizer=None, batch_norm=args.batch_norm, dropout=args.dropout)
optimizer = optax.adam(learning_rate=args.lr)
pgd_linf, pgd_l2 = adversaries.build_pgd_adversaries(eval_loss_fn, epsilon=0.031, alpha=0.00078, num_steps=40, batch_norm=args.batch_norm)

train_on_batch = training.build_train_step(train_loss_fn, optimizer, batch_norm=args.batch_norm, dropout=args.dropout)
eval_on_batch = training.build_eval_batch(eval_loss_fn, batch_norm=args.batch_norm)

if args.batch_norm: 
    class State(NamedTuple): 
        params: Any
        batch_stats: Any
        opt_state: Any 
        opt_step: Any
        epoch: Any
        rng: Any
    state = State(params=params, batch_stats=batch_stats, opt_state=optimizer.init(params), opt_step=0, epoch=0, rng=jax.random.PRNGKey(args.seed))
else: 
    class State(NamedTuple): 
        params: Any
        opt_state: Any
        opt_step: Any
        epoch: Any
        rng: Any
    state = State(params=params, opt_state=optimizer.init(params), opt_step=0, epoch=0, rng=jax.random.PRNGKey(args.seed))

def build_train_function(train_on_batch, eval_on_batch, adversary): 
    def evaluate(state, batched_validation_set): 
        state, eval_metrics = jax.lax.scan(
            eval_on_batch, 
            state, 
            batched_validation_set 
        )
        if args.batch_norm: 
            variables = {"params": state.params, "batch_stats": state.batch_stats}
        else: 
            variables = {"params": state.params}
        rng, rng_adversary = jax.random.split(state.rng)
        (variables, rng_adversary), batched_adv_validation_set = jax.lax.scan(
            adversary, 
            (variables, rng_adversary), 
            batched_validation_set
        )
        state = state._replace(rng=rng)
        state, adv_metrics = jax.lax.scan(
            eval_on_batch, 
            state, 
            batched_adv_validation_set 
        )
        # print(eval_metrics.loss)
        # print(adv_metrics.loss)
        log_metrics = EvalLogMetrics(
            validation_loss=eval_metrics.loss, 
            validation_accuracy=eval_metrics.accuracy, 
            adv_loss=adv_metrics.loss, 
            adv_accuracy=adv_metrics.accuracy
        )
        return state, log_metrics 
    
    def train_epoch(carry, unused):     
        state, train_set, batched_validation_set = carry
        # Here you can add a data augmentation step 
        if args.batch_norm: 
            variables = {"params": state.params, "batch_stats": state.batch_stats}
        else: 
            variables = {"params": state.params}
        # rng, rng_transform = jax.random.split(state.rng)
        # rngs = jax.random.split(rng_transform, train_set.images.shape[0])
        # augmented_train_set = data_handling.DataBatch(images=batch_transform(rngs, train_set.images), labels=train_set.labels)
        rng, rng_batch, rng_adversary = jax.random.split(state.rng, 3)
        batched_training_set = data_handling.shuffle_and_batch_tree(rng_batch, train_set, args.eval_batch_size)
        (variables, rng), batched_adv_training_set = jax.lax.scan(
            adversary, 
            (variables, rng_adversary), 
            batched_training_set
        )
        # print(batched_training_set.images.shape)
        # print(batched_adv_training_set.images.shape)
        state = state._replace(rng=rng)
        images = jnp.concatenate(
            [batched_training_set.images.reshape((-1, batched_training_set.images.shape[-3], batched_training_set.images.shape[-2], batched_training_set.images.shape[-1])), 
             batched_adv_training_set.images.reshape((-1, batched_adv_training_set.images.shape[-3], batched_adv_training_set.images.shape[-2], batched_adv_training_set.images.shape[-1]))], axis=0)
        labels = jnp.concatenate([batched_training_set.labels.reshape((-1,)), batched_adv_training_set.labels.reshape((-1,))], axis=0)
        training_set = data_handling.DataBatch(images=images, labels=labels)
        training_set = data_handling.DataBatch(images=training_set.images.reshape((-1, training_set.images.shape[-3], training_set.images.shape[-2], training_set.images.shape[-1])), labels=training_set.labels)
        batched_training_set = data_handling.shuffle_and_batch_tree(rng_batch, training_set, args.batch_size)
        #rng, rng_shuffle = jax.random.split(state.rng)
        #batched_training_set = data_handling.shuffle_and_batch_tree(rng_shuffle, augmented_train_set, args.batch_size)
        state, train_metrics = jax.lax.scan(
            train_on_batch, 
            state, 
            batched_training_set 
        )
        empty_log_metrics = EvalLogMetrics(
            validation_loss=jnp.zeros((batched_validation_set.images.shape[0],)), 
            validation_accuracy=jnp.zeros((batched_validation_set.images.shape[0],)), 
            adv_loss=jnp.zeros((batched_validation_set.images.shape[0],)), 
            adv_accuracy=jnp.zeros((batched_validation_set.images.shape[0],))
        )
        state, eval_metrics = jax.lax.cond(
            state.epoch % args.eval_interval == 0, 
            lambda: evaluate(state, batched_validation_set), 
            lambda: (state, empty_log_metrics)
        )
        
        log_metrics = {
            "train_loss": train_metrics.loss.mean(), 
            "train_accuracy": train_metrics.accuracy.mean(), 
            "val_loss": eval_metrics.validation_loss.mean(), 
            "val_accuracy": eval_metrics.validation_accuracy.mean(), 
            "adv_loss": eval_metrics.adv_loss.mean(), 
            "adv_accuracy": eval_metrics.adv_accuracy.mean(), 
            "epoch": state.epoch, 
            "opt_step": state.opt_step
        }
        if args.wandb: 
            def callback(metrics): 
                if metrics["epoch"] % args.eval_interval == 0: 
                    wandb.log(metrics)
            jax.debug.callback(callback, log_metrics)
        state = state._replace(epoch=state.epoch + 1)
        return (state, train_set, batched_validation_set), eval_metrics.validation_accuracy.mean()
    
    def train(carry): 
        (trained_state, train_set, batched_validation_set), mean_eval_accuracies = jax.lax.scan(
            train_epoch, 
            carry, 
            None, 
            length=epochs
        )
        return trained_state, mean_eval_accuracies
    
    return train 
        
carry = (state, train_set, batched_validation_set)
train = jax.jit(build_train_function(train_on_batch, eval_on_batch, pgd_linf))
if args.wandb: 
    wandb.init(project=wandb_project, config=args, name=wandb_name)
trained_state, mean_eval_accuracies = train(carry)

if args.wandb: 
    wandb.finish()







