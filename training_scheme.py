import jax 
import jax.numpy as jnp 


import training 
import models 
import data_handling 
import datasets 


import optax 
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
    batch_size: int = 128
    eval_interval: int = 10
    epochs: int = 101
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



args = TrainingConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

if args.dropout: 
    args.model = f"dropout{args.model}"

wandb_project = f"data-augmentation" 
wandb_name = f"{args.dataset}-{args.model}-lr={args.lr}"


# LOAD DATASET        
if args.dataset == "mnist": 
    dataset = datasets.MNISTDataJAX(data_points_per_class=42*128)
    num_classes = 10
elif args.dataset == "cifar10": 
    dataset = datasets.CIFAR10DataJAX(data_points_per_class=42*128)
    num_classes = 10
elif args.dataset == "cifar100": 
    dataset = datasets.CIFAR100DataJAX(data_points_per_class=42*128)
    num_classes = 100


train_set = dataset.train_set 
validation_set = dataset.validation_set
key = jax.random.PRNGKey(args.seed)
key, shuffle_key = jax.random.split(key)
batched_validation_set = data_handling.shuffle_and_batch_tree(shuffle_key, validation_set, args.batch_size)

model = models.RESNET_CONSTRUCTOR[args.model]
initial_conv_config = {"kernel_size": (3, 3), "strides": 1, "padding": "SAME"}
model_fn = model(num_classes=num_classes, num_filters=32, initial_conv_config=initial_conv_config)

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

def build_train_function(train_on_batch, eval_on_batch): 
    def evaluate(state, batched_validation_set): 
        state, eval_metrics = jax.lax.scan(
            eval_on_batch, 
            state, 
            batched_validation_set 
        )
        return state, eval_metrics 
    
    def train_epoch(carry, unused):     
        state, train_set, batched_validation_set = carry
        # Here you can add a data augmentation step 
        rng, rng_shuffle = jax.random.split(state.rng)
        batched_training_set = data_handling.shuffle_and_batch_tree(rng_shuffle, train_set, args.batch_size)
        state = state._replace(rng=rng)
        state, train_metrics = jax.lax.scan(
            train_on_batch, 
            state, 
            batched_training_set 
        )
        
        state, eval_metrics = jax.lax.cond(
            state.epoch % args.eval_interval == 0, 
            lambda: evaluate(state, batched_validation_set), 
            lambda: (state, training.EvalMetrics(loss=jnp.zeros((batched_validation_set.images.shape[0],)), accuracy=jnp.zeros((batched_validation_set.images.shape[0],))))
        )
        
        log_metrics = {
            "train_loss": train_metrics.loss.mean(), 
            "train_accuracy": train_metrics.accuracy.mean(), 
            "eval_loss": eval_metrics.loss.mean(), 
            "eval_accuracy": eval_metrics.accuracy.mean(), 
            "epoch": state.epoch, 
            "opt_step": state.opt_step
        }
        if args.wandb: 
            def callback(metrics): 
                if metrics["epoch"] % args.eval_interval == 0: 
                    wandb.log(metrics)
            jax.debug.callback(callback, log_metrics)
        state = state._replace(epoch=state.epoch + 1)
        return (state, train_set, batched_validation_set), eval_metrics.accuracy.mean()
    
    def train(carry): 
        (trained_state, train_set, batched_validation_set), mean_eval_accuracies = jax.lax.scan(
            train_epoch, 
            carry, 
            None, 
            length=args.epochs
        )
        return trained_state, mean_eval_accuracies
    
    return train 
        
carry = (state, train_set, batched_validation_set)
train = jax.jit(build_train_function(train_on_batch, eval_on_batch))
if args.wandb: 
    wandb.init(project=wandb_project, config=args, name=wandb_name)
trained_state, mean_eval_accuracies = train(carry)

if args.wandb: 
    wandb.finish()







