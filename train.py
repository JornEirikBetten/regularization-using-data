import jax 
import jax.numpy as jnp 
import optax 
import chex 
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Literal, Any, NamedTuple
from training import build_update_step, build_loss_fn, build_forward_pass, cross_entropy, accuracy_calculation, build_evaluation_step, EvalMetrics
import models 
import datasets 
import data_handling 
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
    epochs: int = 100
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

wandb_project = f"regularize-using-data" 
wandb_name = f"{args.dataset}-{args.model}-lr={args.lr}"

if args.dataset == "mnist": 
    dataset = datasets.MNISTDataJAX(data_points_per_class=42*128)
elif args.dataset == "cifar10": 
    dataset = datasets.CIFAR10DataJAX(data_points_per_class=42*128)
elif args.dataset == "cifar100": 
    dataset = datasets.CIFAR100DataJAX(data_points_per_class=42*128)

# class EvalMetrics(NamedTuple): 
#     loss: chex.Array 
#     accuracy: chex.Array
#learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001]
#for lr in learning_rates: 
#    args.lr = lr 
#wandb_name = f"{args.dataset}_lr={args.lr}"

train_set = dataset.train_set 
validation_set = dataset.validation_set 
model = models.RESNET_CONSTRUCTOR[args.model]
initial_conv_config = {"kernel_size": (3, 3), "strides": 1, "padding": "SAME"}
model_fn = model(num_classes=10, num_filters=32, initial_conv_config=initial_conv_config)

variables = model_fn.init(jax.random.PRNGKey(args.seed), jnp.ones((1,) + train_set.images.shape[1:]))
if args.batch_norm: 
    params, batch_stats = variables["params"], variables["batch_stats"] 
    #batch_stats = model_fn.init_batch_stats(jax.random.PRNGKey(args.seed), jnp.ones((1,) + train_set.images.shape[1:]))
    variables = {"params": params, "batch_stats": batch_stats}
else: 
    params = variables

forward_pass = build_forward_pass(model_fn, dropout=args.dropout, batch_stats=args.batch_norm)
loss_fn = build_loss_fn(distance_metric=cross_entropy, regularizer=None, forward_pass=forward_pass, dropout=args.dropout, batch_stats=args.batch_norm)
optimizer = optax.adam(learning_rate=args.lr)

if args.batch_norm: 
    class TrainState(NamedTuple): 
        params: Any
        batch_stats: Any
        opt_state: Any 
        opt_step: Any
        epoch: Any
        rng: Any
        
    train_state_initializer = TrainState
    train_state = train_state_initializer(
        params=params,
        batch_stats=batch_stats,
        opt_state=optimizer.init(params),
        opt_step=0,
        epoch=0, 
        rng=jax.random.PRNGKey(args.seed)
    )
else: 
    class TrainState(NamedTuple): 
        params: Any
        opt_state: Any
        opt_step: Any
        epoch: Any
        rng: Any
    train_state_initializer = TrainState
    train_state = train_state_initializer(
        params=params,
        opt_state=optimizer.init(params),
        opt_step=0,
        epoch=0,
        rng=jax.random.PRNGKey(args.seed)
    )


update_step = build_update_step(loss_fn, optimizer, train_state_initializer, batch_norm=args.batch_norm, dropout=args.dropout)

eval_step = build_evaluation_step(forward_pass, batch_norm=args.batch_norm, dropout=args.dropout) 

def build_evaluate(eval_step): 
    def evaluate(train_state, batched_validation_set): 
        train_state, eval_metrics = jax.lax.scan(
            eval_step, 
            train_state, 
            batched_validation_set 
        )
        return train_state, eval_metrics 
    return evaluate 
evaluate = build_evaluate(eval_step)

def build_training_loop(update_step, train_state_initializer): 
    def training(carry, unused):
        train_state, train_set, batched_validation_set = carry 
        batched_training_set = data_handling.shuffle_and_batch_tree(train_state.rng, train_set, args.batch_size)
        train_state, metrics = jax.lax.scan(
            update_step, 
            train_state, 
            batched_training_set 
        )
        rng, _ = jax.random.split(train_state.rng)
        train_state = train_state_initializer(
            params=train_state.params,
            opt_state=train_state.opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch + 1,
            rng=rng
        )
        zeros_vector = jnp.zeros((batched_validation_set.images.shape[0],))
        train_state, eval_metrics = jax.lax.cond(
            train_state.epoch % args.eval_interval == 1, 
            lambda: evaluate(train_state, batched_validation_set), 
            lambda: (train_state, EvalMetrics(loss=zeros_vector, accuracy=zeros_vector))
        )
        log_metrics = {
            "train_loss": metrics.loss.mean(), 
            "train_accuracy": metrics.accuracy.mean(), 
            "eval_loss": eval_metrics.loss.mean(), 
            "eval_accuracy": eval_metrics.accuracy.mean(), 
            "epoch": train_state.epoch, 
            "opt_step": train_state.opt_step
        }
        if args.wandb: 
            def callback(metrics): 
                if metrics["epoch"] % args.eval_interval == 1: 
                    wandb.log(metrics)
            jax.debug.callback(callback, log_metrics)
        state = (train_state, train_set, batched_validation_set)
        return state, metrics.accuracy.mean()
    
    def training_with_batch_norm(carry, unused): 
        train_state, train_set, batched_validation_set = carry 
        batched_training_set = data_handling.shuffle_and_batch_tree(train_state.rng, train_set, args.batch_size)
        train_state, metrics = jax.lax.scan(
            update_step, 
            train_state, 
            batched_training_set 
        )
        rng, _ = jax.random.split(train_state.rng)
        train_state = train_state_initializer(
            params=train_state.params,
            batch_stats=train_state.batch_stats,
            opt_state=train_state.opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch + 1,
            rng=rng
        )
        zeros_vector = jnp.zeros((batched_validation_set.images.shape[0],))
        train_state, eval_metrics = jax.lax.cond(
            train_state.epoch % args.eval_interval == 1, 
            lambda: evaluate(train_state, batched_validation_set), 
            lambda: (train_state, EvalMetrics(loss=zeros_vector, accuracy=zeros_vector))
        )
        log_metrics = {
            "train_loss": metrics.loss.mean(), 
            "train_accuracy": metrics.accuracy.mean(), 
            "eval_loss": eval_metrics.loss.mean(), 
            "eval_accuracy": eval_metrics.accuracy.mean(), 
            "epoch": train_state.epoch, 
            "opt_step": train_state.opt_step
        }
        if args.wandb: 
            def callback(metrics): 
                if metrics["epoch"] % args.eval_interval == 1: 
                    wandb.log(metrics)
            jax.debug.callback(callback, log_metrics)
        state = (train_state, train_set, batched_validation_set)
        return state, metrics.accuracy.mean()
    if args.batch_norm: 
        return training_with_batch_norm 
    else: 
        return training 
        
        
batched_validation_set = data_handling.shuffle_and_batch_tree(train_state.rng, validation_set, args.batch_size)
training_loop = build_training_loop(update_step, train_state_initializer)

wandb.init(project=wandb_project, config=args, name=wandb_name)
out, accuracies_mean = jax.lax.scan(
    training_loop, 
    (train_state, train_set, batched_validation_set), 
    None, 
    length=args.epochs
)

wandb.finish()
# train_state, _, _ = out
# if args.save_model: 
#     save_path = f"trained_models/{args.dataset}/convnet/"
#     if not os.path.exists(save_path): 
#         os.makedirs(save_path)
#     save_path = f"{save_path}/{args.wandb_name}.pkl"
#     with open(save_path, "wb") as f: 
#         pickle.dump(train_state, f)
