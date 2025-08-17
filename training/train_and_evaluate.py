import jax 
import jax.numpy as jnp 
import optax 
from typing import NamedTuple, Callable
import chex 

class Metrics(NamedTuple): 
    loss: chex.Array 
    accuracy: chex.Array 

class EvalMetrics(NamedTuple): 
    loss: chex.Array 
    accuracy: chex.Array 

def cross_entropy(logits, labels): 
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def accuracy_calculation(logits, labels): 
    return jnp.mean(jnp.argmax(logits, axis=1) == labels)



def build_loss_fn(model_fn, distance_metric, regularizer=None, batch_norm=False, dropout=False):
    
    if batch_norm and dropout:
        def train_loss_fn(params, batch_stats, images, labels, rng_dropout): 
            logits, batch_stats =  model_fn.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                train=True,
                mutable=["batch_stats"], 
                rngs={"dropout": rng_dropout}
            )
            loss = distance_metric(logits, labels)
            if regularizer is not None: 
                loss = loss + regularizer(params)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (batch_stats, accuracy)
        
        def eval_loss_fn(params, batch_stats, images, labels): 
            logits =  model_fn.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                train=False
            )
            loss = distance_metric(logits, labels)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
    elif batch_norm: 
        def train_loss_fn(params, batch_stats, images, labels): 
            logits, batch_stats =  model_fn.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                train=True,
                mutable=["batch_stats"]
            )
            loss = distance_metric(logits, labels)
            if regularizer is not None: 
                loss = loss + regularizer(params)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (batch_stats, accuracy)
        
        def eval_loss_fn(params, batch_stats, images, labels): 
            logits =  model_fn.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                train=False
            )
            loss = distance_metric(logits, labels)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
    elif dropout: 
        def train_loss_fn(params, images, labels, rng_dropout): 
            logits = model_fn.apply(params, images, train=True, rngs={"dropout": rng_dropout})
            loss = distance_metric(logits, labels)
            if regularizer is not None: 
                loss = loss + regularizer(params)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
        def eval_loss_fn(params, images, labels): 
            logits = model_fn.apply(params, images, train=False)
            loss = distance_metric(logits, labels)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
    else: 
        def train_loss_fn(params, images, labels): 
            logits = model_fn.apply(params, images)
            loss = distance_metric(logits, labels)
            if regularizer is not None: 
                loss = loss + regularizer(params)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
        def eval_loss_fn(params, images, labels): 
            logits = model_fn.apply(params, images)
            loss = distance_metric(logits, labels)
            accuracy = accuracy_calculation(logits, labels)
            return loss, (accuracy)
    return train_loss_fn, eval_loss_fn
    



def build_train_step(train_loss_fn, optimizer, batch_norm=False, dropout=False):
    if batch_norm and dropout: 
        def train_step(train_state, batch): 
            params = train_state.params 
            images, labels = batch.images, batch.labels
            rng, rng_dropout = jax.random.split(train_state.rng)
            simplified_loss = lambda params: train_loss_fn(params, train_state.batch_stats, images, labels, rng_dropout)
            (loss, (batch_stats, accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, train_state.opt_state)
            params = optax.apply_updates(params, updates)
            train_state = train_state._replace(
                params=params,
                batch_stats=batch_stats["batch_stats"],
                opt_state=opt_state,
                opt_step=train_state.opt_step + 1,
                rng=rng
            )
            metrics = Metrics(
                loss=loss, 
                accuracy=accuracy
            )
            return train_state, metrics
    elif batch_norm: 
        def train_step(train_state, batch): 
            params = train_state.params 
            images, labels = batch.images, batch.labels
            simplified_loss = lambda params: train_loss_fn(params, train_state.batch_stats, images, labels)
            (loss, (batch_stats, accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, train_state.opt_state)
            params = optax.apply_updates(params, updates)
            train_state = train_state._replace(
                params=params,
                batch_stats=batch_stats["batch_stats"],
                opt_state=opt_state,
                opt_step=train_state.opt_step + 1,
                rng=train_state.rng
            )
            metrics = Metrics(
                loss=loss, 
                accuracy=accuracy
            )
            return train_state, metrics
    elif dropout: 
        def train_step(train_state, batch): 
            params = train_state.params 
            images, labels = batch.images, batch.labels
            rng, rng_dropout = jax.random.split(train_state.rng)
            simplified_loss = lambda params: train_loss_fn(params, images, labels, rng_dropout)
            (loss, (accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, train_state.opt_state)
            params = optax.apply_updates(params, updates)
            train_state = train_state._replace(
                params=params,
                opt_state=opt_state,
                opt_step=train_state.opt_step + 1,       
                rng=rng
            )
            metrics = Metrics(
                loss=loss, 
                accuracy=accuracy
            )
            return train_state, metrics
    else: 
        def train_step(train_state, batch): 
            params = train_state.params 
            images, labels = batch.images, batch.labels
            simplified_loss = lambda params: train_loss_fn(params, images, labels)
            (loss, (accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, train_state.opt_state)
            params = optax.apply_updates(params, updates)
            train_state = train_state._replace(
                params=params,  
                opt_state=opt_state,
                opt_step=train_state.opt_step + 1,      
                rng=train_state.rng
            )
            metrics = Metrics(
                loss=loss, 
                accuracy=accuracy
            )
            return train_state, metrics
    return train_step


def build_eval_batch(eval_loss_fn, batch_norm=False): 
    if batch_norm: 
        def eval_batch(train_state, batch): 
            params, batch_stats = train_state.params, train_state.batch_stats
            images, labels = batch.images, batch.labels
            loss, accuracy = eval_loss_fn(params, batch_stats, images, labels)
            return train_state, EvalMetrics(loss=loss, accuracy=accuracy)
    else: 
        def eval_batch(train_state, batch): 
            params = train_state.params 
            images, labels = batch.images, batch.labels
            loss, accuracy = eval_loss_fn(params, images, labels)
            return train_state, EvalMetrics(loss=loss, accuracy=accuracy)
    return eval_batch

