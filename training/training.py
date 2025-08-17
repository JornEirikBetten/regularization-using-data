import jax 
import jax.numpy as jnp 
import optax 
from typing import NamedTuple
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

def build_forward_pass(model, dropout=False, batch_stats=False): 
    def forward_pass(params, images): 
        logits = model.apply(params, images)
        return logits
    
    def forward_pass_with_batch_norm(params, batch_stats, images): 
        logits, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=True,
            mutable=["batch_stats"]
        )
        return logits, batch_stats
    
    def forward_pass_with_dropout(params, images, rng_dropout): 
        logits = model.apply(
            params, 
            images, 
            train=True,
            rngs={"dropout": rng_dropout}
        )
        return logits
    
    def forward_pass_with_batch_norm_and_dropout(params, batch_stats, images, rng_dropout): 
        logits, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=True,
            rngs={"dropout": rng_dropout},
            mutable=["batch_stats"]
        )
        return logits, batch_stats
    
    if dropout and batch_stats: 
        return forward_pass_with_batch_norm_and_dropout
    elif dropout: 
        return forward_pass_with_dropout
    elif batch_stats: 
        return forward_pass_with_batch_norm
    else: 
        return forward_pass    
    
def build_loss_fn(distance_metric, regularizer, forward_pass, dropout=False, batch_stats=False): 
    def loss_fn(params, images, labels): 
        logits = forward_pass(params, images)
        loss = distance_metric(logits, labels)
        if regularizer is not None: 
            loss = loss + regularizer(params)
        acc = accuracy_calculation(logits, labels)
        return loss, (acc)
    
    def loss_fn_with_batch_norm(params, batch_stats, images, labels): 
        logits, batch_stats = forward_pass(params, batch_stats, images)
        loss = distance_metric(logits, labels)
        if regularizer is not None: 
            loss = loss + regularizer(params)
        acc = accuracy_calculation(logits, labels)
        return loss, (batch_stats, acc)
    
    
    def loss_fn_with_dropout(params, images, labels, rng_dropout): 
        logits = forward_pass(params, images, rng_dropout)
        loss = distance_metric(logits, labels)
        if regularizer is not None: 
            loss = loss + regularizer(params)
        acc = accuracy_calculation(logits, labels)
        return loss, (acc)
    
    def loss_fn_with_batch_norm_and_dropout(params, batch_stats, images, labels, rng_dropout): 
        logits, batch_stats = forward_pass(params, batch_stats, images, rng_dropout)
        loss = distance_metric(logits, labels)
        if regularizer is not None: 
            loss = loss + regularizer(params)
        accuracy = accuracy_calculation(logits, labels)
        return loss, (batch_stats, accuracy)
    
    if dropout and batch_stats: 
        return loss_fn_with_batch_norm_and_dropout
    elif dropout: 
        return loss_fn_with_dropout
    elif batch_stats: 
        return loss_fn_with_batch_norm
    else: 
        return loss_fn
    
def build_update_step(loss_fn, optimizer, train_state_initializer, batch_norm=False, dropout=False): 
    def update_step(train_state, batch): 
        params = train_state.params 
        images, labels = batch.images, batch.labels
        simplified_loss = lambda params: loss_fn(params, images, labels)
        (loss, (accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(params, updates)
        train_state = train_state_initializer(
            params=params,
            opt_state=opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch, 
            rng=train_state.rng
        )
        metrics = Metrics(
            loss=loss, 
            accuracy=accuracy
        )
        return train_state, metrics
    
    def update_step_with_batch_norm(train_state, batch): 
        params = train_state.params 
        images, labels = batch.images, batch.labels
        simplified_loss = lambda params: loss_fn(params, train_state.batch_stats, images, labels)
        (loss, (batch_stats, accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(params, updates)
        train_state = train_state_initializer(
            params=params,
            batch_stats=batch_stats["batch_stats"],
            opt_state=opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch, 
            rng=train_state.rng
        )
        metrics = Metrics(
            loss=loss, 
            accuracy=accuracy
        )
        return train_state, metrics
    
    def update_step_with_dropout(train_state, batch): 
        params = train_state.params 
        images, labels = batch.images, batch.labels
        rng, rng_dropout = jax.random.split(train_state.rng)
        simplified_loss = lambda params: loss_fn(params, images, labels, rng_dropout)
        (loss, (accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(params, updates)
        train_state = train_state_initializer(
            params=params,
            opt_state=opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch, 
            rng=rng
        )
        metrics = Metrics(
            loss=loss, 
            accuracy=accuracy
        )
        return train_state, metrics
    
    def update_step_with_batch_norm_and_dropout(train_state, batch): 
        params = train_state.params 
        images, labels = batch.images, batch.labels
        rng, rng_dropout = jax.random.split(train_state.rng)
        simplified_loss = lambda params: loss_fn(params, train_state.batch_stats, images, labels, rng_dropout)
        (loss, (batch_stats, accuracy)), grads = jax.value_and_grad(simplified_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(params, updates)
        train_state = train_state_initializer(
            params=params,
            batch_stats=batch_stats["batch_stats"],
            opt_state=opt_state,
            opt_step=train_state.opt_step + 1,
            epoch=train_state.epoch, 
            rng=rng
        )
        metrics = Metrics(
            loss=loss, 
            accuracy=accuracy
        )
        return train_state, metrics
    
    if batch_norm and dropout: 
        return update_step_with_batch_norm_and_dropout
    elif batch_norm: 
        return update_step_with_batch_norm
    elif dropout: 
        return update_step_with_dropout
    else: 
        return update_step



def build_evaluation_step(forward_pass, batch_norm=False, dropout=False): 
    def evaluation_step(train_state, batch): 
        params = train_state.params 
        images, labels = batch.images, batch.labels
        logits = forward_pass(params, images)
        loss = cross_entropy(logits, labels)
        accuracy = accuracy_calculation(logits, labels)
        return train_state, EvalMetrics(loss=loss, accuracy=accuracy)
    
    def evaluation_step_with_batch_norm(train_state, batch): 
        params, batch_stats = train_state.params, train_state.batch_stats
        images, labels = batch.images, batch.labels
        logits, batch_stats = forward_pass(params, batch_stats, images)
        loss = cross_entropy(logits, labels)
        accuracy = accuracy_calculation(logits, labels)
        return train_state, EvalMetrics(loss=loss, accuracy=accuracy)
    
    
    if batch_norm: 
        return evaluation_step_with_batch_norm
    else: 
        return evaluation_step
    
    
