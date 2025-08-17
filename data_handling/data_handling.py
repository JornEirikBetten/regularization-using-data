import jax 
import jax.numpy as jnp 
import chex 
import tensorflow as tf 

from typing import NamedTuple 

"""
        DATA TYPES 
"""

class DataBatch(NamedTuple): 
    images: chex.Array 
    labels: chex.Array 

class PerturbedDataBatch(NamedTuple):
    images: chex.Array 
    labels: chex.Array 
    gradient_norm: chex.Array
    gradient: chex.Array
    signed_gradient: chex.Array
    epsilon: chex.Array
    
class FeatureData(NamedTuple):
    mean_class_features: chex.Array 
    class_centered_features: chex.Array 
    
class TrainingMetrics(NamedTuple): 
    loss: chex.Array 
    accuracy: chex.Array 
    features: chex.Array 
    labels: chex.Array 
    logits: chex.Array
    gradients: chex.Array
    sum_of_squared_weights: chex.Array
    Lp: chex.Array
    XE: chex.Array
    
class EvalMetrics(NamedTuple): 
    loss: chex.Array 
    accuracy: chex.Array 
    logits: chex.Array
    features: chex.Array
    labels: chex.Array

"""
        REPRESENTATION LEARNING METRICS 
"""
   
class RepresentationLearningMetrics(NamedTuple): 
    loss: chex.Array  
    features: chex.Array 
    weight_decay_loss: chex.Array

class RepresentationLearningEvalMetrics(NamedTuple): 
    loss: chex.Array 
    features: chex.Array 
    weight_decay_loss: chex.Array
    pred_images: chex.Array
    
"""
        DATA HANDLING 
"""

def extract_batch(data: DataBatch, indices: chex.Array) -> DataBatch: 
    return DataBatch(images=data.images[indices, :, :, :], labels=data.labels[indices])

def shuffle_and_batch_tree(rng: jax.random.PRNGKey, data: DataBatch, batch_size: int) -> DataBatch: 
    indices = jax.random.choice(rng, jnp.arange(data.images.shape[0]), (data.images.shape[0],), replace=False)
    num_batches = data.images.shape[0] // batch_size
    def slice_indices(count, _):
        sliced_indices = jax.lax.dynamic_slice(
            operand=indices, 
            start_indices=[count], 
            slice_sizes=(batch_size,)
        )   
        count += batch_size
        batch = extract_batch(data, sliced_indices)
        return count, batch
    count = 0 
    count, batches = jax.lax.scan(
        f = slice_indices,
        init = count,
        xs = None, 
        length = num_batches
    )
    return batches 

def shuffle_and_batch_with_adversial_perturbation(rng: jax.random.PRNGKey, params: dict, batch_stats: dict, data: DataBatch, batch_size: int, fgsm: callable) -> DataBatch:
    """
    Shuffles the data and batches it
    """
    images = data.images.reshape((-1, data.images.shape[-2], data.images.shape[-1], 1))
    perturbed_images = fgsm(params, batch_stats, data.images, data.labels)
    perturbed_data = DataBatch(
        images=perturbed_images,
        labels=data.labels
    )
    return shuffle_and_batch_tree(rng, perturbed_data, batch_size)

def make_fgsm_batch(fgsm: callable) -> callable:
    def fgsm_batch(state: tuple, batch: DataBatch) -> DataBatch:
        """
        Applies the FGSM perturbation to a batch of data
        """
        params, batch_stats, _ = state
        perturbed_images, gradient_norm, gradient, signed_gradient, epsilon = fgsm(params, batch_stats, batch.images, batch.labels)
        data_batch = PerturbedDataBatch(
            images=perturbed_images,
            labels=batch.labels, 
            gradient_norm=gradient_norm, 
            gradient=gradient, 
            signed_gradient=signed_gradient, 
            epsilon=epsilon
        )
        return state, data_batch
    return fgsm_batch

def make_fgsm_batch_no_batch_stats(fgsm: callable) -> callable:
    def fgsm_batch_no_batch_stats(state: tuple, batch: DataBatch) -> DataBatch:
        params, _ = state
        perturbed_images = fgsm(params, batch.images, batch.labels)
        data_batch = DataBatch(
            images=perturbed_images,
            labels=batch.labels
        )
        return state, data_batch
    return fgsm_batch_no_batch_stats
