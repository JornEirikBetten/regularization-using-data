import jax
import jax.numpy as jnp


from typing import NamedTuple

class DataBatch(NamedTuple):
    images: jnp.ndarray
    labels: jnp.ndarray

def build_fgsm_adversary(model, epsilon):
    def fgsm_adversary(x, y):
        x_adv = x + epsilon * jnp.sign(jnp.gradient(model(x), x))
        return x_adv
    return fgsm_adversary

def build_pgd_adversaries(loss_fn, epsilon, alpha, num_steps, batch_norm=False):
    def linf_pgd_adversary(carry, batch):
        variables, rng = carry
        rng, _rng = jax.random.split(rng)
        images, labels = batch.images, batch.labels
        # Set up a simple loss function for image input only, as the other inputs are constant 
        if batch_norm:
            image_loss_fn = lambda x: loss_fn(variables["params"], variables["batch_stats"], x, labels)
        else:
            image_loss_fn = lambda x: loss_fn(variables["params"], x, labels)
        # Randomly perturb the input to mitigate gradient masking 
        image_adv = images + jax.random.uniform(_rng, shape=images.shape, minval=-epsilon, maxval=epsilon)
        # Gradient ascent step
        def gradient_ascent(image, _):
            grad, aux = jax.grad(image_loss_fn, has_aux=True, allow_int=True)(image)
            image_adv = image + alpha * jnp.sign(grad)
            image_adv = jnp.clip(image_adv, image - epsilon, image + epsilon)
            return image_adv, 0 
        # Perform gradient ascent for num_steps steps
        image_adv, _ = jax.lax.scan(
            gradient_ascent, image_adv, None, num_steps
        )
        perturbed_batch = DataBatch(
            images=image_adv, 
            labels=batch.labels
        )
        return (variables, rng), perturbed_batch
    
    def l2_pgd_adversary(carry, batch):
        variables, rng = carry
        images, labels = batch.images, batch.labels
        # Set up a simple loss function for image input only, as the other inputs are constant 
        #forward_fn = lambda x: model.apply(variables, x, train=False)
        if batch_norm:
            image_loss_fn = lambda x: loss_fn(variables["params"], variables["batch_stats"], x, labels)
        else:
            image_loss_fn = lambda x: loss_fn(variables["params"], x, labels)
        # Randomly perturb the input to mitigate gradient masking 
        image_adv = images + jax.random.uniform(rng, shape=images.shape, minval=-epsilon, maxval=epsilon)
        # Gradient ascent step
        def gradient_ascent(images, _):    
            grad = jax.grad(image_loss_fn)(images)
            normalized_grad = grad / jnp.linalg.norm(grad)
            images_adv = images + alpha * normalized_grad
            images_adv = jnp.clip(images_adv, images - epsilon, images + epsilon)
            return images_adv, 0 
        # Perform gradient ascent for num_steps steps
        images_adv, _ = jax.lax.scan(
            gradient_ascent, images_adv, None, num_steps
        )
        perturbed_batch = DataBatch(
            images=images_adv, 
            labels=labels
        )
        return (variables, rng), perturbed_batch
    return linf_pgd_adversary, l2_pgd_adversary

# def build_pgd_adversaries_with_batch_stats(model, loss_fn, epsilon, alpha, num_steps):
#     def linf_pgd_adversary(batch, variables, batch_stats, rng):
#         image, target = batch.image, batch.target
#         # Set up a simple loss function for image input only, as the other inputs are constant 
#         forward_fn = lambda x: model.apply(variables, x, train=False, mutable=False)
#         image_loss_fn = lambda x: loss_fn(forward_fn(x), target)
#         # Randomly perturb the input to mitigate gradient masking 
#         image_adv = image + jax.random.uniform(rng, shape=image.shape, minval=-epsilon, maxval=epsilon)
#         # Gradient ascent step
#         def gradient_ascent(tup, _):
#             image, target = tup
#             loss, grad = jax.value_and_grad(image_loss_fn)(image)
#             image_adv = image + alpha * jnp.sign(grad)
#             image_adv = jnp.clip(image_adv, image - epsilon, image + epsilon)
#             return (image_adv, target), loss 
#         # Perform gradient ascent for num_steps steps
#         (image_adv, _), loss_values = jax.lax.scan(
#             gradient_ascent, (image_adv, target), None, num_steps
#         )
#         return image_adv, loss_values    
    
#     def l2_pgd_adversary(batch, params, rng):
#         image, target = batch.image, batch.target
#         # Set up a simple loss function for image input only, as the other inputs are constant 
#         forward_fn = lambda x: model.apply(params, x)
#         image_loss_fn = lambda x: loss_fn(forward_fn(x), target)
#         # Randomly perturb the input to mitigate gradient masking 
#         image_adv = image + jax.random.uniform(rng, shape=image.shape, minval=-epsilon, maxval=epsilon)
#         # Gradient ascent step
#         def gradient_ascent(tup, _):    
#             image, target = tup
#             loss, grad = jax.value_and_grad(image_loss_fn)(image)
#             normalized_grad = grad / jnp.linalg.norm(grad)
#             image_adv = image + alpha * normalized_grad
#             image_adv = jnp.clip(image_adv, image - epsilon, image + epsilon)
#             return (image_adv, target), loss 
#         # Perform gradient ascent for num_steps steps
#         (image_adv, _), loss_values = jax.lax.scan(
#             gradient_ascent, (image_adv, target), None, num_steps
#         )
#         return image_adv, loss_values
#     return linf_pgd_adversary, l2_pgd_adversary