import jax 
import jax.numpy as jnp 
import flax.linen as nn  

from functools import partial
from typing import Any, Callable, Tuple, Sequence, Optional, Dict

# class ConvBlock(nn.Module): 
#     num_filters: int 
#     kernel_size: int 
#     stride: int 
#     padding: int 

#     @nn.compact 
#     def __call__(self, x): 
#         x = nn.Conv(self.num_filters, (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride), padding=self.padding)(x)
#         x = nn.relu(x)
        
# class ConvNet(nn.Module): 
#     num_classes: int 

#     @nn.compact 
#     def __call__(self, x): 
#         x = ConvBlock(32, 3, 1, 1)(x)
#         x = ConvBlock(64, 3, 1, 1)(x)
#         x = ConvBlock(128, 3, 1, 1)(x)
#         x = nn.avg_pool(x, (x.shape[1], x.shape[2]), strides=(x.shape[1], x.shape[2]))
#         x = x.reshape(x.shape[0], -1)
#         x = nn.Dense(self.num_classes)(x)
#         return x 



    

class ConvNet(nn.Module): 
    num_classes: int 
    num_filters: int 
    conv_config: dict 

    @nn.compact 
    def __call__(self, x, train=True): 
        conv = partial(nn.Conv, **self.conv_config)
        norm = partial(
            nn.BatchNorm, 
            use_running_average=not train, 
            momentum=0.9, 
            epsilon=1e-5
        )
        x = conv(self.num_filters)(x)
        x = norm()(x)
        x = nn.relu(x)
        x = conv(self.num_filters*4)(x)
        x = norm()(x)
        x = nn.relu(x)
        x = conv(self.num_filters*8)(x)
        x = norm()(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x 
    
    
ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
          residual
      )
      residual = self.norm(name="norm_proj")(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1), self.strides, name="conv_proj")(
          residual
      )
      residual = self.norm(name="norm_proj")(residual)

    return self.act(residual + y)


class DropoutResNetBlock(nn.Module): 
    """Dropout ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    dropout: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)
        y = self.dropout(rate=self.dropout_rate)(y)
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
        residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)
    
    
class ResNet(nn.Module):
  """ResNetV1."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  initial_conv_config: Optional[Dict[str, Any]] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )

    initial_conv_config = dict(self.initial_conv_config)
    initial_conv_config.setdefault("kernel_size", 7)
    initial_conv_config.setdefault("strides", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")
    x = conv(self.num_filters, **self.initial_conv_config)(x)
    x = norm(name="bn_init")(x)
    x = nn.relu(x)
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    features = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(features)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)

class DropoutResNet(nn.Module):
  """Dropout ResNet."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  initial_conv_config: Optional[Dict[str, Any]] = None
  dropout_rate: float = 0.5
    
    
  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )
    
    dropout = partial(
      nn.Dropout,
      deterministic=not train
    )

    initial_conv_config = dict(self.initial_conv_config)
    initial_conv_config.setdefault("kernel_size", 7)
    initial_conv_config.setdefault("strides", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    x = conv(self.num_filters, **self.initial_conv_config)(x)
    x = norm(name="bn_init")(x)
    x = nn.relu(x)
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
            dropout=dropout,
            dropout_rate=self.dropout_rate
        )(x)
    x = dropout(rate=self.dropout_rate)(x)
    features = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(features)
    x = jnp.asarray(x, self.dtype)
    return x
  
DropoutResNet1 = partial(DropoutResNet, stage_sizes=[1], block_cls=DropoutResNetBlock)
DropoutResNet18 = partial(DropoutResNet, stage_sizes=[2, 2, 2, 2], block_cls=DropoutResNetBlock)
DropoutResNet34 = partial(DropoutResNet, stage_sizes=[3, 4, 6, 3], block_cls=DropoutResNetBlock)
DropoutResNet50 = partial(DropoutResNet, stage_sizes=[3, 4, 6, 3], block_cls=DropoutResNetBlock)
DropoutResNet101 = partial(DropoutResNet, stage_sizes=[3, 4, 23, 3], block_cls=DropoutResNetBlock)
DropoutResNet152 = partial(DropoutResNet, stage_sizes=[3, 8, 36, 3], block_cls=DropoutResNetBlock)
DropoutResNet200 = partial(DropoutResNet, stage_sizes=[3, 24, 36, 3], block_cls=DropoutResNetBlock)



RESNET_CONSTRUCTOR = {
    "resnet1": ResNet1,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnet200": ResNet200,
    "dropoutresnet1": DropoutResNet1,
    "dropoutresnet18": DropoutResNet18,
    "dropoutresnet34": DropoutResNet34,
    "dropoutresnet50": DropoutResNet50,
    "dropoutresnet101": DropoutResNet101,
    "dropoutresnet152": DropoutResNet152,
    "dropoutresnet200": DropoutResNet200,
}