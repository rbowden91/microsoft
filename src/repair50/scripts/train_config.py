from ..my_env import typing
from ..my_env.typing import NamedTuple

class TrainConfig(NamedTuple):
    # learning_rate : float
    # max_epoch : int
    # max_max_epoch : int
    # lr_decay : float
    init_scale : float
    max_grad_norm : float
    num_layers : int
    hidden_size : int
    epochs : int
    drop_prob : float
    batch_size : int

SmallConfig = TrainConfig(
  #"learning_rate" : 1.0,
  #"max_epoch" : 4,
  #"max_max_epoch" : 4,
  #"lr_decay" : 0.5,
  init_scale = 0.1,
  max_grad_norm = 5,
  num_layers = 2,
  hidden_size = 400,
  epochs = 6,
  drop_prob = 1.0,
  batch_size = 20
)

MediumConfig = TrainConfig(
  init_scale = 0.05,
  max_grad_norm = 5,
  num_layers = 2,
  hidden_size = 650,
  epochs = 39,
  drop_prob = 0.5,
  batch_size = 20,
)

LargeConfig = TrainConfig(
  init_scale = 0.04,
  max_grad_norm = 10,
  num_layers = 2,
  hidden_size = 1500,
  epochs = 55,
  drop_prob = 0.35,
  batch_size = 20,
)

TestConfig = TrainConfig(
  init_scale = 0.1,
  max_grad_norm = 1,
  num_layers = 1,
  hidden_size = 4,
  epochs = 1,
  drop_prob = 1.0,
  batch_size = 20,
)
