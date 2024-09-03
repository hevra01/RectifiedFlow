"""Training Rectified Flow on Swissroll with DDPM++."""

from configs.default_toy_dataset import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.snapshot_freq = 50
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'rk45' ### rk45 or euler
  sampling.ode_tol = 1e-5

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.input_dim = 4 # 2D + 2D because the time embedding is 2D and the input is 2D
  model.output_dim = 2
  model.hidden_dim = 128
  model.name = 'Simple2DNetwork'
  model.scale_by_sigma = False
  model.ema_rate = 0.999999
  model.dropout = 0.15
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.conditional = True
  model.fir = False
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'

  return config
