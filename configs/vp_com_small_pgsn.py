"""Training PGSN on Community Small Dataset with GraphGDP"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.model_type = 'sde'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vpsde'
    training.continuous = True
    training.reduce_mean = True

    training.batch_size = 32
    training.n_iters = 100000
    training.snapshot_freq = 5000
    training.log_freq = 200
    training.eval_freq = 1000
    ## store additional checkpoints for preemption
    training.snapshot_freq_for_preemption = 2500
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.rtol = 1e-5
    sampling.atol = 1e-5

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 5
    evaluate.end_ckpt = 20
    evaluate.batch_size = 256
    evaluate.enable_sampling = True
    evaluate.num_samples = 1024
    evaluate.enable_loss = False
    evaluate.enable_bpe = False
    evaluate.bpe_dataset = 'test'
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = 'data'
    data.name = 'Community_small'
    data.split_ratio = 0.8
    data.max_node = 20
    data.num_graphs = 100
    data.num_channels = 1

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'PGSN'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.num_gnn_layers = 4
    # model.conditional = True
    model.size_cond = False
    model.embedding_type = 'positional'
    model.rw_depth = 16
    model.graph_layer = 'PosTransLayer'
    model.edge_th = -1.
    model.heads = 8

    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
