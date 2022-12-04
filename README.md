# GraphGDP: Generative Diffusion Processes for Permutation Invariant Graph Generation

Official Code Repository for GraphGDP (ICDM 2022).

## Dependencies 

The main requirements are:
* pytorch 1.11
* PyG 2.1
* DGL 0.9.1 (for GIN-based metrics from GGM-metrics)

Others see requirements.txt .

## Code Usage

### Training Example
1. Community small dataset
```shell
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode train --workdir YOUR_PATH
```

2. Ego small dataset



### Evaluation Example
* EM method sampling 
```shell
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode eval --workdir YOUR_PATH \
--config.eval.begin_ckpt 150 --config.eval.end_ckpt 150
```

* Langevin correction
```shell
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode eval --workdir YOUR_PATH \
--config.eval.begin_ckpt 150 --config.eval.end_ckpt 150 --config.sampling.corrector langevin --config.sampling.snr 0.20
```

* ODE Solvers
```shell
# scipy ODE (CPU)
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode eval --workdir YOUR_PATH \
--config.eval.begin_ckpt 150 --config.eval.end_ckpt 150 --config.sampling.method ode \
--config.sampling.rtol 1e-4 --config.sampling.atol 1e-4

# Neural ODE (GPU) - Adaptive-step
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode eval --workdir YOUR_PATH \
--config.eval.begin_ckpt 150 --config.eval.end_ckpt 150 --config.sampling.method diffeq \
--config.sampling.ode_method dopri5 --config.sampling.rtol 1e-4 --config.sampling.atol 1e-4

# Neural ODE (GPU) - Fixed-step
python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode eval --workdir YOUR_PATH \
--config.eval.begin_ckpt 150 --config.eval.end_ckpt 150 --config.sampling.method diffeq \
--config.sampling.ode_method rk4 --config.sampling.ode_step 0.10
```

*Note*: we recommend training with config.model.beta_max 20.0 when utilizing probability flow ODEs.

Some models are provided on [Google Drive](https://drive.google.com/drive/folders/103eZR1JsPOXsJztP-RdXUHnoZqvOAOqh?usp=sharing).

## Citation

\



*Acknowledgement:* Our implementation is based on the repo [Score_SDE](https://github.com/yang-song/score_sde_pytorch). 
Evaluation implementation is modified from the repo [GGM-metrics](https://github.com/uoguelph-mlrg/GGM-metrics).

