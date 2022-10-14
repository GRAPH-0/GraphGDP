CUDA_VISIBLE_DEVICES=2 python main.py --config configs/vp_com_small_pgsn.py --mode train --workdir exp/vp_com_small_default

#CUDA_VISIBLE_DEVICES=3 python main.py --config configs/vp_com_small_pgsn.py --config.model.beta_max 5.0 --mode train --workdir exp/vp_com_small_beta5

#CUDA_VISIBLE_DEVICES=2 python main.py --config configs/vp_com_small_pgsn.py --mode eval --config.eval.begin_ckpt 20 --config.eval.end_ckpt 20 --workdir exp/vp_com_small_default