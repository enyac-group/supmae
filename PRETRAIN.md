## Pre-training SupMAE

To pre-train ViT-Base (recommended default) with **single-node training**, run the following on 1 node with 8 GPUs:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_pretrain_supmae.py \
      --global_pool  \
      --batch_size 256 \
      --model mae_vit_base_patch16_d1b \
      --norm_pix_loss  \
      --mask_ratio 0.75 \
      --epochs 400  \
      --warmup_epochs 20 \
      --blr 1.5e-4 \
      --weight_decay 0.05 \
      --data_path ${IMAGENET_DIR}
```
- Because we have BatchNorm in classification head, use `--accum_iter` may be unsafe because BN requires batch statistics.

- Caveat: We DO NOT test ViT-Large or ViT-Huge model.
