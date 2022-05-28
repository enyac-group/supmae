## Fine-tuning Pre-trained SupMAE for Classification

### Evaluation

Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet):
```
python main_finetune.py --eval --resume supmae_vitb_ft_100e.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 83.572 Acc@5 96.574 loss 0.733
```

### Fine-tuning

Get our pre-trained checkpoints from [here](https://drive.google.com/file/d/1YwcTJvASZJvn2LxyZZG4PcgXaBCKEv_4/view?usp=sharing).

To fine-tune our pre-trained ViT-Base with **single-node training**, run the following on 1 node with 8 GPUs:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --batch_size 32 \
    --accum_iter 4 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`accum_iter`) * 8 (gpus) = 1024. `--accum_iter 4` simulates 4 nodes.

#### Notes

- We basically follow the hyper parameters of [MAE](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) with a bit larger learning rate.

- The [pre-trained models we provide](https://drive.google.com/file/d/1YwcTJvASZJvn2LxyZZG4PcgXaBCKEv_4/view?usp=sharing) is trained for 400 epochs, Table 1 in paper.

### Linear Probing
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 2 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

#### Notes

- SupMAE is pre-trained with global pooling feature, instead of the <tt>class</tt> token (see Table 2(b) in the paper). However, linear probing is using the <tt>class</tt> token.


