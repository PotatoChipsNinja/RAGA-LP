# RAGA-LP
Using [RAGA](https://arxiv.org/abs/2103.00791) encoder for link prediction task

## Datasets
- WN18RR
- FB15k-237

## Environment
- Python 3.8
- PyTorch 1.8.1
- CUDA 10.2
- PyTorch Geometric 1.7.0

## Running
### GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py --data WN18RR --emb_dim 100
```

### CPU
```
python train.py --cuda false --data WN18RR --emb_dim 100
```
