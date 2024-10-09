# [NeurIPS 2024] Expanding Sparse Tuning to Low Memory Usage

**This is the official implementation of our paper:** Expanding Sparse Tuning to Low Memory Usage

### Introduction:

We propose a method called SNELL (**S**parse tuning with ker**NEL**ized **L**oRA) to enable sparse tuning with low memory usage. SNELL decomposes the tunable matrix for sparsification into two learnable low-rank matrices, saving from the costly storage of the original full matrix. To maintain the effectiveness of sparse tuning with low-rank matrices, we extend the low-rank decomposition from a kernel perspective. Specifically, we apply nonlinear kernel functions to the full-matrix merging and gain an increase in the rank of the merged matrix.  Employing higher ranks enhances the ability of SNELL to optimize the pre-trained model sparsely for downstream tasks. To further reduce the memory usage in sparse tuning, we introduce a competition-based sparsification mechanism, avoiding the storage of tunable weight indexes. Extensive experiments on multiple downstream tasks show that SNELL achieves state-of-the-art performance with low memory usage, extending effective PEFT with sparse tuning to large-scale models.

![framework](./main.png)

If you find this repository or our paper useful, please consider cite and star us!

```
@InProceedings{shen24neurips,
  title = {Expanding Sparse Tuning to Low Memory Usage},
  author = {Shen, Shufan and Sun, Junshu and Ji, Xiangyang and Huang, Qingming and Wang, Shuhui},
  booktitle = {Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year = {2024}
}
```

------

## Getting started on SNELL:

### Install dependency:

We have tested our code on both Torch 1.8.0, and 1.10.0. Please install the other dependencies with the following code in the home directory:

```
pip install -r requirements.txt
```

#### Data preparation:

Please download the FGVC dataset following [VPT](https://github.com/KMnP/vpt) and put the dataset under `./data/fgvc`.

#### Download pre-trained models:

Please download the backbones with the following code:

```
mkdir checkpoints
cd checkpoints

# Supervised pre-trained ViT-B/16
wget https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz

# MAE pre-trained ViT-B/16
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# MoCo V3 pre-trained ViT-B/16
wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar
```

#### PEFT with SNELL:

We have provided the following training code on FGVC, for example:

```bash
# Fine-tuning supervised pre-trained ViT-B/16 with SNELL-32 for CUB dataset
bash configs/fgvc/snell32/vit_cub_snell.sh
```
### Acknowledgements:
Our code is modified from [SPT](https://github.com/ziplab/SPT). We thank the authors for their open-sourced code.
