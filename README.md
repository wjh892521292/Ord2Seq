# Ord2Seq: Regarding Ordinal Regression as Label Sequence Prediction

PyTorch implementation of **Ord2Seq**, presented at ICCV 2023.

Jinhong Wang, Yi Cheng, Jintai Chen, TingTing Chen, Danny Chen, and Jian Wu

[CVF paper and supplement](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.html) · [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10376973) · [arXiv](https://arxiv.org/abs/2307.09004)

## Overview

Ordinal regression assigns inputs to ordered categories, such as age groups or disease severity grades. Adjacent categories can be difficult to distinguish because their visual features often overlap.

Ord2Seq represents each ordinal label as a sequence of decisions inspired by binary search. On a five-level scale, for example, the model can first decide whether the score is above 3 and then refine its prediction within the remaining range. A Transformer decoder predicts the sequence from image features, while a masking strategy reduces the influence of categories ruled out at earlier steps. The paper evaluates the framework with VGG and Pyramid Vision Transformer (PVT) backbones.

## Paper results

The paper evaluates Ord2Seq on age-group estimation, image aesthetics assessment, historical image dating, and diabetic retinopathy grading. The table shows selected results reported in the paper. See Tables 2, 4, and 6 for the corresponding comparisons and Table 5 for historical image dating. [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.pdf)

| Dataset | Backbone | Accuracy ↑ | MAE ↓ |
| --- | --- | ---: | ---: |
| Adience | VGG | 63.5% | 0.44 |
| Adience | PVT | 63.9% | 0.43 |
| Image Aesthetics, overall | VGG | 74.02% | 0.267 |
| Image Aesthetics, overall | PVT | 74.43% | 0.264 |
| Diabetic Retinopathy | VGG | 84.0 ± 0.6% | 0.25 ± 0.07 |
| Diabetic Retinopathy | PVT | 84.2 ± 0.5% | 0.25 ± 0.07 |

## Environment

The original environment used **Python 3.8.13** and **PyTorch 1.12.1**. Training requires an NVIDIA GPU with CUDA support. The example below installs the CUDA 11.3 build of PyTorch and the matching torchvision version.

```bash
git clone https://github.com/wjh892521292/Ord2Seq.git
cd Ord2Seq

conda create -n ord2seq python=3.8.13 pip -y
conda activate ord2seq

conda install -y -c pytorch pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3

python -m pip install numpy opencv-python scikit-learn tensorboardX
```

## Data preparation

Obtain the images and annotations for your dataset. Use the annotation format and directory layout expected by the corresponding loader in `utils.data_load`. The training entry point selects the loader using `--data_name` and requests the `train` and `valid` splits. When comparing against paper results, follow the split and label definitions in the paper for that dataset. [Training entry point](main.py) · [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.pdf)

## Training and validation

The experiment block in `main.py` is configured for UTK with eight ordered classes. Before running it, edit the argument list in `main.py` to set `--img_root` and `--data_root` to your local paths. For another dataset, update `--data_name`, `--fold`, and `--num_classes` to match its loader and split.

Set `ORD2SEQ_MODEL` to a model name exported by `models/__init__.py`, then run the entry point from the repository root. Replace `YOUR_MODEL_NAME` with the selected export.

```bash
ORD2SEQ_MODEL=YOUR_MODEL_NAME python main.py
```

If model imports fail, check that the module names and capitalization in `models/__init__.py` match the files available in your checkout. [Model exports](models/__init__.py)

The trainer runs validation on the `valid` split during training. The experiment block configures checkpoint and log directories under `outputs/checkpoints/` and `outputs/logs/`. For checkpoint-based evaluation, see `DefaultTrainer.load_model()` and `DefaultTrainer.validation()`. The checkpoint-loading call for `--resume` is commented out in the trainer. [Trainer source](training/trainer.py)

## Citation

```bibtex
@inproceedings{wang2023ord2seq,
  title     = {Ord2Seq: Regarding Ordinal Regression as Label Sequence Prediction},
  author    = {Wang, Jinhong and Cheng, Yi and Chen, Jintai and Chen, TingTing
               and Chen, Danny and Wu, Jian},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {5865--5875}
}
```
