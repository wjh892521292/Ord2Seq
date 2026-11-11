# Ord2Seq: Regarding Ordinal Regression as Label Sequence Prediction

PyTorch research implementation of **Ord2Seq**, presented at ICCV 2023.

Jinhong Wang, Yi Cheng, Jintai Chen, TingTing Chen, Danny Chen, and Jian Wu

[CVF paper and supplement](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.html) · [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10376973) · [arXiv](https://arxiv.org/abs/2307.09004)

## Overview

Ordinal regression assigns inputs to ordered categories, such as age groups or disease severity grades. Adjacent categories can be difficult to distinguish because their visual features often overlap.

Ord2Seq encodes each ordinal label as a sequence of decisions inspired by binary search. For example, on a five-level scale, the model can first decide whether a score is above 3 and then refine its prediction within the remaining range. A Transformer decoder predicts the label sequence from image features. At each step, a masking strategy suppresses the contribution of categories ruled out by earlier decisions. The paper evaluates the framework with VGG and Pyramid Vision Transformer (PVT) backbones.

## Paper results

The paper evaluates Ord2Seq on age-group estimation, image aesthetics assessment, historical image dating, and diabetic retinopathy grading. Selected results below are from the paper. See Tables 2, 4, and 6 for the comparison protocols and baselines, and Table 5 for historical image dating. [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.pdf)

| Dataset | Backbone | Accuracy ↑ | MAE ↓ |
| --- | --- | ---: | ---: |
| Adience | VGG | 63.5% | 0.44 |
| Adience | PVT | 63.9% | 0.43 |
| Image Aesthetics, overall | VGG | 74.02% | 0.267 |
| Image Aesthetics, overall | PVT | 74.43% | 0.264 |
| Diabetic Retinopathy | VGG | 84.0 ± 0.6% | 0.25 ± 0.07 |
| Diabetic Retinopathy | PVT | 84.2 ± 0.5% | 0.25 ± 0.07 |

## Repository layout

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py) | Builds the training and validation loaders and starts the trainer |
| [`config/cfg.py`](config/cfg.py) | Defines experiment arguments and default settings |
| [`models/`](models/) | Contains backbone and sequence prediction code |
| [`training/trainer.py`](training/trainer.py) | Implements training, validation, logging, and checkpoint handling |
| [`evaluater/`](evaluater/) | Contains evaluation utilities |
| [`utils/`](utils/) | Contains supporting utilities |

## Environment

The original environment used **Python 3.8.13** and **PyTorch 1.12.1**. The trainer moves the model and batches to CUDA, so use a CUDA-capable GPU and install a CUDA build of PyTorch with the matching torchvision release (`0.13.1`). The entry point and trainer also import NumPy, OpenCV, scikit-learn, and TensorBoardX. [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1121)

```bash
git clone https://github.com/wjh892521292/Ord2Seq.git
cd Ord2Seq

# Install PyTorch 1.12.1 and torchvision 0.13.1 for your CUDA setup first.
python -m pip install numpy opencv-python scikit-learn tensorboardX
```

## Prepare data and configure an experiment

Obtain the images and annotations for the dataset you want to use. Match the split and label definitions in the paper if you are comparing against its results. Inspect the corresponding dataset loader for the annotation format, file names, and directory layout. The entry point looks up the loader by `data_name` and requests the `train` and `valid` splits. [Training entry point](main.py)

| Argument | What to set |
| --- | --- |
| `--data_name` | Name of the loader exposed by `utils.data_load` |
| `--img_root` | Local path to the images |
| `--data_root` | Local annotation or split path expected by the loader |
| `--fold` | Split index used by the loader |
| `--num_classes` | Number of ordered categories |
| `--model_name` | Model name exposed by `models` |
| `--gpu_id` | CUDA device index |
| `--save_folder` | Local checkpoint directory |
| `--save_log` | Local log directory |

## Training and validation

The public snapshot contains experiment-specific settings that need to be adapted before running it:

1. **Check the model imports.** [`models/__init__.py`](models/__init__.py) references modules such as `VggTrans` and `pvt_trans`, while the repository lists files named `vggtrans.py` and `pvttrans.py`. Check the import paths and exported classes for the model you intend to use. The active `vgg0528_new` preset in `main.py` also needs a corresponding export in `models`.
2. **Check the dataset loader.** Verify that the loader for your selected dataset is available through `utils.data_load` and accepts `img_root`, `data_root`, `dataset`, `transform`, and `fold`.
3. **Set local paths and experiment options.** The active `main.py` preset targets UTK with eight classes and passes a fixed argument list to `BaseConfig.initialize`. Edit that preset and supply local values for both data paths and both output paths. To configure the experiment from the command line, replace `cfg.initialize(fixed)` in the active entry point with `cfg.initialize()` and pass your settings explicitly.

Once the imports, loader, and local paths are in place, start the configured experiment from the repository root:

```bash
python main.py
```

The training loop runs validation using the selected `valid` split and writes checkpoints and TensorBoard logs to the configured directories. For checkpoint-based evaluation, consult `DefaultTrainer.load_model()` and `DefaultTrainer.validation()`. The call that loads `--resume` during trainer initialization is commented out in the public snapshot. [Trainer source](training/trainer.py)

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
