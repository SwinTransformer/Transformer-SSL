# MoBY with Swin Transformer, Self-Supervised Pre-training and ImageNet-1K Linear Evaluation

This folder contains the implementation of the `MoBY` with `Swin Transformer` for image classification.

## Model Zoo

### ImageNet-1K Linear Evaluation Results

|      Method      | Architecture | Epochs | Params | FLOPs | img/s | Top-1 Accuracy |                                                                                            Checkpoint                                                                                            |
| :--------------: | :----------: | :----: | :----: | :---: | :---: | :------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Supervised    |    Swin-T    |  300   |  28M   | 4.5G  | 755.2 |      81.2      |                                              [Here](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models)                                               |
|       MoBY       |    Swin-T    |  100   |  28M   | 4.5G  | 755.2 |      70.9      |                                                                                             [TBA]()                                                                                              |
| MoBY<sup>1</sup> |    Swin-T    |  100   |  28M   | 4.5G  | 755.2 |      72.0      |                                                                                             [TBA]()                                                                                              |
|       MoBY       |    DeiT-S    |  300   |  22M   | 4.6G  | 940.4 |      72.8      |                                                                                             [TBA]()                                                                                              |
|       MoBY       |    Swin-T    |  300   |  28M   | 4.5G  | 755.2 |      75.0      | [Pre-trained](https://drive.google.com/file/d/1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u/view?usp=sharing) / [Linear](https://drive.google.com/file/d/1gbQynZy07uXPO-c0tOLeyG1pQzlnVHx9/view?usp=sharing) |

- <sup>1</sup> denotes the result of MoBY which has adopted a trick from MoCo v3 that replace theLayerNorm layers before the MLP blocks by BatchNorm.

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/Swin-Transformer/Transformer-SSL
cd Transformer-SSL
```

- Create a conda virtual environment and activate it:

```bash
conda create -n transformer-ssl python=3.7 -y
conda activate transformer-ssl
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 diffdist
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet 
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

### Self-Supervised Pre-training

To train `MoBY` with `Swin Transformer` on ImageNet, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 moby_main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

- Recommand using `--output` and `--tag` to tidy your experiments.

**Notes**:

- To use zipped ImageNet instead of folder dataset, add `--zip` to the parameters.
    - To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will
      shard the dataset into non-overlapping pieces for different GPUs and only load the corresponding one for each GPU.
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.
    - Use gradient checkpointing by adding `--use-checkpoint`, e.g., it saves about 60% memory when training `Swin-B`.
      Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
    - We recommend using multi-node with more GPUs for training very large models, a tutorial can be found
      in [this page](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
- To change config options in general, you can use `--opts KEY1 VALUE1 KEY2 VALUE2`, e.g.,
  `--opts TRAIN.EPOCHS 100 TRAIN.WARMUP_EPOCHS 5` will change total epochs to 100 and warm-up epochs to 5.
- For additional options, see [config](config.py) and run `python moby_main.py --help` to get detailed message.

For example, to train `MoBY` with `Swin Transformer` with 8 GPU on a single node for 300 epochs, run:

`MoBY Swin-T`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  moby_main.py \
--cfg configs/moby_swin_tiny.yaml --data-path <imagenet-path> --batch-size 64
```

### Linear Evaluation

To evaluate a pre-trained `MoBY` with `Swin Transformer` on ImageNet-1K linear evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 moby_linear.py \
--cfg <config-file> --data-path <imagenet-path>
```
**Notes**:

- Make sure the `<config-file>`, `<output-directory>` and `<tag>` are the same as in the pre-training stage.
- Note that some configurations are fixed in [`moby_linear.py`](moby_linear.py#L78) for simplicity.

For example, to evaluate `MoBY Swin-T` with 8 GPU on a single node on ImageNet-1K linear evluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  moby_linear.py \
--cfg configs/moby_swin_tiny.yaml --data-path <imagenet-path> --batch-size 64
```

### Evaluate

To evaluate a `MoBY` with `Swin Transformer` linear evaluation model on ImageNet-1K, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 moby_linear.py \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> --eval
```

For example, to evaluate the provided `MoBY Swin-T` linear evaluation model with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 moby_linear.py \
--cfg configs/moby_swin_tiny.yaml --resume moby_swin_t_300ep_linear.pth --data-path <imagenet-path> --eval
```
