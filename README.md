# Towards Visual-Prompt Temporal Answering Grounding in Instructional Video


<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/WENGSYX/VPTSL.svg?color=blue&style=flat-square">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/WENGSYX/VPTSL">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/WENGSYX/VPTSL">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/WENGSYX/VPTSL">
</p>



**Authors**: Shutao Li, Bin Li, Bin Sun, Yixuan Weng😎

**[Contact]** If you have any questions, feel free to contact me via ([Email](mailto:libincn@hnu.edu.cn?subject=VPTSL)).

This repository contains code, models, and other related resources of our paper ["Towards Visual-Prompt Temporal Answering Grounding in Instructional Video"](https://www.techrxiv.org/doi/full/10.36227/techrxiv.22182736.v1).

* [2024/02/28] We released the dataset! 
* [2023/03/02] We have published the paper!
* [2023/02/14] We released the code!
  
****

**Welcome to the VPTSL Project - We design the text span-based predictor, where the input text question, video subtitles, and visual prompt features are jointly learned with the pre-trained language model for enhancing the joint semantic representations.**🚀🚅



## Prerequisites
- python 3.7 with pytorch (`1.10.0`), transformers(`4.15.0`), tqdm, accelerate, pandas, numpy, glob, sentencepiece
- cuda10/cuda11

#### Installing the GPU driver

```shell script
# preparing environment
sudo apt-get install gcc
sudo apt-get install make
wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
sudo sh cuda_11.5.1_495.29.05_linux.run
```

#### Installing Conda and Python

```shell script
# preparing environment
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n VPTSL python==3.7
conda activate VPTSL
```

#### Installing Python Libraries

```plain
# preparing environment
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm transformers sklearn pandas numpy glob accelerate sentencepiece
```

### Download Data

##### Download the MedVidQA dataset from [here](https://pan.baidu.com/s/1lcBUb8JYWUVaZcRZq3RT5Q?pwd=8888 
)  [Copyright belongs to [NIH](https://arxiv.org/abs/2201.12888)]

Download the **TutorialVQA** dataset from [here ](mailto:libincn@hnu.edu.cn?subject=Get%20TutorialVQA%20Dataset) (Researchers should get in touch with us and sign a copyright agreement)

Download the **VehicleVQA** dataset from [here](mailto:libincn@hnu.edu.cn?subject=Get%20VehicleVQA%20Dataset) (Researchers should get in touch with us and sign a copyright agreement)

place them in `./data` directory.

## Quick Start
### Get Best

```shell script
bash run.sh
```
> All our hyperparameters are saved to  `run.sh` file, you can easily reproduce our best results.

### Try it yourself

##### Make model

```shell script
python set_model.py --shape large
```
> Our Text Encoder uses the [DeBERTa]([microsoft/deberta-v3-base · Hugging Face](https://huggingface.co/microsoft/deberta-v3-base)) model (To support longer text), the other layers are initialized randomly. 
>
> You can choose  a (xsmall/small/base/large) model for train.

##### Training and testing

```shell script
python main.py --shape large \
	--seed 42 \
	--maxlen 1800 \
	--epochs 32 \
	--batchsize 4 \
	--lr 1e-5 \
	--highlight_hyperparameter 0.25 \
	--loss_hyperparameter 0.1
```

> In this phase, training and testing will be carried out.
>
> In addition, after each round of training, it will be tested in the valid and test sets. In our paper, we report the model with the highest valid set and its score in the test set
