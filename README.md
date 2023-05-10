# directNNTest

## Prerequisite
We first create a conda environment and then install some necessary packages:
`conda create --n test python=3.8`

`pip install -r requirements.txt`

## Ranking experiment
This is for RQ1 in the submission
Suppose we want to compare the two ranking methods: boosting diversity using the logit layer and the forward score ranking method

`python3 main.py --experiment ranking --batch_size 500 --ranking_method1 margin_forward --ranking_method2 bd --adv_samples 14 --transformation adv --bd_layer final`

By default, we use VGG network and CIFAR10 dataset. One can use `resnet9` and `resnet18` by specifying the argument for `--model` and dataset `svhn` and `cifar100` by specifying the argument for `--dataset`.