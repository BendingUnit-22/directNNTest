# Directions
This repo contains the code used for the evaluation in an ASE submission.

## Prerequisites
We first create a conda environment and then install some necessary packages:

`conda create --n test python=3.8`

`pip install -r requirements.txt`

## Ranking experiment (RQ1)
This is for RQ1 in the submission
Suppose we want to compare the two ranking methods: boosting diversity using the logit layer and the forward score ranking method:

`python3 main.py --experiment ranking --batch_size 500 --ranking_method1 margin_forward --ranking_method2 bd --adv_samples 14 --transformation adv --bd_layer final`

By default, we use VGG network and CIFAR10 dataset. One can use `resnet9` and `resnet18` by specifying the argument for `--model` and dataset `svhn` and `cifar100` by specifying the argument for `--dataset`.

When first conduct the experiments, the script will check whether a model is trained. If it was not, the script will train a model and then save it. The next time the same command is used, the scrip will load the trained model.

One can compare other ranking methods by specifying arguments for `--ranking_method1` and `--ranking_method2`, the options are `margin_forward`, `ce_forward`, `margin_backward`,`ce_backward`,`ats`,`bd`,`dg`. Notice that when using `bd`, one can also specify `--bd_layer` with either `final` or `int` to decide which layer to use for the `bd` method.

## Testing with benign transformations (RQ2)

## Testing with adversarial transformations (RQ3)