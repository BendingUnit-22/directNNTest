# Instructions
This repo contains the code used for the evaluation in an ASE submission.

## Prerequisites
We first create a conda environment and then install some necessary packages:

`conda create -n test python=3.8`

`conda activate test`

`pip install -r requirements.txt`

## Ranking experiment (RQ1)
This is for RQ1 in the submission
Suppose we want to compare the two ranking methods: boosting diversity using the logit layer and the forward score ranking method:

`python main.py --experiment ranking --batch_size 500 --ranking_method1 margin_forward --ranking_method2 bd --adv_samples 14 --transformation adv --bd_layer final`

A possible output might look like:

> Avg l2 norm: 0.20034253264644317 Avg similarity score: 0.5837496669696597

By default, we use VGG network and CIFAR10 dataset. One can use `resnet9` and `resnet18` by specifying the argument for `--model` and dataset `svhn` and `cifar100` by specifying the argument for `--dataset`.

When first conduct the experiments, the script will check whether a model is trained. If it was not, the script will train a model and then save it. The next time the same command is used, the scrip will load the trained model.

One can compare other ranking methods by specifying arguments for `--ranking_method1` and `--ranking_method2`, the options are `margin_forward`, `ce_forward`, `margin_backward`,`ce_backward`,`ats`,`bd`,`dg`. Notice that when using `bd`, one can also specify `--bd_layer` with either `final` or `int` to decide which layer to use for the `bd` method.

One can specify what input transformation to use to measure the ranking similarity, with options `adv`, `benign`, `mixed` to flag `--transformation`, and tune how many adversarial examples to generate by tuning the argument `--adv_samples`.

## Testing with benign transformations (RQ2)

A sample command is:

`python main.py --experiment testing --test_batch_size 1000 --testing bd --transformation benign --iterations 5`

A possible output might look like:

> accracy after experiment: [89.56, 20.19, 10.77, 8.86, 7.22, 5.82]

The first accuracy is the clean accuracy, and the following are the accuracies after each iteration for five iterations.

One can change the testing method to `margin_forward`, `ce_forward`, `margin_backward`, `ce_backward`, `ats`, `bd`, `dg` as options to `testing`. Because the tests is adaptive, one can change the number of adaptive iterations by specifying the argument `--iterations`.

## Testing with adversarial transformations (RQ3)

A sample command is:

`python main.py --experiment testing --test_batch_size 1000 --testing bd --transformation adv --iterations 5 --adv_samples 70`

A possible output might look like:

> accracy after experiment: [89.56, 5.21, 0.09, 0.0, 0.0, 0.0]

The interpretation of results is similar ot RQ2's result.

One can change the transformation to `adv` as options to `--transformation` to use `adv` tests only. The number of adversarial examples can be specified with `--adv_samples`. `--testing` methods include `margin_forward`, `ce_forward`, `margin_backward`, `ce_backward`, `margin_mixed`, `ce_mixed`, `ats`, `bd`, `dg`.

## Metamorphic testing

Additionally, we also provide a metamorphic testing mode for our tools, by supplying the pseudo-label to the tool, rather than the ground-truth label. A sample command is:

`python main.py --experiment testing --test_batch_size 1000 --testing ce_forward --transformation benign --iterations 5 --test_mode metamorphic`

A possible output might look like:

> accracy after experiment: [100.0, 8.5, 1.19, 0.21, 0.06, 0.02]

Instead of feeding the ground-truth labels to the testing pipeline, we supply the pseudo-labels generated from the prediction of the model, and then the goal is to generate tests that will be predicted differently from the pseudo-label. Notice that the initial accuracy is `100` because the accuracy relative to the pseudo-label is 100% initially.
