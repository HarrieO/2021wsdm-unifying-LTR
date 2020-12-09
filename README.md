# Unifying Online and Counterfactual Learning to Rank
This repository contains the code used for the experiments in "Unifying Online and Counterfactual Learning to Rank" published at WSDM 2021 ([preprint available](https://arxiv.org/abs/2012.04426)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our WSDM 2021 paper:

```
@inproceedings{oosterhuis2021onlinecounterltr,
  Author = {Oosterhuis, Harrie and de Rijke, Maarten},
  Booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining (WSDM'21)},
  Organization = {ACM},
  Title = {Unifying Online and Counterfactual Learning to Rank},
  Year = {2021}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/) and the [numpy](https://numpy.org/) package, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
To start, the (pretrained) logging model can be optimized using the following command, this will perform supervised training based on 20 training queries: *--n_train_queries 20* and 5 validation queries: *--n_vali_queries 5* on the Yahoo! dataset: *--dataset Webscope_C14_Set1*.
The resulting model is stored in *local_output/pretrained_model.h5*.
```
python3 pretrained_run.py local_output/pretrained_model.h5 --n_train_queries 20 --n_vali_queries 5 --dataset Webscope_C14_Set1
```
The following command runs with the intervention-aware estimator, counterfactually since no interventions are performed: *--n_updates 0*, the logging model *--pretrained_model local_output/pretrained_model.h5* is used to gather clicks:
```
python3 interventionaware_run.py local_output/interventionaware.txt --n_updates 0 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1
```
The run will simulate 100,000,000 queries, and the results are stored in *local_output/interventionaware.txt*.
To run the intervention aware estimator (semi-)online, we simply change the number of updates. For instance, the following command will cause 100 interventions to take place:
```
python3 interventionaware_run.py local_output/interventionaware_100interventions.txt --n_updates 100 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1
```
Each run performed for the paper has its own file in the top folder and uses similar arguments.