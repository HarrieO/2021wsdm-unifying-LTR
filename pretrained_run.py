# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import pickle
import random
import tensorflow as tf

import utils.dataset as dataset
import utils.clicks as clk
import utils.nnmodel as nn
import utils.optimization as opt
import utils.evaluation as evl


parser = argparse.ArgumentParser()
# parser.add_argument("model_file", type=str,
#                     help="Model file output from pretrained model.")
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='default')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--n_train_queries", type=int,
                    help="Number of randomly selected training queries used for training.",
                    default=20)
parser.add_argument("--n_vali_queries", type=int,
                    help="Number of randomly selected training queries used for early stopping.",
                    default=5)

args = parser.parse_args()

click_model_name = args.click_model
cutoff = args.cutoff
n_train_queries = args.n_train_queries
n_vali_queries = args.n_vali_queries

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = False,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

max_ranking_size = np.min((cutoff, data.max_query_size()))

click_model = clk.get_click_model(click_model_name)

alpha, beta = click_model(np.arange(max_ranking_size))

model_params = {'hidden units': [32, 32],}
model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.Adam()

train_selection = np.random.choice(data.train.num_queries(),
                                  replace=False,
                                  size=n_train_queries)
vali_selection = np.random.choice(data.validation.num_queries(),
                                  replace=False,
                                  size=n_vali_queries)

selected_train_rewards = np.zeros_like(data.train.label_vector, dtype=np.float64)
selected_vali_rewards = np.zeros_like(data.validation.label_vector, dtype=np.float64)
for qid in train_selection:
  q_rewards = data.train.query_values_from_vector(
                                  qid, selected_train_rewards)
  q_rewards[:] = data.train.query_values_from_vector(
                                  qid, data.train.label_vector)*0.25
for qid in vali_selection:
  q_rewards = data.validation.query_values_from_vector(
                                  qid, selected_vali_rewards)
  q_rewards[:] = data.validation.query_values_from_vector(
                                  qid, data.validation.label_vector)*0.25
selected_train_rewards *= data.train.num_queries()/float(n_train_queries)
selected_vali_rewards *= data.validation.num_queries()/float(n_vali_queries)

first_results = evl.evaluate_policy(
                      model,
                      data.test,
                      data.test.label_vector*0.25,
                      alpha,
                      beta,
                    )

print(first_results)

print('Optimizing')
model, vali_reward = opt.optimize_policy(model, optimizer,
                    data_train=data.train,
                    train_doc_weights=selected_train_rewards,
                    train_alpha=alpha+beta,
                    train_beta=np.zeros_like(beta),
                    data_vali=data.train,
                    vali_doc_weights=selected_train_rewards,
                    vali_alpha=alpha+beta,
                    vali_beta=np.zeros_like(beta),
                    early_stop_per_epochs = 5,
                    early_stop_diff = 0.0,
                    max_epochs=1000,
                    print_updates=True,
                    )

final_results = evl.evaluate_policy(
                      model,
                      data.test,
                      data.test.label_vector*0.25,
                      alpha,
                      beta,
                    )

print(final_results)

model.save(args.output_path)
