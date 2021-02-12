# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import json
import numpy as np
import tensorflow as tf
import time

import utils.click_generation as clkgn
import utils.clicks as clk
import utils.dataset as dataset
import utils.estimators as est
import utils.evaluation as evl
import utils.nnmodel as nn
import utils.optimization as opt


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
parser.add_argument("--n_updates", type=int,
                    help="Number of updates during run.",
                    default=0)
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")

args = parser.parse_args()

click_model_name = args.click_model
cutoff = args.cutoff
n_updates = args.n_updates

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

true_train_doc_weights = data.train.label_vector*0.25
true_vali_doc_weights = data.validation.label_vector*0.25
true_test_doc_weights = data.test.label_vector*0.25

model_params = {'hidden units': [32, 32],}
model = nn.init_model(model_params)
logging_model = nn.init_model(model_params)
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.build(input_shape=data.train.feature_matrix.shape)
logging_model.build(input_shape=data.train.feature_matrix.shape)
if args.pretrained_model:
  model.load_weights(args.pretrained_model)
  logging_model.load_weights(args.pretrained_model)
init_weights = model.get_weights()
init_opt_weights = optimizer.get_weights()

n_sampled = 0

train_clicks = np.zeros((data.train.num_docs(), max_ranking_size), dtype=np.int64)
train_displays = np.zeros((data.train.num_docs(), max_ranking_size), dtype=np.int64)
train_query_freq = np.zeros(data.train.num_queries(), dtype=np.int64)
train_policy_scores = logging_model(data.train.feature_matrix)[:, 0].numpy()

vali_clicks = np.zeros((data.validation.num_docs(), max_ranking_size), dtype=np.int64)
vali_displays = np.zeros((data.validation.num_docs(), max_ranking_size), dtype=np.int64)
vali_query_freq = np.zeros(data.validation.num_queries(), dtype=np.int64)
vali_policy_scores = logging_model(data.validation.feature_matrix)[:, 0].numpy()

n_eval = 50
# n_eval = 5
update_points = np.logspace(
                          2, 8, n_updates,
                          endpoint=False,
                          dtype=np.int32)
update_points = list(np.unique(update_points))

eval_points = np.logspace(
                          2, 8, n_eval+2,
                          endpoint=True,
                          dtype=np.int32)

add_every = (n_eval+1) / float(n_updates+1)
next_to_add = add_every
result = []
for i in range(1, n_eval+1):
  if i >= next_to_add:
    result.append(i)
    next_to_add += add_every
result = np.array(result, dtype=np.int64)
update_points = eval_points[result]

results = []
output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': 'affine estimator',
  'number of updates': n_updates,
  'number of evaluation points': n_eval,
  'update iterations': [int(x) for x in update_points],
  'evaluation iterations': [int(x) for x in eval_points],
  'model hyperparameters': model_params,
  'results': results,
}

if args.pretrained_model:
  output['initial model'] = args.pretrained_model

logging_policy_metrics = evl.evaluate_policy(
                                model,
                                data.test,
                                true_test_doc_weights,
                                alpha,
                                beta,
                              )

train_policy_scores[:] = logging_model(data.train.feature_matrix)[:, 0].numpy()
vali_policy_scores[:] = logging_model(data.validation.feature_matrix)[:, 0].numpy()

i_update = 0
for n_queries_sampled in eval_points:

  (new_train_clicks, new_train_displays,
   new_train_query_freq,
   new_vali_clicks, new_vali_displays,
   new_vali_query_freq,) = clkgn.simulate_on_dataset(
                           data.train,
                           data.validation,
                           n_queries_sampled - n_sampled,
                           true_train_doc_weights,
                           true_vali_doc_weights,
                           alpha,
                           beta,
                           model=logging_model,
                           train_policy_scores=train_policy_scores,
                           vali_policy_scores=vali_policy_scores,
                           return_display=True,
                           store_per_rank=True,)

  alpha_clip = min(10./np.sqrt(n_queries_sampled), 1.)
  beta_clip = 0.
  train_doc_weights = est.update_weights_affine(
                            data.train,
                            train_clicks,
                            train_displays,
                            train_query_freq,
                            new_train_clicks,
                            new_train_displays,
                            new_train_query_freq,
                            alpha,
                            beta,
                            model=logging_model,
                            all_policy_scores=train_policy_scores,
                            alpha_clip=alpha_clip,
                            beta_clip=beta_clip)
  vali_doc_weights = est.update_weights_affine(
                            data.validation,
                            vali_clicks,
                            vali_displays,
                            vali_query_freq,
                            new_vali_clicks,
                            new_vali_displays,
                            new_vali_query_freq,
                            alpha,
                            beta,
                            model=logging_model,
                            all_policy_scores=vali_policy_scores)

  model.set_weights(init_weights)
  # optimizer = tf.keras.optimizers.Adam()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  model, vali_reward = opt.optimize_policy(model, optimizer,
                      data_train=data.train,
                      train_doc_weights=train_doc_weights,
                      train_alpha=alpha+beta,
                      train_beta=np.zeros_like(beta),
                      data_vali=data.validation,
                      vali_doc_weights=vali_doc_weights,
                      vali_alpha=alpha+beta,
                      vali_beta=np.zeros_like(beta),
                      # max_epochs=1,
                      early_stop_per_epochs=1,
                      print_updates=False,
                      )

  cur_metrics = evl.evaluate_policy(
                        model,
                        data.test,
                        true_test_doc_weights,
                        alpha,
                        beta,
                      )
  if n_queries_sampled in update_points:
    logging_model.set_weights(model.get_weights())
    logging_policy_metrics = cur_metrics
    train_policy_scores[:] = logging_model(data.train.feature_matrix)[:, 0].numpy()
    vali_policy_scores[:] = logging_model(data.validation.feature_matrix)[:, 0].numpy()

  cur_results = {
    'iteration': int(n_queries_sampled),
    'metrics': cur_metrics,
    'logging policy metrics': logging_policy_metrics,
    'evaluate': n_queries_sampled in eval_points,
    'update': n_queries_sampled in update_points,
  }
  results.append(cur_results)

  if n_queries_sampled in update_points:
    print('No. query %09d, NRCTR %0.5f, NDCG %0.5f, UPDATE' % (
        n_queries_sampled, cur_metrics['NRCTR'], cur_metrics['NDCG']))
  else:
    print('No. query %09d, NRCTR %0.5f, NDCG %0.5f' % (
        n_queries_sampled, cur_metrics['NRCTR'], cur_metrics['NDCG']))

  n_sampled = n_queries_sampled

print(output)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
