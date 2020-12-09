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
import utils.COLTR as COLTR


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
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")

args = parser.parse_args()

click_model_name = args.click_model
cutoff = args.cutoff

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
model.build(input_shape=data.train.feature_matrix.shape)
if args.pretrained_model:
  model.load_weights(args.pretrained_model)

candidate_models = COLTR.prepare_candidate_models(model, 50)

n_sampled = 0

n_eval = 50
eval_points = np.logspace(
                          2, 5, n_eval+2,
                          endpoint=True,
                          dtype=np.int32)

results = []
output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': 'COLTR',
  'number of evaluation points': n_eval,
  'evaluation iterations': [int(x) for x in eval_points],
  'model hyperparameters': model_params,
  'results': results,
}
if args.pretrained_model:
  output['initial model'] = args.pretrained_model

start_time = time.time()

n_train_queries = data.train.num_queries()
n_vali_queries = data.validation.num_queries()
train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)
for i in range(np.amax(eval_points)):
  train_or_vali = np.random.choice(2, p=[1.-train_ratio, train_ratio])

  if train_or_vali:
    data_split = data.train
    qid = np.random.choice(n_train_queries)
    doc_weights = true_train_doc_weights
  else:
    data_split = data.validation
    qid = np.random.choice(n_vali_queries)
    doc_weights = true_vali_doc_weights

  (ranking, clicks, scores) = clkgn.single_ranking_generation(
                    qid,
                    data_split,
                    doc_weights,
                    alpha,
                    beta,
                    model=model,
                    return_scores=True,)

  COLTR.update(model, candidate_models,
              qid, data_split,
              ranking, clicks, scores,
              learning_rate=0.05,
              unit=1.,
              var_weight=1.0)
  
  n_queries_sampled = i + 1
  
  if n_queries_sampled in eval_points:
    cur_metrics = evl.evaluate_policy(
                        model,
                        data.test,
                        true_test_doc_weights,
                        alpha,
                        beta,
                      )

    cur_results = {
      'iteration': int(n_queries_sampled),
      'metrics': cur_metrics,
      'logging policy metrics': cur_metrics,
      'evaluate': n_queries_sampled in eval_points,
      'update': True,
    }
    results.append(cur_results)

    print('Seconds per iterations: %f' % ((time.time() - start_time)/n_queries_sampled))

    print('No. query %09d, NRCTR %0.5f, NDCG %0.5f' % (
        n_queries_sampled, cur_metrics['NRCTR'], cur_metrics['NDCG']))

print(output)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)

