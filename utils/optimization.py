# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf
import time

import utils.plackettluce as pl

def optimize_policy(model, optimizer,
                    data_train, train_doc_weights, train_alpha, train_beta,
                    data_vali,  vali_doc_weights, vali_alpha, vali_beta,
                    n_grad_samples = 100,
                    n_eval_samples = 100,
                    max_epochs = 50,
                    early_stop_diff = 0.001,
                    early_stop_per_epochs = 3,
                    print_updates=False):

  early_stop_per_epochs = min(early_stop_per_epochs, max_epochs)

  stacked_alphas = np.stack([train_alpha, vali_alpha], axis=-1)
  stacked_betas = np.stack([train_beta, vali_beta], axis=-1)

  cutoff = stacked_alphas.shape[0]

  policy_vali_scores = model(data_vali.feature_matrix)[:,0].numpy()
  metrics =  pl.datasplit_metrics(
                  data_vali,
                  policy_vali_scores,
                  stacked_alphas,
                  stacked_betas,
                  vali_doc_weights,
                  n_samples=n_eval_samples,
                )
  if print_updates:
    print('epoch %d: train %0.04f vali %0.04f' % (0, metrics[0], metrics[1]))
  first_metric_value = metrics[1]
  last_metric_value = metrics[1]

  best_metric_value = metrics[1]
  best_weights = model.get_weights()

  cum_doc_weights = np.cumsum(np.abs(train_doc_weights))
  start_weights = cum_doc_weights[data_train.doclist_ranges[:-1]]
  end_weights = cum_doc_weights[data_train.doclist_ranges[1:]-1]
  qid_included = np.where(np.not_equal(start_weights, end_weights))[0]
  qid_included = np.random.permutation(qid_included)

  start_time = time.time()
  n_queries = qid_included.shape[0]
  for i in range(n_queries*max_epochs):
    qid = qid_included[i%n_queries]

    q_doc_weights = data_train.query_values_from_vector(qid, train_doc_weights)
    q_feat = data_train.query_feat(qid)
    q_cutoff = min(cutoff, data_train.query_size(qid))

    # print(q_doc_weights)

    with tf.GradientTape() as tape:

      tf_scores = model(q_feat)[:, 0]
      scores = tf_scores.numpy()

      sampled_rankings = pl.gumbel_sample_rankings(
                               scores,
                               n_grad_samples,
                               cutoff=q_cutoff,
                              )[0]
      gradient = pl.fast_gradient_based_on_samples(
                        sampled_rankings,
                        train_alpha,
                        q_doc_weights,
                        scores,
                        cutoff=q_cutoff)

      # hybrid_gradient = pl.hybrid_gradient_based_on_samples(
      #                   sampled_rankings,
      #                   train_alpha,
      #                   q_doc_weights,
      #                   scores,
      #                   cutoff=q_cutoff)

      loss = -tf.reduce_sum(tf_scores * gradient)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # reshuffle queries every epoch
    if (i+1) % (n_queries) == 0:
      qid_included = np.random.permutation(qid_included)
    if (i+1) % (n_queries*early_stop_per_epochs) == 0:
      epoch_i = (i+1) / n_queries
      policy_vali_scores = model(data_vali.feature_matrix)[:,0].numpy()
      metrics =  pl.datasplit_metrics(
                      data_vali,
                      policy_vali_scores,
                      stacked_alphas,
                      stacked_betas,
                      vali_doc_weights,
                      n_samples=n_eval_samples,
                    )
      abs_improvement = metrics[1] - last_metric_value
      if print_updates:
        improvement = metrics[1]/last_metric_value - 1.
        total_improvement = metrics[1]/first_metric_value - 1.
        average_time = (time.time() - start_time)/(i+1.)*n_queries
        print('epoch %d: '
              'train %0.04f '
              'vali %0.04f '
              'epoch-time %0.04f '
              'abs-improvement %0.05f '
              'improvement %0.05f '
              'total-improvement %0.05f ' % (epoch_i, 
                 metrics[0], metrics[1],
                 average_time,
                 abs_improvement, improvement, total_improvement)
              )
      last_metric_value = metrics[1]
      if best_metric_value < metrics[1]:
        best_weights = model.get_weights()
      if abs_improvement < early_stop_diff:
        break

  model.set_weights(best_weights)
  return model, last_metric_value