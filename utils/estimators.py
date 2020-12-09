# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def prob_per_rank_query(n_samples, cutoff, policy_scores):
  n_docs = policy_scores.size
  q_cutoff = min(cutoff, policy_scores.size)
  if n_docs <= 1:
    return np.ones((n_docs, q_cutoff))
  rankings = pl.gumbel_sample_rankings(
                      policy_scores,
                      n_samples,
                      q_cutoff)[0]
  freq_per_rank = np.zeros((n_docs, q_cutoff), dtype=np.int64)
  np.add.at(freq_per_rank, (rankings[:,:-1], np.arange(q_cutoff-1)[None,:]), 1)
  prob_per_rank = freq_per_rank.astype(np.float64)/n_samples

  scores_per_ranking = np.tile(policy_scores, (n_samples, 1)).astype(np.float64)
  scores_per_ranking[np.arange(n_samples)[:, None], rankings[:,:-1]] = np.NINF
  scores_per_ranking -= np.amax(scores_per_ranking, axis=1)[:, None]
  denom = np.log(np.sum(np.exp(scores_per_ranking), axis=1))[:, None]

  prob_per_rank[:, -1] = np.mean(np.exp(scores_per_ranking - denom), axis=0)

  return prob_per_rank

def expected_alpha_beta(
                    data_split,
                    n_samples,
                    alpha,
                    beta,
                    model=None,
                    all_policy_scores=None):
  if all_policy_scores is None:
    all_policy_scores = model(data_split.feature_matrix)[:,0].numpy()

  n_docs = data_split.num_docs()
  expected_alpha = np.zeros(n_docs, dtype=np.float64)
  expected_beta = np.zeros(n_docs, dtype=np.float64)
  for qid in range(data_split.num_queries()):
    cutoff = min(alpha.shape[0],
                 data_split.query_size(qid))

    q_alpha = data_split.query_values_from_vector(
                                  qid, expected_alpha)
    q_beta = data_split.query_values_from_vector(
                                  qid, expected_beta)
    policy_scores = data_split.query_values_from_vector(
                                  qid, all_policy_scores)

    q_prob_per_rank = prob_per_rank_query(n_samples,
                                          cutoff,
                                          policy_scores)

    q_alpha[:] = np.sum(q_prob_per_rank*alpha[None, :cutoff], axis=1)
    q_beta[:] = np.sum(q_prob_per_rank*beta[None, :cutoff], axis=1)

    assert np.all(np.greater(q_alpha, 0)), 'Zero alpha: %s' % q_alpha

  return expected_alpha, expected_beta

def compute_weights(data_split,
                    alpha_per_doc,
                    beta_per_doc,
                    clicks,
                    displays,
                    normalize=False,
                    n_queries_sampled=None,
                    alpha_clip=None,
                    beta_clip=None):
  if alpha_clip is None:
    clipped_alpha = alpha_per_doc
  else:
    clipped_alpha = np.maximum(alpha_per_doc, alpha_clip)
  if beta_clip is None:
    clipped_beta = beta_per_doc
  else:
    clipped_beta = np.maximum(beta_per_doc, beta_clip)
    
  weights = (clicks-displays*clipped_beta)/clipped_alpha

  if len(weights.shape) == 2:
    weights = np.sum(weights, axis=0)

  if normalize:
    weights /= float(n_queries_sampled/data_split.num_queries())

  return weights

def update_weights(data_split,
                   prev_clicks,
                   prev_displays,
                   prev_query_freq,
                   prev_exp_alpha,
                   prev_exp_beta,
                   new_clicks,
                   new_displays,
                   new_query_freq,
                   new_exp_alpha,
                   new_exp_beta,
                   n_samples,
                   alpha,
                   beta,
                   model=None,
                   all_policy_scores=None,
                   alpha_clip=None,
                   beta_clip=None):

  prev_n_sampled_queries = np.sum(prev_query_freq)
  new_n_sampled_queries = np.sum(new_query_freq)
  total_queries = prev_n_sampled_queries + new_n_sampled_queries
  prev_weight = prev_n_sampled_queries / total_queries
  new_weight = new_n_sampled_queries / total_queries

  prev_clicks += new_clicks
  prev_displays += new_displays
  prev_query_freq += new_query_freq

  prev_exp_alpha *= prev_weight
  prev_exp_alpha += new_weight*new_exp_alpha
  prev_exp_beta *= prev_weight
  prev_exp_beta += new_weight*new_exp_beta

  doc_weights = compute_weights(
                    data_split,
                    prev_exp_alpha,
                    prev_exp_beta,
                    prev_clicks,
                    prev_displays,
                    normalize=True,
                    n_queries_sampled=np.sum(prev_query_freq),
                    alpha_clip=alpha_clip,
                    beta_clip=beta_clip,
                  )

  return doc_weights
  
def update_weights_policy_aware(
                  i_update,
                  data_split,
                  prev_clicks,
                  prev_displays,
                  prev_query_freq,
                  prev_exp_alpha,
                  prev_exp_beta,
                  new_clicks,
                  new_displays,
                  new_query_freq,
                  n_samples,
                  alpha,
                  beta,
                  model=None,
                  all_policy_scores=None,
                  alpha_clip=None,
                  beta_clip=None):

  prev_clicks[i_update,:] += new_clicks
  prev_displays[i_update,:] += new_displays
  prev_query_freq += new_query_freq

  doc_weights = compute_weights(
                    data_split,
                    prev_exp_alpha,
                    prev_exp_beta,
                    prev_clicks,
                    prev_displays,
                    normalize=True,
                    n_queries_sampled=np.sum(prev_query_freq),
                    alpha_clip=alpha_clip,
                    beta_clip=beta_clip,
                  )

  return doc_weights

def update_weights_affine(
                  data_split,
                  prev_clicks,
                  prev_displays,
                  prev_query_freq,
                  new_clicks,
                  new_displays,
                  new_query_freq,
                  alpha,
                  beta,
                  model=None,
                  all_policy_scores=None,
                  alpha_clip=None,
                  beta_clip=None):

  prev_clicks += new_clicks
  prev_displays += new_displays
  prev_query_freq += new_query_freq

  n_queries_sampled = np.sum(prev_query_freq)

  if alpha_clip is None:
    clipped_alpha = alpha
  else:
    clipped_alpha = np.maximum(alpha, alpha_clip)
  if beta_clip is None:
    clipped_beta = beta
  else:
    clipped_beta = np.maximum(beta, beta_clip)
  
  weights = (prev_clicks-prev_displays*clipped_beta[None, :])/clipped_alpha[None, :]
  if len(weights.shape) == 2:
    weights = np.sum(weights, axis=1)

  weights /= float(n_queries_sampled/data_split.num_queries())

  return weights