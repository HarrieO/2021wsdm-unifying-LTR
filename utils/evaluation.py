# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def evaluate_policy(model,
                    data_split,
                    doc_weights,
                    click_alpha,
                    click_beta,
                    n_samples = 100):
  cutoff = click_alpha.shape[0]
  dcg_weights = 1./np.log2(np.arange(cutoff)+2.)
  stacked_alphas = np.stack([click_alpha,
                             click_alpha+click_beta,
                             dcg_weights,
                            ], axis=-1)
  stacked_betas = np.stack([click_beta,
                            np.zeros_like(click_beta),
                            np.zeros_like(click_beta),
                          ], axis=-1)

  norm_factors = max_score_per_query(data_split,
                                     doc_weights,
                                     stacked_alphas,
                                     stacked_betas)

  # repeat alphas to calculate both normalized and unnormalized
  stacked_alphas = stacked_alphas[:,[0,0,1,1,2,2]]
  stacked_betas = stacked_betas[:,[0,0,1,1,2,2]]
  norm_factors = np.stack([
                           norm_factors[:, 0],
                           np.ones_like(norm_factors[:, 0]),
                           norm_factors[:, 1],
                           np.ones_like(norm_factors[:, 0]),
                           norm_factors[:, 2],
                           np.ones_like(norm_factors[:, 0]),
                          ], axis=-1)
  policy_scores = model(data_split.feature_matrix)[:,0].numpy()
  metrics =  pl.datasplit_metrics(
                  data_split,
                  policy_scores,
                  stacked_alphas,
                  stacked_betas,
                  doc_weights,
                  query_norm_factors=norm_factors,
                  n_samples=n_samples,
                )
  result = {
      'NCTR':  metrics[0],
      'CTR':   metrics[1],
      'NRCTR': metrics[2],
      'RCTR':  metrics[3],
      'NDCG':  metrics[4],
      'DCG':   metrics[5],
    }
  for k, v in result.items():
    result[k] = float(v)
  return result

def max_score_per_query(data_split, weight_per_doc,
                        weight_per_rank, addition_per_rank):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)

  sort_i = np.argsort(-weight_per_rank, axis=0)
  sorted_weights = weight_per_rank[
          sort_i,
          np.arange(weight_per_rank.shape[1])[None, :],
        ]
  sorted_additions = addition_per_rank[
          sort_i,
          np.arange(weight_per_rank.shape[1])[None, :],
        ]

  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    best_ranking = np.argsort(-q_doc_weights)[:cutoff]
    results[qid] = pl.metrics_based_on_samples(best_ranking[None, :],
                                                sorted_weights,
                                                sorted_additions,
                                                q_doc_weights[:, None])
  return results