# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.ranking as rnk

def sample_rankings(log_scores, n_samples, cutoff=None, prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings = np.empty((n_samples, ranking_len), dtype=np.int32)
  inv_rankings = np.empty((n_samples, n_docs), dtype=np.int32)
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)

  if cutoff:
    inv_rankings[:] = ranking_len

  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    cumprobs = np.cumsum(probs, axis=1)
    random_values = np.random.uniform(size=n_samples)
    greater_equal_mask = np.greater_equal(random_values[:,None], cumprobs)
    sampled_ind = np.sum(greater_equal_mask, axis=1)

    rankings[:, i] = sampled_ind
    inv_rankings[ind, sampled_ind] = i
    rankings_prob[:, i] = probs[ind, sampled_ind]
    log_scores[ind, sampled_ind] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob

def gumbel_sample_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
  gumbel_scores = -(log_scores[None,:]+gumbel_samples)

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                gumbel_scores,
                                ranking_len,
                                invert=inverted)

  if not doc_prob:
    return rankings, inv_rankings, None, None

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    rankings_prob[:, i] = probs[ind, rankings[:, i]]
    log_scores[ind, rankings[:, i]] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob, None

def metrics_based_on_samples(sampled_rankings,
                             weight_per_rank,
                             addition_per_rank,
                             weight_per_doc,):
  cutoff = sampled_rankings.shape[1]
  return np.sum(np.mean(
              weight_per_doc[sampled_rankings]*weight_per_rank[None, :cutoff],
            axis=0) + addition_per_rank[:cutoff], axis=0)

def datasplit_metrics(data_split,
                      policy_scores,
                      weight_per_rank,
                      addition_per_rank,
                      weight_per_doc,
                      query_norm_factors=None,
                      n_samples=1000):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)
  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    if not np.all(np.equal(q_doc_weights, 0.)):
      q_policy_scores = data_split.query_values_from_vector(qid, policy_scores)
      sampled_rankings = gumbel_sample_rankings(q_policy_scores,
                                                n_samples,
                                                cutoff=cutoff)[0]
      results[qid] = metrics_based_on_samples(sampled_rankings,
                                              weight_per_rank,
                                              addition_per_rank,
                                              q_doc_weights[:, None])
  if query_norm_factors is not None:
    results /= query_norm_factors

  return np.mean(results, axis=0)


def gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              log_scores):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  cutoff = sampled_rankings.shape[1]

  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)
  result = np.zeros((n_docs, n_docs))
  log_scores = np.tile(log_scores[None,:], (n_samples, 1))

  cumulative_grad = np.zeros((n_samples, n_docs))
  cur_grad = np.zeros((n_docs, n_docs))

  for i in range(cutoff):
    cur_grad[:] = 0.
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    denom = np.log(np.sum(np.exp(log_scores), axis=1))
    cur_doc_prob = np.exp(log_scores[:,:] - denom[:, None])

    cur_grad[doc_ind, doc_ind] += np.mean(cur_doc_prob, axis=0)
    cur_grad -= np.mean(cur_doc_prob[:, :, None]*cur_doc_prob[:, None, :], axis=0)
    if i > 0:
      cur_grad += np.mean(cur_doc_prob[:, :, None]*cumulative_grad[:, None, :], axis=0)

    result += weight_per_rank[i]*cur_grad

    if i < n_docs - 1:
      cumulative_grad[sample_ind, sampled_rankings[:, i]] += 1
      cumulative_grad -= cur_doc_prob

      log_scores[sample_ind, sampled_rankings[:, i]] = np.NINF

  return result

def fast_gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              weight_per_doc,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:,:] - denom[:, :, None])

  # # delete very large matrices
  del inf_mask
  del log_scores
  del denom

  doc_grad_per_rank = np.zeros((n_samples, ranking_len, n_docs))
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:],
                    sampled_rankings] += 1
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:], :] -= doc_prob_per_sample

  cum_grad = np.cumsum(doc_grad_per_rank, axis=1)
  cum_grad *= weight_per_rank[None, :ranking_len, None]
  cum_grad *= weight_per_doc[sampled_rankings][:, :, None]
  
  return np.sum(np.mean(cum_grad, axis=0), axis=0)

def slow_gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:,:] - denom[:, :, None])

  # # delete very large matrices
  del inf_mask
  del log_scores
  del denom

  result = np.zeros((n_docs, n_docs))

  cur_doc_grad = np.zeros((ranking_len, n_docs, n_docs))
  cur_doc_grad[rank_ind[:, None],
               doc_ind[None, :],
               doc_ind[None, :]] = np.mean(doc_prob_per_sample, axis=0)
  cur_doc_grad -= np.mean(doc_prob_per_sample[:, :, :, None]
                          *doc_prob_per_sample[:, :, None, :],
                          axis=0)
  cur_doc_grad *= weight_per_rank[:ranking_len, None, None]

  result += np.sum(cur_doc_grad, axis=0)

  del cur_doc_grad

  cumulative_grad = np.zeros((n_samples, ranking_len-1, n_docs))
  cumulative_grad[sample_ind[:, None], rank_ind[None, :-1],
                  sampled_rankings[:, :-1]] += 1
  cumulative_grad -= doc_prob_per_sample[:, :-1, :]
  cumulative_grad = np.cumsum(cumulative_grad, axis=1)
  cumulative_grad *= weight_per_rank[None, 1:ranking_len, None]

  per_doc_cum_grad = np.mean(cumulative_grad[:,:,None,:]
                             *doc_prob_per_sample[:,:-1,:,None],
                             axis=0)
  result += np.sum(per_doc_cum_grad, axis=0)

  return result

def hybrid_gradient_based_on_samples(
                              sampled_rankings,
                              weight_per_rank,
                              weight_per_doc,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:-1,:] - denom[:, :-1, None])

  # delete very large matrices
  del inf_mask
  del denom

  final_scores = log_scores[:, -1, :]
  final_denom = np.log(np.sum(np.exp(log_scores[:, -1, :]), axis=1))
  final_prob = np.exp(final_scores - final_denom[:, None])

  # delete very large matrices
  del log_scores
  del final_scores
  del final_denom

  doc_grad_per_rank = np.zeros((n_samples, ranking_len-1, n_docs), dtype=np.float64)
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:-1],
                    sampled_rankings[:,:-1]] += 1
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:-1], :] -= doc_prob_per_sample

  cum_grad = np.cumsum(doc_grad_per_rank, axis=1)
  weighted_cum_grad = cum_grad*weight_per_rank[None, :ranking_len-1, None]
  weighted_cum_grad *= weight_per_doc[sampled_rankings[:,:-1]][:, :, None]
  
  sample_based_grad = np.sum(np.mean(weighted_cum_grad, axis=0), axis=0)

  rel_mask = np.not_equal(weight_per_doc, 0)
  rel_ind = doc_ind[rel_mask]
  n_rel = rel_ind.shape[0]

  final_grad = np.zeros((n_samples, n_rel, n_docs), dtype=np.float64)
  final_grad -= final_prob[:,None,:]
  final_grad[sample_ind[:,None],
             doc_ind[None,:n_rel],
             rel_ind[None,:]] += 1.
  final_grad += cum_grad[:, -1, None, :]
  final_grad *= final_prob[:, rel_ind, None]
  final_grad = np.mean(final_grad, axis=0)
  final_grad *= weight_per_doc[rel_ind, None]

  final_grad = np.sum(final_grad, axis=0)
  final_grad *= weight_per_rank[ranking_len-1] 

  return sample_based_grad+final_grad
