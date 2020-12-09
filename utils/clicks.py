# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import sharedmem
import numpy as np
import utils.ranking as rnk

def get_click_model(click_model_name):
  if click_model_name == 'default':
    def alpha_beta(ranks):
      pos_bias = 0.35*1/(ranks+1.) + 0.65/(1.+0.05*ranks)
      eplus = 1./(1.+0.005*ranks)
      emin = 0.65/(ranks+1.)
      return pos_bias*(eplus - emin), pos_bias*emin
    return alpha_beta
  else:
    raise NotImplementedError('Click model %s is not implemented' % click_model_name)

def sample_from_click_probs(click_probs):
  coin_flips = np.random.uniform(size=click_probs.shape)
  return np.where(np.less(coin_flips, click_probs))[0]

def inverse_rank_prob(inv_ranking, eta):
    return (1./(inv_ranking+1.))**eta

def get_relevance_click_model(click_model):
  if click_model == 'binarized':
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      rel_prob = np.full(n_docs, 0.1)
      rel_prob[labels>2] = 1.
      return rel_prob
  elif 'linear' in click_model:
    min_prob = 0.2
    max_prob = float(click_model.replace('linear', ''))
    lin_scalar = (max_prob-min_prob)/4.
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      rel_prob = np.full(n_docs, min_prob)
      rel_prob += labels*lin_scalar
      return rel_prob
  else:
    raise ValueError('Unknown click model: %s' % click_model)
  return relevance_click_prob

def generate_clicks(data_split,
                    inverted_ranking,
                    click_model,
                    n_clicks,
                    eta,
                    query_prob=None,
                    click_patterns=False):

  relevance_click_prob = get_relevance_click_model(click_model)

  # Number of documents per query
  doc_per_q = (data_split.doclist_ranges[1:]
                 - data_split.doclist_ranges[:-1])
  # Number of possible combinations of documents and positions
  n_doc_pos = np.sum(doc_per_q**2)

  docpos_ranges = np.zeros(data_split.num_queries()+1, dtype=np.int64)
  docpos_ranges[1:] = np.cumsum(doc_per_q**2)

  query_freq = np.zeros(data_split.num_queries(), dtype=np.int64)
  docpos_display = np.zeros(n_doc_pos, dtype=np.int64)
  doc_click = np.zeros(data_split.num_docs(), dtype=np.int64)
  if click_patterns:
    clk_patterns = {}

  click_prob = np.zeros(data_split.num_docs(),
                        dtype=np.float64)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_docs = e_i - s_i
    q_labels = data_split.query_labels(qid)
    rel_click_prob = relevance_click_prob(q_labels)
    obs_click_prob = inverse_rank_prob(inverted_ranking[s_i:e_i], eta)
    click_prob[s_i:e_i] = rel_click_prob*obs_click_prob

  clicks_generated = 0
  docs_shown = 0
  while clicks_generated < n_clicks:
    qid = np.random.choice(data_split.num_queries(), p=query_prob)
    query_freq[qid] += 1

    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_docs = e_i - s_i
    q_click_prob = click_prob[s_i:e_i]
    c_i = sample_from_click_probs(q_click_prob)
    docs_shown += n_docs

    if click_patterns:
      key = tuple(s_i + c_i)
      clk_patterns[key] = clk_patterns.get(key, 0) + 1

    clicks_generated += c_i.size
    doc_click[s_i + c_i] += 1

    s_j, e_j = docpos_ranges[qid:qid+2]
    q_display = np.reshape(docpos_display[s_j:e_j],
                           (n_docs, n_docs))
    q_display[np.arange(n_docs), inverted_ranking[s_i:e_i]] += 1

  result = {
      'query_freq': query_freq,
      'data_split_name': data_split.name,
      'doc_position_display_freq': docpos_display,
      'clicks_per_doc': doc_click,
      'num_clicks': clicks_generated,
      'cutoff': None,
    }
  if click_patterns:
    result['click_patterns'] = clk_patterns
  return result

def compute_weights(clicks, data_split, eta):
  query_freq = clicks['query_freq']
  docpos_display = clicks['doc_position_display_freq']
  clicks_per_doc = clicks['clicks_per_doc']

  doc_per_q = (data_split.doclist_ranges[1:]
                 - data_split.doclist_ranges[:-1])
  docpos_ranges = np.zeros(data_split.num_queries()+1, dtype=np.int64)
  docpos_ranges[1:] = np.cumsum(doc_per_q**2)

  doc_weights = np.zeros(clicks_per_doc.size)
  inv_prop = np.zeros(clicks_per_doc.size)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    s_j, e_j = docpos_ranges[qid:qid+2]

    n_docs = e_i - s_i
    q_freq = query_freq[qid]
    q_click = clicks_per_doc[s_i:e_i]
    q_display = np.reshape(docpos_display[s_j:e_j],
                           (n_docs, n_docs))

    estimated_policy = q_display/float(np.maximum(q_freq, 1.))
    pos_exam_prob = (1./(np.arange(n_docs)+1.))**eta
    estimated_exam_prob = np.sum(estimated_policy*pos_exam_prob[None, :],
                                 axis=1)
    click_prob = q_click.astype(np.float64)/max(np.amax(q_click),1.)

    estimated_exam_prob[np.equal(estimated_exam_prob, 0)] = 1
    unnorm_weights = click_prob/estimated_exam_prob
    if np.sum(unnorm_weights) == 0:
      norm_weights = unnorm_weights
    else:
      norm_weights = unnorm_weights/np.sum(unnorm_weights)
    norm_weights *= float(q_freq)/np.sum(query_freq)

    doc_weights[s_i:e_i] = norm_weights
    inv_prop[s_i:e_i] = estimated_exam_prob

  return doc_weights, inv_prop

def sample_clicks(data_split, clicks, percentage, q_freq):
  doc_click = clicks['clicks_per_doc']
  doc_probs = doc_click/float(np.sum(doc_click))
  n_docs = doc_click.size
  n_clicks = np.sum(doc_click)
  n_target = int(n_clicks*percentage)
  n_sampled = 0
  sampled_clicks = np.zeros(n_docs, dtype=np.int64)
  while n_sampled < n_target:
    selection = np.random.choice(n_docs,
                                 size=n_target-n_sampled,
                                 replace=True,
                                 p=doc_probs)
    sampled_clicks += np.bincount(selection, minlength=n_docs)
    sampled_clicks = np.minimum(sampled_clicks, doc_click)
    n_sampled = np.sum(sampled_clicks)

  q_freq_left = np.zeros(n_docs, dtype=np.float64)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.query_range(qid)
    q_click = doc_click[s_i:e_i]
    q_sampled = sampled_clicks[s_i:e_i]
    if np.sum(q_click) > 0:
      perc_sampled = np.sum(q_sampled)/float(np.sum(q_click))
      perc_left = 1. - perc_sampled
      q_freq_left[qid] = q_freq[qid]*perc_left
    else:
      q_freq_left[qid] = q_freq[qid]*(1.-percentage)
  return doc_click - sampled_clicks, q_freq_left, sampled_clicks

def weights_from_clicks(data_split, inv_prop, clicks_per_doc, query_freq):
  doc_weights = np.zeros(clicks_per_doc.size)
  query_freq_sum = np.sum(query_freq)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.query_range(qid)
    q_freq = query_freq[qid]
    n_docs = e_i - s_i
    q_click = clicks_per_doc[s_i:e_i]
    q_inv_prop = inv_prop[s_i:e_i].copy()

    click_prob = q_click.astype(np.float64)/max(np.amax(q_click),1.)

    q_inv_prop[np.equal(q_inv_prop, 0)] = 1
    unnorm_weights = click_prob/q_inv_prop
    if np.sum(unnorm_weights) == 0:
      norm_weights = unnorm_weights
    else:
      norm_weights = unnorm_weights/np.sum(unnorm_weights)
    norm_weights *= float(q_freq)/query_freq_sum
    doc_weights[s_i:e_i] = norm_weights
  return doc_weights

def add_clicks(click_list):
  result = click_list[0]
  for clicks in click_list[1:]:
    assert result['data_split_name'] == clicks['data_split_name']
    assert result['cutoff'] == clicks['cutoff']
    result['query_freq'] += clicks['query_freq']
    result['doc_position_display_freq'] += clicks['doc_position_display_freq']
    result['clicks_per_doc'] += clicks['clicks_per_doc']
    result['num_clicks'] += clicks['num_clicks']
    if 'click_patterns' in result:
      cur_clk = result['click_patterns']
      for k, v in clicks['click_patterns'].items():
        cur_clk[k] = cur_clk.get(k, 0) + v
  return result

def _make_shared(numpy_matrix):
    """
    Avoids the copying of Read-Only shared memory.
    """
    if numpy_matrix is None:
      return None
    else:
      shared = sharedmem.empty(numpy_matrix.shape,
                               dtype=numpy_matrix.dtype)
      shared[:] = numpy_matrix[:]
      return shared

def make_clicks_shared(clicks, num_proc):
  if num_proc > 1:
    for k in [
        'query_freq',
        'doc_position_display_freq',
        'clicks_per_doc']:
      clicks[k] = _make_shared(clicks[k])
