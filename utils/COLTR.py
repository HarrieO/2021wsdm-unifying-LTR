# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf

def prepare_candidate_models(model, n_candidates):
  n_feat = model.get_weights()[0].shape[0]
  candidates = [tf.keras.models.clone_model(model)
                for _ in range(n_candidates)]
  [m.build(input_shape=(1, n_feat)) for m in candidates]
  return candidates

def update(model, candidates, qid,
           data_split, ranking,
           clicks, scores, learning_rate=0.01,
           unit=1.0, var_weight=1.0):

  if not np.any(clicks) or np.all(clicks):
    return

  n_docs = scores.shape[0]
  cutoff = ranking.shape[0]
  n_candidates = len(candidates)

  included = np.greater(np.cumsum(clicks[::-1])[::-1],0)
  risk = np.logical_xor(included, clicks)

  if not np.any(risk):
    return

  threshold = np.mean(risk)
  variance = np.sum((risk-np.mean(risk))**2./risk.shape[0]**2.)
  threshold += var_weight*np.sqrt(variance/risk.shape[0])

  ninf_mask = np.zeros(n_docs)
  ninf_mask[ranking] = np.NINF
  rest_denom = np.sum(np.exp(scores + ninf_mask))

  ranking_scores = np.exp(scores[ranking])
  denom = np.cumsum(ranking_scores[::-1])[::-1]
  ranking_probs = ranking_scores/(denom + rest_denom)

  total_units = np.sum([np.prod(s.shape) for s in model.get_weights()])
  vectors = np.random.randn(n_candidates, total_units)
  vector_norms = np.sqrt(np.sum(vectors ** 2., axis=1))
  vectors /= vector_norms[:, None]/unit

  all_weights = []
  i = 0
  for s in model.get_weights():
    tile_shape = (n_candidates,) + (1,)*len(s.shape)
    tiled_weights = np.tile(s, tile_shape)
    cur_size = int(np.prod(s.shape))
    tiled_weights += np.reshape(vectors[:, i:i+cur_size],
                                tiled_weights.shape)
    all_weights.append(tiled_weights)
    i += cur_size

  cand_probs = np.empty((n_candidates, ranking_probs.shape[0]))
  q_feat = data_split.query_feat(qid)
  for i, cand in enumerate(candidates):
    cand_weights = [w[i,:] for w in all_weights]
    cand.set_weights(cand_weights)

    c_scores = cand(q_feat)[:,0].numpy()

    c_rest_denom = np.sum(np.exp(c_scores + ninf_mask))

    c_ranking_scores = np.exp(c_scores[ranking])
    c_denom = np.cumsum(c_ranking_scores[::-1])[::-1]

    cand_probs[i,:] = c_ranking_scores/(c_denom + rest_denom)

  ratios = cand_probs/ranking_probs[None,:]

  r = np.sum(risk[None, :]*ratios, axis=1)
  s = np.sum(ratios, axis=1)

  c_variance = np.sum((risk-(r/s)[:, None])**2.*(ratios)**2.,
                    axis=1)
  c_variance /= s**2.
  c_variance = np.sqrt(c_variance/risk.shape[0])

  # print('_____')
  # print(clicks)
  # print(risk)
  # print(threshold)
  # print(r/s + c_variance)

  improvements = np.less(r/s + var_weight*c_variance, threshold)
  if np.any(improvements):
    # print(vectors[improvements, :].shape)
    new_weight_vector = np.mean(vectors[improvements, :], axis=0)

    new_weights = []
    i = 0
    for s in model.get_weights():
      cur_size = int(np.prod(s.shape))
      shaped_weights = np.reshape(new_weight_vector[i:i+cur_size],
                                  s.shape)
      new_weights.append(s + learning_rate*shaped_weights)
      i += cur_size
    model.set_weights(new_weights)
