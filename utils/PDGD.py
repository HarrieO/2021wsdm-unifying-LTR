# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf

def update(model, optimizer, qid,
          data_split, ranking,
          clicks, scores):

  if not np.any(clicks) or np.all(clicks):
    return

  n_docs = scores.shape[0]
  cutoff = ranking.shape[0]
  cutoff_range = np.arange(cutoff)

  included = np.ones(cutoff, dtype=np.int32)
  if not clicks[-1]:
    included[1:] = np.cumsum(clicks[::-1])[:0:-1]
  neg_ind = np.where(np.logical_xor(clicks, included))[0]
  pos_ind = np.where(clicks)[0]

  n_pos = pos_ind.shape[0]
  n_neg = neg_ind.shape[0]
  n_pairs = n_pos*n_neg

  ninf_mask = np.zeros(n_docs)
  ninf_mask[ranking] = np.NINF
  rest_denom = np.sum(np.exp(scores + ninf_mask))

  ranking_scores = np.exp(scores[ranking])
  denom = np.cumsum(ranking_scores[::-1])[::-1]
  ranking_probs = ranking_scores/(denom + rest_denom)

  pair_pos = np.tile(pos_ind, n_neg)
  pair_neg = np.repeat(neg_ind, n_pos)

  pair_range = np.arange(n_pairs)
  flipped_rankings = np.tile(np.arange(cutoff)[None, :], [n_pairs, 1])
  flipped_rankings[pair_range, pair_pos] = pair_neg
  flipped_rankings[pair_range, pair_neg] = pair_pos
  flipped_scores = ranking_scores[flipped_rankings]
  flipped_denom = np.cumsum(flipped_scores[:,::-1], axis=1)[:,::-1]
  flipped_probs = flipped_scores/(flipped_denom + rest_denom)

  ranking_prob = np.prod(ranking_probs)
  flipped_prob = np.prod(flipped_probs, axis=1)
  pair_weights = flipped_prob/(flipped_prob + ranking_prob)

  q_feat = data_split.query_feat(qid)
  ranking_feat = q_feat[ranking, :]
  with tf.GradientTape() as tape:

      tf_scores = tf.exp(model(ranking_feat)[:, 0])
      tf_pos = tf.convert_to_tensor(pair_pos)
      tf_neg = tf.convert_to_tensor(pair_neg)
      pos_scores = tf.gather(tf_scores, tf_pos)
      neg_scores = tf.gather(tf_scores, tf_neg)
      pair_probs = pos_scores/(pos_scores + neg_scores)
      
      loss = -tf.reduce_sum(pair_probs*pair_weights)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def biased_update(
          model, optimizer, qid,
          data_split, ranking,
          clicks, scores):

  if not np.any(clicks) or np.all(clicks):
    return

  n_docs = scores.shape[0]
  cutoff = ranking.shape[0]
  cutoff_range = np.arange(cutoff)

  included = np.ones(cutoff, dtype=np.int32)
  if not clicks[-1]:
    included[1:] = np.cumsum(clicks[::-1])[:0:-1]
  neg_ind = np.where(np.logical_xor(clicks, included))[0]
  pos_ind = np.where(clicks)[0]

  n_pos = pos_ind.shape[0]
  n_neg = neg_ind.shape[0]
  n_pairs = n_pos*n_neg

  pair_pos = np.tile(pos_ind, n_neg)
  pair_neg = np.repeat(neg_ind, n_pos)

  q_feat = data_split.query_feat(qid)
  ranking_feat = q_feat[ranking, :]
  with tf.GradientTape() as tape:

      tf_scores = tf.exp(model(ranking_feat)[:, 0])
      tf_pos = tf.convert_to_tensor(pair_pos)
      tf_neg = tf.convert_to_tensor(pair_neg)
      pos_scores = tf.gather(tf_scores, tf_pos)
      neg_scores = tf.gather(tf_scores, tf_neg)
      pair_probs = pos_scores/(pos_scores + neg_scores)
      
      loss = -tf.reduce_sum(pair_probs)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))