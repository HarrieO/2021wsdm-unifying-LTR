# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

def efficient_spacing(n1, n2, min_exp, max_exp):
  min_n = min(n1, n2)
  max_n = max(n1, n2)
  max_points = np.logspace(min_exp, max_exp, max_n+2,
                            endpoint=True,
                            dtype=np.int64)
  min_points = np.logspace(min_exp, max_exp, min_n+2,
                            endpoint=True,
                            dtype=np.int64)

  result = np.zeros(min_n+2, dtype=np.int32)
  start = 1
  for i in range(1, min_n+1):
    min_i = np.argmin(np.abs(max_points[start:]-min_points[i]))
    result[i] = min_i + start
  result[-1] = max_n+1

  if n1 < n2:
    return max_points[result], max_points
  else:
    return max_points, max_points[result]
