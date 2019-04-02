# cudanova - fast PERMANOVA testing with GPUs.

An implementation of permutational multivariate ANOVA which is capable of taking advantage of multiple GPUs for fast, parallelized non-parametric significance testing.

Includes F-value caching to speed up multiple tests carried out on the same data. Even before caching, cudanova performs about **14x faster** on 2x Titan V GPUs than scikit-bio's CPU implementation on 2x Xeon E5-2690 v2 CPUs.

## Example usage

```
import cudanova
import numpy as np

data = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 0., 1]])

# Each one of these groupings represents an individual test.
groupings = [[0, 1, 0, 1], [1, 0, 1, 0]]

p_values = cudanova.permanova(data, groupings, permutations=10000, num_gpus=2)
```