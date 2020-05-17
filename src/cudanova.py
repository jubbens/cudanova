import tensorflow as tf
import numpy as np
from tqdm import trange
import sys
import psutil

cn_num_groups = 2
cn_cache_percent_limit = 90.0


def permanova(points, groupings, permutations=10000, sig_level=5e-2, distances=None, num_gpus=1, max_batch_size=1000, show_progress=False):
    """GPU accelerated implementation of permutational multivariate analysis of variance (PERMANOVA).

    :param points: A list of vectors.
    :param groupings: A list of groupings, where the length of the list is equal to the number of tests and
    each element is a list of ones and zeros representing the group memberships for an individual test.
    :param sig_level: Optional significance level. Used for reporting number of tests with significant p-values on the
    fly.
    :param permutations: Optional number of permutations (default 1000). More permuatations means a smaller lower bound
    on p-values.
    :param distances: Optional distance matrix. If not provided, a euclidean distance matrix is generated.
    :param num_gpus: Optional number of GPUs to use. Multiple tests can be parallelized across multiple GPUs.
    :param max_batch_size: Optional maximum of permutations to do on each GPU in parallel (default 1000). If you get a
    GPU memory error, try a lower max batch size.
    :param show_progress: Optional set to True if you want to see a progress bar with info during the run.
    :return: an ndarray of p-values.
    """

    def list_to_dict(l, v):
        dict = {}

        for item in l:
            dict[item] = v

        return dict

    def make_distance_matrix(s):
        dim = len(s)
        ret = np.zeros(shape=[dim, dim])

        for i in range(dim):
            for j in range(dim):
                ret[i][j] = np.linalg.norm(s[i] - s[j])

        return ret

    if num_gpus < 1:
        raise Exception('num_gpus should be >=1.')

    # Calculate the best batch size
    batch_size = permutations / num_gpus

    while batch_size > max_batch_size:
        batch_size /= 2

    points = np.array(points)
    groupings = np.array(groupings)

    # Make a distance matrix of points
    if distances is None:
        distances = make_distance_matrix(points)

    # Normalize distances
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    num_tests = len(groupings)
    N = len(points)

    groupings = np.split(groupings, num_tests, axis=0)
    accumulated_F = [0 for j in range(num_tests)]

    all_forward_ops = []
    all_fp = []
    placeholders = []

    graph = tf.Graph()

    with graph.as_default():
        for dev in range(num_gpus):
            device_c = '/device:GPU:{0}'.format(dev)

            with graph.device(device_c):
                placeholders.append(tf.placeholder(tf.int32, shape=(1, N)))
                ag = placeholders[dev]

                dm = tf.Variable(distances, name='distances', dtype=tf.float32)

                with graph.device('/cpu:0'):
                    group_n = tf.map_fn(lambda x: tf.count_nonzero(x, dtype=tf.int32), ag)
                    random_groupings = tf.map_fn(lambda g: tf.random_shuffle(g), tf.tile(ag, [batch_size, 1]))

                def SS(mat, groups):
                    groups = tf.cast(groups, dtype=tf.float32)
                    mask = tf.tile(tf.expand_dims(groups, -1), [1, 1, N])
                    mask_t = tf.multiply(mask, tf.transpose(mask, perm=[0, 2, 1]))
                    mat_masked = tf.multiply(mat, mask_t)

                    ss = tf.map_fn(lambda m: tf.reduce_sum(tf.square(m)), mat_masked)
                    gnf = tf.cast(group_n, dtype=tf.float32)
                    ss_normalized = tf.divide(ss, (gnf * (gnf - 1.)))

                    return ss_normalized

                # Total sum of squares
                SSt = tf.reduce_sum(tf.square(dm)) / tf.constant((N * (N - 1)), dtype=tf.float32)
                SSw = SS(dm, random_groupings)
                SSw_actual = SS(dm, ag)

                SSa = tf.subtract(SSt, SSw)
                F_num = tf.divide(SSa, cn_num_groups - 1)

                SSa_actual = tf.subtract(SSt, SSw_actual)
                F_num_actual = tf.divide(SSa_actual, cn_num_groups - 1)

                F_actual = tf.divide(F_num_actual, tf.divide(SSw_actual, N - cn_num_groups))
                F_p = tf.divide(F_num, tf.divide(SSw, N - cn_num_groups))
                is_greater = tf.cast(tf.greater_equal(F_p, F_actual), dtype=tf.int32)

                all_forward_ops.append(tf.reduce_sum(is_greater, axis=0))
                all_fp.append(F_p)

        session = tf.Session(graph=graph)

        session.run(tf.initialize_all_variables())

        Ftionary = {}

        for j in range(N+1):
            Ftionary[j] = []

        num_sig = 0

        if show_progress:
            t = trange(num_tests)
        else:
            t = range(num_tests)

        for i in t:
            fd = list_to_dict(placeholders, groupings[i])

            num_in_grouping = np.count_nonzero(groupings[i])
            cache_MB = sum(map(sys.getsizeof, Ftionary.itervalues())) / float(1048576)

            # First, get an F value and compare it to cached F values
            current_F = session.run(F_actual, feed_dict=fd)
            accumulated_F[i] = len([x for x in Ftionary[num_in_grouping] if x > current_F])
            cache_size = len(Ftionary[num_in_grouping])

            for j in range(cache_size, permutations, num_gpus * batch_size):
                if show_progress:
                    t.set_description('Significant: {1}, Cache: {2:.2f} MB, Iteration: {0}'.format(j, num_sig, cache_MB))

                current_passes, fp = session.run([all_forward_ops, all_fp], feed_dict=fd)

                # Add the F values to the cache
                if cache_size < permutations and psutil.virtual_memory().percent < cn_cache_percent_limit:
                    for f in fp:
                        Ftionary[num_in_grouping].extend(f)

                    if len(Ftionary[num_in_grouping]) > permutations:
                        Ftionary[num_in_grouping] = Ftionary[num_in_grouping][:permutations]

                accumulated_F[i] = accumulated_F[i] + np.sum(current_passes)

            if accumulated_F[i] <= sig_level * permutations:
                num_sig = num_sig + 1

    p_values = np.divide(np.add(np.array(accumulated_F), 1), float(permutations + 1))

    # remove p-values which resulted from padding (if any)
    p_values = p_values[:num_tests]

    return p_values
