from collections import Counter

import numpy as np
import pandas as pd


def drift_psi(var_count_train, var_test: pd.Series, threshold=1.0):
    """
    Compute the Population Stability Index (PSI) for a categorical variable between a train and test datasets
    See https://www.mdpi.com/2227-9091/7/2/53/htm
    Parameters
    ----------
    var_train: pd.Series
        A categorical column in train dataset
    var_test: pd.Series
        A categorical column in test dataset that is compared to var_train
    max_num_categories: int = 10
        Maximum number of modalities
    Returns
    -------
    psi
        The PSI score
    output_data
        Pandas dataframe giving frequencies and psi for each modality
   """

    # var_count_train = Counter(var_train)
    var_count_test = Counter(var_test)
    min_distribution_probability = 0.0001
    all_modalities = list(set(var_count_train.keys()).union(set(var_test)))
    train_distribution = np.array([var_count_train.get(i, min_distribution_probability) for i in all_modalities]) / sum(var_count_train.values())
    test_distribution = np.array([var_count_test.get(i, min_distribution_probability) for i in all_modalities]) / sum(var_count_test.values())

    psi = np.sum((train_distribution - test_distribution) * np.log(train_distribution / test_distribution))
    return int(psi >= threshold)