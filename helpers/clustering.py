import numpy as np
import pandas as pd

class extract_clusters(labels, path):
    for label in labels:
    indices = np.where(labels == label)
    cluster = data_xyz.iloc[indices]
    cluster.to_csv(path + str(label) + '.csv')