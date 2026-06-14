import numpy as np
import pandas as pd


def write_to_csv(x, n, u, T, q, filename):
    df = pd.DataFrame({'x': x, 'n': n, 'u': u, 'T': T, 'q': q})
    df.to_csv(filename)
