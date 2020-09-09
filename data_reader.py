import os

import pandas as pd
import numpy as np
from constants import MAPPING

BASE_DIR = os.path.dirname(__file__)

class Data:
    def __init__(self):
        train_data_path = os.path.join(BASE_DIR, 'data/train.csv')
        test_data_path = os.path.join(BASE_DIR, 'data/test.csv')
        self.train = pd.read_csv(train_data_path)
        self.test = pd.read_csv(test_data_path)
        self._correct_variables_with_map()

    def _correct_variables_with_map(self):
        # Apply same operation to both train and test
        for df in [self.train, self.test]:
            # Fill in the values with the correct mapping
            df['dependency'] = df['dependency'].replace(MAPPING).astype(np.float64)
            df['edjefa'] = df['edjefa'].replace(MAPPING).astype(np.float64)
            df['edjefe'] = df['edjefe'].replace(MAPPING).astype(np.float64)