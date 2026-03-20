import pandas as pd
import numpy as np

class ProcessedData:

    BIN_EDGES = range(0, 170, 10)
    BIN_LABELS = range(5, 165, 10)

    def __init__(self, df):
        self.df = df
        self.predict_class_size()
        self.grade = int(df["grade"].iloc[0])
        self.label = f"{self.grade}th Grade"
        self.n_classes = len(df)
        self.n_schools = df["schlcode"].nunique()

    @staticmethod
    def maimonides_rule(enrollment):
        return enrollment / (np.floor((enrollment - 1) / 40) + 1)

    @classmethod
    def enrollment_bins(cls, enrollment):
        return pd.cut(enrollment, bins=cls.BIN_EDGES, labels=cls.BIN_LABELS).astype(float)
    
    def predict_class_size(self):
        self.df['p_size'] = self.maimonides_rule(self.df['c_size'])
        return self.df