import pandas as pd
import numpy as np

class ProcessedData:
    """Base class for data processing with Maimonides Rule calculations."""

    BIN_EDGES = range(0, 170, 10)
    BIN_LABELS = range(5, 165, 10)

    def __init__(self, df):
        """Initialize with dataframe; compute grade, label, and school/class counts."""
        self.df = df
        self.predict_class_size()
        self.grade = int(df["grade"].iloc[0])
        self.label = f"{self.grade}th Grade"
        self.n_classes = len(df)
        self.n_schools = df["schlcode"].nunique()

    @staticmethod
    def maimonides_rule(enrollment):
        """Apply Maimonides' rule: predicted class size based on enrollment."""
        return enrollment / (np.floor((enrollment - 1) / 40) + 1)

    @classmethod
    def enrollment_bins(cls, enrollment):
        """Bin enrollment into 10-unit groups for analysis."""
        return pd.cut(enrollment, bins=cls.BIN_EDGES, labels=cls.BIN_LABELS).astype(float)
    
    def predict_class_size(self):
        """Add predicted class size (p_size) column using Maimonides rule."""
        self.df['p_size'] = self.maimonides_rule(self.df['c_size'])
        return self.df