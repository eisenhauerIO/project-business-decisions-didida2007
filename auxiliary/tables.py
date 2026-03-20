import pandas as pd
from great_tables import GT, style, loc

from .dataframe_analysis import ProcessedData



class TableGenerator(ProcessedData):

    def __init__(self, df):
        
        super().__init__(df)
        self.col = ["classize", "c_size", "tipuach", "verbsize", "mathsize", "avgverb", "avgmath"]
        self.row_index = ["Class Size", "Enrollment", "Percentage Disadvantaged", "Reading Size", "Math Size", "Average Verbal", "Average Math"]
        self.label_sc = f"{self.label} ({self.n_classes} classes, {self.n_schools} schools)"

    @property
    def row(self):
        return self.df[self.col]



    def descriptive_table(self):

        row = self.row

        table = pd.DataFrame({
            "mean": row.mean(),
            "std": row.std(),
            "q10": row.quantile(0.1),
            "q25": row.quantile(0.25),
            "q50": row.quantile(0.5),
            "q75": row.quantile(0.75),
            "q90": row.quantile(0.9)
        })

        table.index = self.row_index
        table.index.name = "Variable"

        table = table.round(1).astype(object)
        table.iloc[:5, -5:] = table.iloc[:5, -5:].round(0).astype(int)

        table_gt = GT(table.reset_index()
        ).tab_spanner(
            label = "Quantiles",
            columns = ["q10", "q25", "q50", "q75", "q90"]
        ).tab_header(
            title = "Unweighted Descriptive Statistics",
            subtitle = self.label_sc
        ).cols_label(
            mean ="Mean",
            std = "S.D.",
            q10 = "0.10",
            q25 = "0.25",
            q50 = "0.50",
            q75 = "0.75",
            q90 = "0.90"
        ).cols_align(
            align = "left",
            columns = "Variable"
        ).cols_align(
            align = "center",
            columns = ["mean", "std", "q10", "q25", "q50", "q75", "q90"]
        )

        return table_gt
    


    def discontinuity_table(self, other):

        query_str = (
        "(c_size >= 36 & c_size <= 45) | "
        "(c_size >= 76 & c_size <= 85) | "
        "(c_size >= 116 & c_size <= 125)"
        )

        self_filtered, other_filtered = [df.query(query_str) for df in [self.df, other.df]]

        col = self.col
        self_row = self_filtered[col]
        other_row = other_filtered[col]
        row_index = self.row_index

        table = pd.DataFrame({
            "mean1": self_row.mean(),
            "std1": self_row.std(),
            "mean2": other_row.mean(),
            "std2": other_row.std()
        })

        table.index = row_index
        table.index.name = "Variable"

        table = table.round(1)

        table_gt = GT(table.reset_index()
        ).tab_spanner(
            label = self.label_sc,
            columns = ["mean1", "std1"]
        ).tab_spanner(
            label = other.label_sc,
            columns = ["mean2", "std2"]
        ).tab_header(
            title = "+/-5 Discontinuity Sample",
            subtitle = "Enrollment 36-45, 76-85, 116-124"
        ).cols_label(
            mean1 = "Mean",
            std1 = "S.D.",
            mean2 = "Mean",
            std2 = "S.D."
        ).cols_align(
            align="left",
            columns="Variable"
        ).cols_align(
            align="center",
            columns=["mean1", "std1", "mean2", "std2"]
        )

        return table_gt
    

