import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from great_tables import GT, style, loc

from .dataframe_analysis import ProcessedData


class TableGenerator(ProcessedData):
    """Generate publication-quality regression and summary tables."""

    # Common query applied across multiple table builders
    DISC_QUERY = "(c_size >= 36 & c_size <= 45) | (c_size >= 76 & c_size <= 85) | (c_size >= 116 & c_size <= 125)"
    PM5_QUERY = "(c_size >= 36 & c_size <= 45) | (c_size >= 76 & c_size <= 85) | (c_size >= 116 & c_size <= 125)"
    PM3_QUERY = "(c_size >= 38 & c_size <= 43) | (c_size >= 78 & c_size <= 83) | (c_size >= 118 & c_size <= 123)"

    @staticmethod
    def _piecewise_linear_trend(enrollment):
        """Compute piecewise-linear trend for IV specification based on enrollment thresholds."""
        e = pd.Series(enrollment, copy=False)
        trend = pd.Series(np.nan, index=e.index, dtype=float)

        m1 = (e >= 0) & (e <= 40)
        m2 = (e >= 41) & (e <= 80)
        m3 = (e >= 81) & (e <= 120)
        m4 = (e >= 121) & (e <= 160)

        trend.loc[m1] = e.loc[m1]
        trend.loc[m2] = 20 + (e.loc[m2] / 2)
        trend.loc[m3] = (100 / 3) + (e.loc[m3] / 3)
        trend.loc[m4] = (130 / 3) + (e.loc[m4] / 4)
        return trend

    def __init__(self, df):
        """Initialize with columns and row labels for descriptive tables."""
        super().__init__(df)
        self.col = ["classize", "c_size", "tipuach", "verbsize", "mathsize", "avgverb", "avgmath"]
        self.row_index = ["Class size", "Enrollment", "Percent disadvantaged", "Reading size", "Math size", "Average verbal", "Average math"]
        self.label_sc = f"{self.grade}th grade ({self.n_classes} classes, {self.n_schools} schools, tested in 1991)"

    @property
    def row(self):
        """Get selected columns as property."""
        return self.df[self.col]

    @staticmethod
    def _fmt_coef(value):
        """Format coefficient: trim leading zero (e.g., 0.123 → .123)."""
        return f"{value:.3f}".replace("0.", ".").replace("-0.", "-.")

    @staticmethod
    def _fmt_se(value):
        """Format standard error with parentheses and leading zero trim."""
        return f"({value:.3f})".replace("0.", ".").replace("-0.", "-.")

    @staticmethod
    def _extract_param(model_params, model_bse, var_name):
        """Safely extract coefficient and SE for a variable; return formatted strings or empty."""
        if var_name not in model_params.index:
            return "", ""
        coef_str = TableGenerator._fmt_coef(model_params[var_name])
        se_str = TableGenerator._fmt_se(model_bse[var_name])
        return coef_str, se_str

    def descriptive_table(self, other):
        """Generate summary table comparing distributions across grades and samples."""
        num_cols = ["mean", "std", "q10", "q25", "q50", "q75", "q90"]

        def compute_stats(df):
            row = df[self.col]
            t = pd.DataFrame({
                "mean": row.mean(),
                "std": row.std(),
                "q10": row.quantile(0.1),
                "q25": row.quantile(0.25),
                "q50": row.quantile(0.5),
                "q75": row.quantile(0.75),
                "q90": row.quantile(0.9),
            })
            t.index = self.row_index
            t.index.name = "Variable"
            t = t.round(1).astype(object)
            t.iloc[:5, -5:] = t.iloc[:5, -5:].astype(float).round(0).astype(int)
            return t.reset_index()

        def header_row(label):
            return {"Variable": label, **{c: "" for c in num_cols}}

        grid = pd.concat([
            pd.DataFrame([header_row("A. Full sample")]),
            pd.DataFrame([header_row(self.label_sc)]),
            compute_stats(self.df),
            pd.DataFrame([header_row("")]),
            pd.DataFrame([header_row(other.label_sc)]),
            compute_stats(other.df),
        ], ignore_index=True)

        return (
            GT(grid)
            .tab_header(title="UNWEIGHTED DESCRIPTIVE STATISTICS")
            .tab_spanner(label="Quantiles", columns=["q10", "q25", "q50", "q75", "q90"])
            .cols_label(
                Variable="Variable",
                mean="Mean",
                std="S.D.",
                q10="0.10",
                q25="0.25",
                q50="0.50",
                q75="0.75",
                q90="0.90",
            )
            .cols_align(align="left", columns="Variable")
            .cols_align(align="center", columns=num_cols)
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )
    
    def discontinuity_table(self, other):
        """Generate table comparing ±5 discontinuity samples across grades."""
        self_filtered, other_filtered = [df.query(self.DISC_QUERY) for df in [self.df, other.df]]

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

        return (
            GT(table.reset_index())
            .tab_spanner(label=self.label_sc, columns=["mean1", "std1"])
            .tab_spanner(label=other.label_sc, columns=["mean2", "std2"])
            .tab_header(
                title="+/-5 Discontinuity Sample",
                subtitle="Enrollment 36-45, 76-85, 116-124"
            )
            .cols_label(
                mean1="Mean",
                std1="S.D.",
                mean2="Mean",
                std2="S.D."
            )
            .cols_align(align="left", columns="Variable")
            .cols_align(align="center", columns=["mean1", "std1", "mean2", "std2"])
        )

    def ols_estimate(self, regressor):
        """Run OLS with three specifications: base, + disadvantaged, + enrollment controls."""
        specs = [f"{regressor} ~ classize",
                 f"{regressor} ~ classize + tipuach",
                 f"{regressor} ~ classize + tipuach + c_size"]
        models = [sm.OLS.from_formula(s, data=self.df).fit() for s in specs]
        return models

    def build_custom_ols_grid(self, other):
        """Build grid for 12-column OLS table across grades and outcomes."""
        models = {
            "(1)": self.ols_estimate("avgverb")[0],
            "(2)": self.ols_estimate("avgverb")[1],
            "(3)": self.ols_estimate("avgverb")[2],
            "(4)": self.ols_estimate("avgmath")[0],
            "(5)": self.ols_estimate("avgmath")[1],
            "(6)": self.ols_estimate("avgmath")[2],
            "(7)": other.ols_estimate("avgverb")[0],
            "(8)": other.ols_estimate("avgverb")[1],
            "(9)": other.ols_estimate("avgverb")[2],
            "(10)": other.ols_estimate("avgmath")[0],
            "(11)": other.ols_estimate("avgmath")[1],
            "(12)": other.ols_estimate("avgmath")[2],
        }

        cols = [f"({i})" for i in range(1, 13)]
        rows = [
            "Mean score", "(s.d.)", "Regressors",
            "Class size", "",
            "Percent disadvantaged", " ",
            "Enrollment", "  ",
            "Root MSE", "R2", "N",
        ]
        grid = pd.DataFrame("", index=rows, columns=cols)

        # Fill mean/SD/N at block centers
        block_stats = [
            (self.df["avgverb"].mean(), self.df["avgverb"].std(), "(2)"),
            (self.df["avgmath"].mean(), self.df["avgmath"].std(), "(5)"),
            (other.df["avgverb"].mean(), other.df["avgverb"].std(), "(8)"),
            (other.df["avgmath"].mean(), other.df["avgmath"].std(), "(11)"),
        ]
        for mean_val, sd_val, display_col in block_stats:
            grid.at["Mean score", display_col] = f"{mean_val:.1f}"
            grid.at["(s.d.)", display_col] = f"({sd_val:.1f})"

        n_by_block = {
            "(2)": int(models["(1)"].nobs),
            "(5)": int(models["(4)"].nobs),
            "(8)": int(models["(7)"].nobs),
            "(11)": int(models["(10)"].nobs),
        }
        for display_col, n_val in n_by_block.items():
            grid.at["N", display_col] = f"{n_val:,}"

        # Fill coefficients/SEs column by column
        for c, res in models.items():
            if "classize" in res.params:
                grid.at["Class size", c] = self._fmt_coef(res.params["classize"])
                grid.at["", c] = self._fmt_se(res.bse["classize"])
            if "tipuach" in res.params:
                grid.at["Percent disadvantaged", c] = self._fmt_coef(res.params["tipuach"])
                grid.at[" ", c] = self._fmt_se(res.bse["tipuach"])
            if "c_size" in res.params:
                grid.at["Enrollment", c] = self._fmt_coef(res.params["c_size"])
                grid.at["  ", c] = self._fmt_se(res.bse["c_size"])

            grid.at["Root MSE", c] = f"{np.sqrt(res.mse_resid):.2f}"
            grid.at["R2", c] = f"{res.rsquared:.3f}".replace("0.", ".")

        return grid

    def format_ols_table(self, grid):
        """Format OLS grid as great_tables GT object."""
        return (
            GT(grid.reset_index().rename(columns={"index": "Regressors"}))
            .tab_header(title="OLS ESTIMATES FOR 1991")
            .tab_spanner(label="5th Grade", columns=[f"({i})" for i in range(1, 7)])
            .tab_spanner(label="4th Grade", columns=[f"({i})" for i in range(7, 13)])
            .cols_label(Regressors="")
            .cols_align(align="left", columns="Regressors")
            .cols_align(align="center", columns=[f"({i})" for i in range(1, 13)])
            .tab_style(style=style.text(weight="bold"), locations=loc.body(rows=[3, 5, 7]))
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )

    def custom_ols_table(self, other):
        """Generate formatted OLS table for publication."""
        grid = self.build_custom_ols_grid(other)
        return self.format_ols_table(grid)

    def _reduced_form_models(self, df, outcome):
        """Fit two reduced-form specifications: p_size + controls, and with enrollment."""
        specs = [
            f"{outcome} ~ p_size + tipuach",
            f"{outcome} ~ p_size + tipuach + c_size",
        ]
        return [sm.OLS.from_formula(s, data=df).fit() for s in specs]

    def _build_reduced_form_panel_rows(self, left_df, right_df, panel_title):
        """Build rows for reduced-form panel comparing left/right dataframes."""
        cols = [f"({i})" for i in range(1, 13)]
        rows = []

        pairs = [
            ("5_class", left_df, "classize", "(1)", "(2)"),
            ("5_read", left_df, "avgverb", "(3)", "(4)"),
            ("5_math", left_df, "avgmath", "(5)", "(6)"),
            ("4_class", right_df, "classize", "(7)", "(8)"),
            ("4_read", right_df, "avgverb", "(9)", "(10)"),
            ("4_math", right_df, "avgmath", "(11)", "(12)"),
        ]

        base_rows = [
            panel_title, "Means", "(s.d.)", "f_sc", "",
            "Percent disadvantaged", " ", "Enrollment", "  ",
            "Root MSE", "R2", "N",
        ]
        for label in base_rows:
            row = {"Regressors": label}
            row.update({c: "" for c in cols})
            rows.append(row)

        row_map = {r["Regressors"]: r for r in rows}

        for _, df_use, outcome, c1, c2 in pairs:
            m1, m2 = self._reduced_form_models(df_use, outcome)

            row_map["Means"][c2] = f"{df_use[outcome].mean():.1f}"
            row_map["(s.d.)"][c2] = f"({df_use[outcome].std():.1f})"
            row_map["N"][c2] = f"{int(m2.nobs):,}"

            # Extract p_size parameter
            coef, se = self._extract_param(m1.params, m1.bse, "p_size")
            row_map["f_sc"][c1] = coef
            row_map[""][c1] = se
            coef, se = self._extract_param(m2.params, m2.bse, "p_size")
            row_map["f_sc"][c2] = coef
            row_map[""][c2] = se

            # Extract tipuach parameter
            coef, se = self._extract_param(m1.params, m1.bse, "tipuach")
            row_map["Percent disadvantaged"][c1] = coef
            row_map[" "][c1] = se
            coef, se = self._extract_param(m2.params, m2.bse, "tipuach")
            row_map["Percent disadvantaged"][c2] = coef
            row_map[" "][c2] = se

            # Extract c_size parameter (only in spec 2)
            coef, se = self._extract_param(m2.params, m2.bse, "c_size")
            row_map["Enrollment"][c2] = coef
            row_map["  "][c2] = se

            row_map["Root MSE"][c1] = f"{np.sqrt(m1.mse_resid):.2f}"
            row_map["Root MSE"][c2] = f"{np.sqrt(m2.mse_resid):.2f}"
            row_map["R2"][c1] = f"{m1.rsquared:.3f}".replace("0.", ".")
            row_map["R2"][c2] = f"{m2.rsquared:.3f}".replace("0.", ".")

        return rows

    def build_reduced_form_grid(self, other):
        """Build full reduced-form grid (full sample and discontinuity sample)."""
        full_panel = self._build_reduced_form_panel_rows(self.df, other.df, "A. Full sample")
        disc_left = self.df.query(self.DISC_QUERY).copy()
        disc_right = other.df.query(self.DISC_QUERY).copy()
        disc_panel = self._build_reduced_form_panel_rows(disc_left, disc_right, "B. Discontinuity sample")

        return pd.DataFrame(full_panel + disc_panel)

    def format_reduced_form_table(self, grid):
        """Format reduced-form grid as GT object with nested spanners."""
        cols = [f"({i})" for i in range(1, 13)]
        return (
            GT(grid)
            .tab_header(title="REDUCED-FORM ESTIMATES FOR 1991")
            .tab_spanner(label="5th Graders", columns=[f"({i})" for i in range(1, 7)])
            .tab_spanner(label="4th Graders", columns=[f"({i})" for i in range(7, 13)])
            .tab_spanner(label="Class size", columns=["(1)", "(2)"], id="class_size_5")
            .tab_spanner(label="Reading", columns=["(3)", "(4)"], id="reading_5")
            .tab_spanner(label="Math", columns=["(5)", "(6)"], id="math_5")
            .tab_spanner(label="Class size", columns=["(7)", "(8)"], id="class_size_4")
            .tab_spanner(label="Reading", columns=["(9)", "(10)"], id="reading_4")
            .tab_spanner(label="Math", columns=["(11)", "(12)"], id="math_4")
            .cols_label(Regressors="")
            .cols_align(align="left", columns="Regressors")
            .cols_align(align="center", columns=cols)
            .tab_style(style=style.text(weight="bold"), locations=loc.body(rows=[0, 12]))
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )

    def custom_reduced_form_table(self, other):
        """Generate formatted reduced-form table for publication."""
        grid = self.build_reduced_form_grid(other)
        return self.format_reduced_form_table(grid)

    # ========== 2SLS Table ==========

    def _prep_iv_df(self, df):
        """Prepare IV dataframe: add squared enrollment and segment/trend variables."""
        df = df.copy()
        df['c_size_sq100'] = df['c_size'] ** 2 / 100
        df['seg'] = np.floor((df['c_size'] - 1) / 40).astype(int)
        df['plin'] = self._piecewise_linear_trend(df['c_size'])
        return df

    def _run_iv(self, df, outcome, exog_vars):
        """Fit 2SLS: classize instrumented by p_size with given exogenous variables."""
        df = df.copy()
        df['Intercept'] = 1.0
        # Expand categorical segment dummies if needed
        if 'C(seg)' in exog_vars:
            seg_dummies = pd.get_dummies(df['seg'], prefix='seg', drop_first=True).astype(float)
            df = pd.concat([df, seg_dummies], axis=1)
            exog_cols = ['Intercept'] + [c for c in seg_dummies.columns] + [v for v in exog_vars if v != 'C(seg)']
        else:
            exog_cols = ['Intercept'] + exog_vars
        needed = [outcome, 'classize', 'p_size'] + exog_cols
        df = df[needed].dropna()
        model = IV2SLS(df[outcome], df[exog_cols], df[['classize']], df[['p_size']]).fit(cov_type='unadjusted')
        return model

    def _iv_models(self, df, outcome):
        """Generate suite of IV specifications across full and discontinuity samples."""
        full = self._prep_iv_df(df)
        disc = self._prep_iv_df(df.query(self.DISC_QUERY).copy())
        return [
            self._run_iv(full, outcome, ['tipuach']),
            self._run_iv(full, outcome, ['tipuach', 'c_size']),
            self._run_iv(full, outcome, ['tipuach', 'c_size', 'c_size_sq100']),
            self._run_iv(full, outcome, ['plin']),
            self._run_iv(disc, outcome, ['tipuach']),
            self._run_iv(disc, outcome, ['tipuach', 'c_size']),
        ]

    def build_twoSLS_grid(self):
        """Build grid for 2SLS table across outcomes and specifications."""
        cols = [f'({i})' for i in range(1, 13)]
        verb_models = self._iv_models(self.df, 'avgverb')
        math_models = self._iv_models(self.df, 'avgmath')
        all_models = dict(zip(cols, verb_models + math_models))

        disc_df = self.df.query(self.DISC_QUERY)

        rows = [
            'Mean score', '(s.d.)', 'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Enrollment', '  ',
            'Enrollment squared/100', '   ',
            'Piecewise linear trend', '    ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        # Mean/SD/N at block centers
        block_stats = [
            (self.df['avgverb'],  '(2)'),
            (disc_df['avgverb'],  '(5)'),
            (self.df['avgmath'],  '(9)'),
            (disc_df['avgmath'],  '(11)'),
        ]
        for series, display_col in block_stats:
            grid.at['Mean score', display_col] = f'{series.mean():.1f}'
            grid.at['(s.d.)', display_col] = f'({series.std():.1f})'

        # Fill coefficients/SEs
        param_map = {
            'classize':      ('Class size',              ''),
            'tipuach':       ('Percent disadvantaged',   ' '),
            'c_size':        ('Enrollment',              '  '),
            'c_size_sq100':  ('Enrollment squared/100',  '   '),
            'plin':          ('Piecewise linear trend',  '    '),
        }
        for col, res in all_models.items():
            params = res.params
            std_errors = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                if var in params.index:
                    grid.at[row_coef, col] = self._fmt_coef(params[var])
                    grid.at[row_se,   col] = self._fmt_se(std_errors[var])
            grid.at['Root MSE', col] = f'{np.sqrt(float(res.resids.T @ res.resids) / res.df_resid):.2f}'

        # N at block centres
        n_block = [
            (verb_models[1], '(2)'),
            (verb_models[4], '(5)'),
            (math_models[1], '(9)'),
            (math_models[4], '(11)'),
        ]
        for res, display_col in n_block:
            grid.at['N', display_col] = f'{int(res.nobs):,}'

        return grid

    def format_twoSLS_table(self, grid):
        """Format 2SLS grid as GT object."""
        grade = self.grade
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title=f'2SLS ESTIMATES FOR 1991 ({grade}TH GRADERS)')
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(1, 7)])
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(7, 13)])
            .tab_spanner(label='Full sample',           columns=[f'({i})' for i in range(1, 5)],  id='full_verb')
            .tab_spanner(label='+/- 5 Discontinuity sample', columns=['(5)', '(6)'],              id='disc_verb')
            .tab_spanner(label='Full sample',           columns=[f'({i})' for i in range(7, 11)], id='full_math')
            .tab_spanner(label='+/- 5 Discontinuity sample', columns=['(11)', '(12)'],             id='disc_math')
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=[f'({i})' for i in range(1, 13)])
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0, 1, 2]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[3, 5, 7, 9, 11]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_twoSLS_table(self):
        """Generate formatted 2SLS table for publication."""
        grid = self.build_twoSLS_grid()
        return self.format_twoSLS_table(grid)

    # ========== Dummy-Instrument Table ==========

    def _prep_dummy_iv_df(self, df):
        """Prepare DF for dummy IV: segment membership and above-cutoff dummies."""
        df = df.copy()
        df['seg1_dummy'] = ((df['c_size'] >= 36) & (df['c_size'] <= 45)).astype(float)
        df['seg2_dummy'] = ((df['c_size'] >= 76) & (df['c_size'] <= 85)).astype(float)
        df['d1'] = ((df['c_size'] >= 41) & (df['c_size'] <= 45)).astype(float)
        df['d2'] = ((df['c_size'] >= 81) & (df['c_size'] <= 85)).astype(float)
        df['d3'] = ((df['c_size'] >= 121) & (df['c_size'] <= 125)).astype(float)
        return df

    def _run_dummy_iv(self, df, outcome, include_tipuach):
        """Fit 2SLS with dummy instruments and segment controls."""
        df = df.copy()
        df['Intercept'] = 1.0
        exog_cols = ['Intercept', 'seg1_dummy', 'seg2_dummy']
        if include_tipuach:
            exog_cols.append('tipuach')
        needed = [outcome, 'classize'] + exog_cols + ['d1', 'd2', 'd3']
        df = df[needed].dropna()
        return IV2SLS(df[outcome], df[exog_cols], df[['classize']], df[['d1', 'd2', 'd3']]).fit(cov_type='unadjusted')

    def build_dummy_iv_grid(self, other):
        """Build grid for dummy-instrument estimates across samples and outcomes."""
        s5_pm5 = self._prep_dummy_iv_df(self.df.query(self.PM5_QUERY).copy())
        s5_pm3 = self._prep_dummy_iv_df(self.df.query(self.PM3_QUERY).copy())
        s4_pm5 = self._prep_dummy_iv_df(other.df.query(self.PM5_QUERY).copy())
        s4_pm3 = self._prep_dummy_iv_df(other.df.query(self.PM3_QUERY).copy())

        run = self._run_dummy_iv
        cols = [f'({i})' for i in range(1, 13)]
        models = dict(zip(cols, [
            run(s5_pm5, 'avgverb', True),   # (1)
            run(s5_pm3, 'avgverb', True),   # (2)
            run(s5_pm3, 'avgverb', False),  # (3)
            run(s5_pm5, 'avgmath', True),   # (4)
            run(s5_pm3, 'avgmath', True),   # (5)
            run(s5_pm3, 'avgmath', False),  # (6)
            run(s4_pm5, 'avgverb', True),   # (7)
            run(s4_pm3, 'avgverb', True),   # (8)
            run(s4_pm3, 'avgverb', False),  # (9)
            run(s4_pm5, 'avgmath', True),   # (10)
            run(s4_pm3, 'avgmath', True),   # (11)
            run(s4_pm3, 'avgmath', False),  # (12)
        ]))

        rows = [
            'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Segment 1 (36-45)', '  ',
            'Segment 2 (76-85)', '   ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        param_map = {
            'classize':   ('Class size',            ''),
            'tipuach':    ('Percent disadvantaged',  ' '),
            'seg1_dummy': ('Segment 1 (36-45)',      '  '),
            'seg2_dummy': ('Segment 2 (76-85)',      '   '),
        }
        for col, res in models.items():
            params = res.params
            ses = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                coef, se = self._extract_param(params, ses, var)
                grid.at[row_coef, col] = coef
                grid.at[row_se, col] = se
            grid.at['Root MSE', col] = f'{np.sqrt(float(res.resids.T @ res.resids) / res.df_resid):.2f}'

        for col in ['(1)', '(2)', '(4)', '(5)', '(7)', '(8)', '(10)', '(11)']:
            grid.at['N', col] = f'{int(models[col].nobs):,}'

        return grid

    def format_dummy_iv_table(self, grid):
        """Format dummy-IV grid as GT object."""
        cols = [f'({i})' for i in range(1, 13)]
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title='DUMMY-INSTRUMENT RESULTS FOR DISCONTINUITY SAMPLES')
            .tab_spanner(label='5th grade', columns=[f'({i})' for i in range(1, 7)])
            .tab_spanner(label='4th grade', columns=[f'({i})' for i in range(7, 13)])
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(1, 4)],  id='rc_5')
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(4, 7)],  id='m_5')
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(7, 10)], id='rc_4')
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(10, 13)],id='m_4')
            .tab_spanner(label='+/- 5 Sample', columns=['(1)'],         id='pm5_rc5')
            .tab_spanner(label='+/- 3 Sample', columns=['(2)', '(3)'],  id='pm3_rc5')
            .tab_spanner(label='+/- 5 Sample', columns=['(4)'],         id='pm5_m5')
            .tab_spanner(label='+/- 3 Sample', columns=['(5)', '(6)'],  id='pm3_m5')
            .tab_spanner(label='+/- 5 Sample', columns=['(7)'],         id='pm5_rc4')
            .tab_spanner(label='+/- 3 Sample', columns=['(8)', '(9)'],  id='pm3_rc4')
            .tab_spanner(label='+/- 5 Sample', columns=['(10)'],        id='pm5_m4')
            .tab_spanner(label='+/- 3 Sample', columns=['(11)', '(12)'],id='pm3_m4')
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=cols)
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[1, 3, 5, 7]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_dummy_iv_table(self, other):
        """Generate formatted dummy-IV table for publication."""
        grid = self.build_dummy_iv_grid(other)
        return self.format_dummy_iv_table(grid)

    # ========== Pooled Interaction Table ==========

    def _run_interaction_iv(self, df, outcome, with_interaction, grade4_dummy=False):
        """Fit 2SLS with optional interaction: classize * percent disadvantaged."""
        df = df.copy()
        extra = ['grade4'] if grade4_dummy else []
        base_controls = ['tipuach', 'c_size'] + extra
        needed = [outcome, 'classize', 'p_size'] + base_controls
        df = df[needed].dropna()

        # Stage 1: classize on p_size + controls
        X1 = sm.add_constant(df[base_controls + ['p_size']])
        fs = sm.OLS(df['classize'], X1).fit()
        df['classize_hat'] = fs.fittedvalues
        df['classize_hat_pd'] = df['classize_hat'] * df['tipuach']
        df['classize_pd'] = df['classize'] * df['tipuach']

        # Stage 2: OLS with fitted classize [+ fitted interaction] + controls
        if with_interaction:
            stage2_vars = ['classize_hat', 'classize_hat_pd'] + base_controls
        else:
            stage2_vars = ['classize_hat'] + base_controls
        X2 = sm.add_constant(df[stage2_vars])
        ss = sm.OLS(df[outcome], X2).fit()

        # Structural residuals using true classize
        if with_interaction:
            fitted_structural = (
                ss.params['const']
                + ss.params['classize_hat']    * df['classize']
                + ss.params['classize_hat_pd'] * df['classize_pd']
                + sum(ss.params[v] * df[v] for v in base_controls)
            )
        else:
            fitted_structural = (
                ss.params['const']
                + ss.params['classize_hat'] * df['classize']
                + sum(ss.params[v] * df[v] for v in base_controls)
            )
        resids = df[outcome] - fitted_structural
        rmse = float(np.sqrt((resids ** 2).sum() / (len(df) - len(ss.params))))

        # Package as simple result object
        class _IVResult:
            pass
        r = _IVResult()
        r.params = ss.params.rename({'classize_hat': 'classize', 'classize_hat_pd': 'classize_pd'})
        r.std_errors = ss.bse.rename({'classize_hat': 'classize', 'classize_hat_pd': 'classize_pd'})
        r.rmse = rmse
        r.nobs = len(df)
        return r

    def build_pooled_interaction_grid(self, other):
        """Build grid for pooled models with optional interaction terms."""
        pooled = pd.concat([
            self.df.assign(grade4=0.0),
            other.df.assign(grade4=1.0),
        ], ignore_index=True)

        run = self._run_interaction_iv
        cols = [f'({i})' for i in range(1, 9)]
        models = dict(zip(cols, [
            run(self.df,  'avgverb', with_interaction=True),
            run(self.df,  'avgmath', with_interaction=True),
            run(other.df, 'avgverb', with_interaction=True),
            run(other.df, 'avgmath', with_interaction=True),
            run(pooled,   'avgverb', with_interaction=False, grade4_dummy=True),
            run(pooled,   'avgverb', with_interaction=True,  grade4_dummy=True),
            run(pooled,   'avgmath', with_interaction=False, grade4_dummy=True),
            run(pooled,   'avgmath', with_interaction=True,  grade4_dummy=True),
        ]))

        rows = [
            'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Grade 4', '  ',
            'Enrollment', '   ',
            'Interaction',
            'Class size*PD', '    ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        param_map = {
            'classize':     ('Class size',            ''),
            'tipuach':      ('Percent disadvantaged', ' '),
            'grade4':       ('Grade 4',               '  '),
            'c_size':       ('Enrollment',             '   '),
            'classize_pd':  ('Class size*PD',          '    '),
        }
        for col, res in models.items():
            params = res.params
            ses    = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                coef, se = self._extract_param(params, ses, var)
                grid.at[row_coef, col] = coef
                grid.at[row_se, col] = se
            grid.at['Root MSE', col] = f'{res.rmse:.2f}'

        # N: individual for cols 1-4, block centres for pooled
        for col in ['(1)', '(2)', '(3)', '(4)']:
            grid.at['N', col] = f'{int(models[col].nobs):,}'
        grid.at['N', '(5)'] = f'{int(models["(5)"].nobs):,}'
        grid.at['N', '(7)'] = f'{int(models["(7)"].nobs):,}'

        return grid

    def format_pooled_interaction_table(self, grid):
        """Format pooled interaction grid as GT object."""
        cols = [f'({i})' for i in range(1, 9)]
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title='POOLED ESTIMATES AND MODELS WITH PERCENT DISADVANTAGED INTERACTION TERMS')
            .tab_spanner(label='5th grade',        columns=['(1)', '(2)'])
            .tab_spanner(label='4th grade',        columns=['(3)', '(4)'])
            .tab_spanner(label='Pooled estimates', columns=[f'({i})' for i in range(5, 9)])
            .tab_spanner(label='Reading',          columns=['(5)', '(6)'], id='pool_read')
            .tab_spanner(label='Math',             columns=['(7)', '(8)'], id='pool_math')
            .cols_label(**{
                '(1)': 'Reading\n(1)', '(2)': 'Math\n(2)',
                '(3)': 'Reading\n(3)', '(4)': 'Math\n(4)',
                '(5)': '(5)', '(6)': '(6)', '(7)': '(7)', '(8)': '(8)',
            })
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=cols)
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0, 9]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[1, 3, 5, 7, 10]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_pooled_interaction_table(self, other):
        """Generate formatted pooled interaction table for publication."""
        grid = self.build_pooled_interaction_grid(other)
        return self.format_pooled_interaction_table(grid)


    @staticmethod
    def _piecewise_linear_trend(enrollment):
        """Piecewise-linear trend used in the IV specification."""
        e = pd.Series(enrollment, copy=False)
        trend = pd.Series(np.nan, index=e.index, dtype=float)

        m1 = (e >= 0) & (e <= 40)
        m2 = (e >= 41) & (e <= 80)
        m3 = (e >= 81) & (e <= 120)
        m4 = (e >= 121) & (e <= 160)

        trend.loc[m1] = e.loc[m1]
        trend.loc[m2] = 20 + (e.loc[m2] / 2)
        trend.loc[m3] = (100 / 3) + (e.loc[m3] / 3)
        trend.loc[m4] = (130 / 3) + (e.loc[m4] / 4)
        return trend

    def __init__(self, df):
        
        super().__init__(df)
        self.col = ["classize", "c_size", "tipuach", "verbsize", "mathsize", "avgverb", "avgmath"]
        self.row_index = ["Class size", "Enrollment", "Percent disadvantaged", "Reading size", "Math size", "Average verbal", "Average math"]
        self.label_sc = f"{self.grade}th grade ({self.n_classes} classes, {self.n_schools} schools, tested in 1991)"

    @property
    def row(self):
        return self.df[self.col]



    def descriptive_table(self, other):
        num_cols = ["mean", "std", "q10", "q25", "q50", "q75", "q90"]

        def compute_stats(df):
            row = df[self.col]
            t = pd.DataFrame({
                "mean": row.mean(),
                "std": row.std(),
                "q10": row.quantile(0.1),
                "q25": row.quantile(0.25),
                "q50": row.quantile(0.5),
                "q75": row.quantile(0.75),
                "q90": row.quantile(0.9),
            })
            t.index = self.row_index
            t.index.name = "Variable"
            t = t.round(1).astype(object)
            t.iloc[:5, -5:] = t.iloc[:5, -5:].astype(float).round(0).astype(int)
            return t.reset_index()

        def header_row(label):
            return {"Variable": label, **{c: "" for c in num_cols}}

        grid = pd.concat([
            pd.DataFrame([header_row("A. Full sample")]),
            pd.DataFrame([header_row(self.label_sc)]),
            compute_stats(self.df),
            pd.DataFrame([header_row("")]),
            pd.DataFrame([header_row(other.label_sc)]),
            compute_stats(other.df),
        ], ignore_index=True)

        return (
            GT(grid)
            .tab_header(title="UNWEIGHTED DESCRIPTIVE STATISTICS")
            .tab_spanner(label="Quantiles", columns=["q10", "q25", "q50", "q75", "q90"])
            .cols_label(
                Variable="Variable",
                mean="Mean",
                std="S.D.",
                q10="0.10",
                q25="0.25",
                q50="0.50",
                q75="0.75",
                q90="0.90",
            )
            .cols_align(align="left", columns="Variable")
            .cols_align(align="center", columns=num_cols)
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )
    


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
    


    def ols_estimate(self, regressor):
        """Run OLS with multiple specifications."""
        specs = [f"{regressor} ~ classize",
                 f"{regressor} ~ classize + tipuach",
                 f"{regressor} ~ classize + tipuach + c_size"]
        models = [sm.OLS.from_formula(s, data=self.df).fit() for s in specs]
        return models



    @staticmethod
    def _fmt_coef(value):
        return f"{value:.3f}".replace("0.", ".").replace("-0.", "-.")

    @staticmethod
    def _fmt_se(value):
        return f"({value:.3f})".replace("0.", ".").replace("-0.", "-.")

    def build_custom_ols_grid(self, other):
        """Manually fill every table cell for the 12-column OLS layout."""
        models = {
            "(1)": self.ols_estimate("avgverb")[0],
            "(2)": self.ols_estimate("avgverb")[1],
            "(3)": self.ols_estimate("avgverb")[2],
            "(4)": self.ols_estimate("avgmath")[0],
            "(5)": self.ols_estimate("avgmath")[1],
            "(6)": self.ols_estimate("avgmath")[2],
            "(7)": other.ols_estimate("avgverb")[0],
            "(8)": other.ols_estimate("avgverb")[1],
            "(9)": other.ols_estimate("avgverb")[2],
            "(10)": other.ols_estimate("avgmath")[0],
            "(11)": other.ols_estimate("avgmath")[1],
            "(12)": other.ols_estimate("avgmath")[2],
        }

        cols = [f"({i})" for i in range(1, 13)]
        rows = [
            "Mean score",
            "(s.d.)",
            "Regressors",
            "Class size",
            "",
            "Percent disadvantaged",
            " ",
            "Enrollment",
            "  ",
            "Root MSE",
            "R2",
            "N",
        ]
        grid = pd.DataFrame("", index=rows, columns=cols)

        # Aggregate Mean/(s.d.) by 3-column blocks and place once at block center.
        block_stats = [
            (self.df["avgverb"].mean(), self.df["avgverb"].std(), "(2)"),
            (self.df["avgmath"].mean(), self.df["avgmath"].std(), "(5)"),
            (other.df["avgverb"].mean(), other.df["avgverb"].std(), "(8)"),
            (other.df["avgmath"].mean(), other.df["avgmath"].std(), "(11)"),
        ]
        for mean_val, sd_val, display_col in block_stats:
            grid.at["Mean score", display_col] = f"{mean_val:.1f}"
            grid.at["(s.d.)", display_col] = f"({sd_val:.1f})"

        # Aggregate N by the same 3-column blocks and place once at block center.
        n_by_block = {
            "(2)": int(models["(1)"].nobs),
            "(5)": int(models["(4)"].nobs),
            "(8)": int(models["(7)"].nobs),
            "(11)": int(models["(10)"].nobs),
        }
        for display_col, n_val in n_by_block.items():
            grid.at["N", display_col] = f"{n_val:,}"

        # Fill coefficients/SEs and bottom stats column by column.
        for c, res in models.items():
            if "classize" in res.params:
                grid.at["Class size", c] = self._fmt_coef(res.params["classize"])
                grid.at["", c] = self._fmt_se(res.bse["classize"])
            if "tipuach" in res.params:
                grid.at["Percent disadvantaged", c] = self._fmt_coef(res.params["tipuach"])
                grid.at[" ", c] = self._fmt_se(res.bse["tipuach"])
            if "c_size" in res.params:
                grid.at["Enrollment", c] = self._fmt_coef(res.params["c_size"])
                grid.at["  ", c] = self._fmt_se(res.bse["c_size"])

            grid.at["Root MSE", c] = f"{np.sqrt(res.mse_resid):.2f}"
            grid.at["R2", c] = f"{res.rsquared:.3f}".replace("0.", ".")

        return grid

    def format_ols_table(self, grid):
        return (
            GT(grid.reset_index().rename(columns={"index": "Regressors"}))
            .tab_header(title="OLS ESTIMATES FOR 1991")
            .tab_spanner(label="5th Grade", columns=[f"({i})" for i in range(1, 7)])
            .tab_spanner(label="4th Grade", columns=[f"({i})" for i in range(7, 13)])
            .cols_label(Regressors="")
            .cols_align(align="left", columns="Regressors")
            .cols_align(align="center", columns=[f"({i})" for i in range(1, 13)])
            .tab_style(style=style.text(weight="bold"), locations=loc.body(rows=[3, 5, 7]))
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )

    def custom_ols_table(self, other):
        grid = self.build_custom_ols_grid(other)
        return self.format_ols_table(grid)

    def _reduced_form_models(self, df, outcome):
        specs = [
            f"{outcome} ~ p_size + tipuach",
            f"{outcome} ~ p_size + tipuach + c_size",
        ]
        return [sm.OLS.from_formula(s, data=df).fit() for s in specs]

    def _build_reduced_form_panel_rows(self, left_df, right_df, panel_title):
        cols = [f"({i})" for i in range(1, 13)]
        rows = []

        # Pair layout: (label, df, outcome, col1, col2)
        pairs = [
            ("5_class", left_df, "classize", "(1)", "(2)"),
            ("5_read", left_df, "avgverb", "(3)", "(4)"),
            ("5_math", left_df, "avgmath", "(5)", "(6)"),
            ("4_class", right_df, "classize", "(7)", "(8)"),
            ("4_read", right_df, "avgverb", "(9)", "(10)"),
            ("4_math", right_df, "avgmath", "(11)", "(12)"),
        ]

        # Prepare empty rows for this panel.
        base_rows = [
            panel_title,
            "Means",
            "(s.d.)",
            "f_sc",
            "",
            "Percent disadvantaged",
            " ",
            "Enrollment",
            "  ",
            "Root MSE",
            "R2",
            "N",
        ]
        for label in base_rows:
            row = {"Regressors": label}
            row.update({c: "" for c in cols})
            rows.append(row)

        row_map = {r["Regressors"]: r for r in rows}

        for _, df_use, outcome, c1, c2 in pairs:
            m1, m2 = self._reduced_form_models(df_use, outcome)

            # Aggregate Means/(s.d.) and N once per 2-column block.
            row_map["Means"][c2] = f"{df_use[outcome].mean():.1f}"
            row_map["(s.d.)"][c2] = f"({df_use[outcome].std():.1f})"
            row_map["N"][c2] = f"{int(m2.nobs):,}"

            # f_sc is p_size in the data.
            row_map["f_sc"][c1] = self._fmt_coef(m1.params.get("p_size", np.nan)) if "p_size" in m1.params else ""
            row_map[""][c1] = self._fmt_se(m1.bse.get("p_size", np.nan)) if "p_size" in m1.bse else ""
            row_map["f_sc"][c2] = self._fmt_coef(m2.params.get("p_size", np.nan)) if "p_size" in m2.params else ""
            row_map[""][c2] = self._fmt_se(m2.bse.get("p_size", np.nan)) if "p_size" in m2.bse else ""

            # Percent disadvantaged appears in both specs.
            row_map["Percent disadvantaged"][c1] = self._fmt_coef(m1.params.get("tipuach", np.nan)) if "tipuach" in m1.params else ""
            row_map[" "][c1] = self._fmt_se(m1.bse.get("tipuach", np.nan)) if "tipuach" in m1.bse else ""
            row_map["Percent disadvantaged"][c2] = self._fmt_coef(m2.params.get("tipuach", np.nan)) if "tipuach" in m2.params else ""
            row_map[" "][c2] = self._fmt_se(m2.bse.get("tipuach", np.nan)) if "tipuach" in m2.bse else ""

            # Enrollment only in spec 2.
            row_map["Enrollment"][c2] = self._fmt_coef(m2.params.get("c_size", np.nan)) if "c_size" in m2.params else ""
            row_map["  "][c2] = self._fmt_se(m2.bse.get("c_size", np.nan)) if "c_size" in m2.bse else ""

            # Bottom stats by column.
            row_map["Root MSE"][c1] = f"{np.sqrt(m1.mse_resid):.2f}"
            row_map["Root MSE"][c2] = f"{np.sqrt(m2.mse_resid):.2f}"
            row_map["R2"][c1] = f"{m1.rsquared:.3f}".replace("0.", ".")
            row_map["R2"][c2] = f"{m2.rsquared:.3f}".replace("0.", ".")

        return rows

    def build_reduced_form_grid(self, other):
        query_str = (
            "(c_size >= 36 & c_size <= 45) | "
            "(c_size >= 76 & c_size <= 85) | "
            "(c_size >= 116 & c_size <= 125)"
        )

        full_panel = self._build_reduced_form_panel_rows(self.df, other.df, "A. Full sample")
        disc_left = self.df.query(query_str).copy()
        disc_right = other.df.query(query_str).copy()
        disc_panel = self._build_reduced_form_panel_rows(disc_left, disc_right, "B. Discontinuity sample")

        return pd.DataFrame(full_panel + disc_panel)

    def format_reduced_form_table(self, grid):
        cols = [f"({i})" for i in range(1, 13)]
        return (
            GT(grid)
            .tab_header(title="REDUCED-FORM ESTIMATES FOR 1991")
            .tab_spanner(label="5th Graders", columns=[f"({i})" for i in range(1, 7)])
            .tab_spanner(label="4th Graders", columns=[f"({i})" for i in range(7, 13)])
            .tab_spanner(label="Class size", columns=["(1)", "(2)"], id="class_size_5")
            .tab_spanner(label="Reading", columns=["(3)", "(4)"], id="reading_5")
            .tab_spanner(label="Math", columns=["(5)", "(6)"], id="math_5")
            .tab_spanner(label="Class size", columns=["(7)", "(8)"], id="class_size_4")
            .tab_spanner(label="Reading", columns=["(9)", "(10)"], id="reading_4")
            .tab_spanner(label="Math", columns=["(11)", "(12)"], id="math_4")
            .cols_label(Regressors="")
            .cols_align(align="left", columns="Regressors")
            .cols_align(align="center", columns=cols)
            .tab_style(style=style.text(weight="bold"), locations=loc.body(rows=[0, 12]))
            .tab_options(
                table_border_top_style="double",
                table_border_bottom_style="double",
                heading_border_bottom_style="solid",
                heading_border_bottom_width="2px",
            )
        )

    def custom_reduced_form_table(self, other):
        grid = self.build_reduced_form_grid(other)
        return self.format_reduced_form_table(grid)

    # ------------------------------------------------------------------
    # 2SLS table
    # ------------------------------------------------------------------

    def _prep_iv_df(self, df):
        df = df.copy()
        df['c_size_sq100'] = df['c_size'] ** 2 / 100
        df['seg'] = np.floor((df['c_size'] - 1) / 40).astype(int)
        df['plin'] = self._piecewise_linear_trend(df['c_size'])
        return df

    def _run_iv(self, df, outcome, exog_vars):
        """2SLS: classize instrumented by p_size."""
        df = df.copy()
        df['Intercept'] = 1.0
        # If exog_vars contains C(seg), expand dummies manually
        if 'C(seg)' in exog_vars:
            seg_dummies = pd.get_dummies(df['seg'], prefix='seg', drop_first=True).astype(float)
            df = pd.concat([df, seg_dummies], axis=1)
            exog_cols = ['Intercept'] + [c for c in seg_dummies.columns] + [v for v in exog_vars if v != 'C(seg)']
        else:
            exog_cols = ['Intercept'] + exog_vars
        needed = [outcome, 'classize', 'p_size'] + exog_cols
        df = df[needed].dropna()
        model = IV2SLS(df[outcome], df[exog_cols], df[['classize']], df[['p_size']]).fit(cov_type='unadjusted')
        return model

    def _iv_models(self, df, outcome):
        disc_query = (
            "(c_size >= 36 & c_size <= 45) | "
            "(c_size >= 76 & c_size <= 85) | "
            "(c_size >= 116 & c_size <= 125)"
        )
        full = self._prep_iv_df(df)
        disc = self._prep_iv_df(df.query(disc_query).copy())
        return [
            self._run_iv(full, outcome, ['tipuach']),
            self._run_iv(full, outcome, ['tipuach', 'c_size']),
            self._run_iv(full, outcome, ['tipuach', 'c_size', 'c_size_sq100']),
            self._run_iv(full, outcome, ['plin']),
            self._run_iv(disc, outcome, ['tipuach']),
            self._run_iv(disc, outcome, ['tipuach', 'c_size']),
        ]

    def build_twoSLS_grid(self):
        cols = [f'({i})' for i in range(1, 13)]

        verb_models = self._iv_models(self.df, 'avgverb')
        math_models = self._iv_models(self.df, 'avgmath')
        all_models = dict(zip(cols, verb_models + math_models))

        disc_query = (
            "(c_size >= 36 & c_size <= 45) | "
            "(c_size >= 76 & c_size <= 85) | "
            "(c_size >= 116 & c_size <= 125)"
        )
        disc_df = self.df.query(disc_query)

        rows = [
            'Mean score', '(s.d.)', 'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Enrollment', '  ',
            'Enrollment squared/100', '   ',
            'Piecewise linear trend', '    ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        # Mean score / (s.d.) / N — centred in middle of each block
        block_stats = [
            (self.df['avgverb'],  '(2)'),   # Reading full (cols 1-4)
            (disc_df['avgverb'],  '(5)'),   # Reading disc (cols 5-6)
            (self.df['avgmath'],  '(9)'),   # Math full    (cols 7-10)
            (disc_df['avgmath'],  '(11)'),  # Math disc    (cols 11-12)
        ]
        n_cols = {'(2)': '(2)', '(5)': '(5)', '(9)': '(9)', '(11)': '(11)'}
        for series, display_col in block_stats:
            grid.at['Mean score', display_col] = f'{series.mean():.1f}'
            grid.at['(s.d.)', display_col] = f'({series.std():.1f})'

        # Fill coefficients and SEs
        param_map = {
            'classize':      ('Class size',              ''),
            'tipuach':       ('Percent disadvantaged',   ' '),
            'c_size':        ('Enrollment',              '  '),
            'c_size_sq100':  ('Enrollment squared/100',  '   '),
            'plin':          ('Piecewise linear trend',  '    '),
        }
        for col, res in all_models.items():
            params = res.params
            std_errors = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                if var in params.index:
                    grid.at[row_coef, col] = self._fmt_coef(params[var])
                    grid.at[row_se,   col] = self._fmt_se(std_errors[var])
            grid.at['Root MSE', col] = f'{np.sqrt(float(res.resids.T @ res.resids) / res.df_resid):.2f}'

        # N — same block centres as Mean score
        n_block = [
            (verb_models[1], '(2)'),
            (verb_models[4], '(5)'),
            (math_models[1], '(9)'),
            (math_models[4], '(11)'),
        ]
        for res, display_col in n_block:
            grid.at['N', display_col] = f'{int(res.nobs):,}'

        return grid

    def format_twoSLS_table(self, grid):
        grade = self.grade
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title=f'2SLS ESTIMATES FOR 1991 ({grade}TH GRADERS)')
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(1, 7)])
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(7, 13)])
            .tab_spanner(label='Full sample',           columns=[f'({i})' for i in range(1, 5)],  id='full_verb')
            .tab_spanner(label='+/- 5 Discontinuity sample', columns=['(5)', '(6)'],              id='disc_verb')
            .tab_spanner(label='Full sample',           columns=[f'({i})' for i in range(7, 11)], id='full_math')
            .tab_spanner(label='+/- 5 Discontinuity sample', columns=['(11)', '(12)'],             id='disc_math')
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=[f'({i})' for i in range(1, 13)])
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0, 1, 2]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[3, 5, 7, 9, 11]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_twoSLS_table(self):
        grid = self.build_twoSLS_grid()
        return self.format_twoSLS_table(grid)

    # ------------------------------------------------------------------
    # Dummy-instrument table
    # ------------------------------------------------------------------

    def _prep_dummy_iv_df(self, df):
        df = df.copy()
        # Segment membership dummies (segment 3 = 116-125 is omitted reference)
        df['seg1_dummy'] = ((df['c_size'] >= 36) & (df['c_size'] <= 45)).astype(float)
        df['seg2_dummy'] = ((df['c_size'] >= 76) & (df['c_size'] <= 85)).astype(float)
        # Dummy instruments: above-cutoff within each segment
        df['d1'] = ((df['c_size'] >= 41) & (df['c_size'] <= 45)).astype(float)
        df['d2'] = ((df['c_size'] >= 81) & (df['c_size'] <= 85)).astype(float)
        df['d3'] = ((df['c_size'] >= 121) & (df['c_size'] <= 125)).astype(float)
        return df

    def _run_dummy_iv(self, df, outcome, include_tipuach):
        """2SLS with [d1, d2, d3] as instruments for classize; seg dummies as controls."""
        df = df.copy()
        df['Intercept'] = 1.0
        exog_cols = ['Intercept', 'seg1_dummy', 'seg2_dummy']
        if include_tipuach:
            exog_cols.append('tipuach')
        needed = [outcome, 'classize'] + exog_cols + ['d1', 'd2', 'd3']
        df = df[needed].dropna()
        return IV2SLS(df[outcome], df[exog_cols], df[['classize']], df[['d1', 'd2', 'd3']]).fit(cov_type='unadjusted')

    def build_dummy_iv_grid(self, other):
        pm5_q = "(c_size >= 36 & c_size <= 45) | (c_size >= 76 & c_size <= 85) | (c_size >= 116 & c_size <= 125)"
        pm3_q = "(c_size >= 38 & c_size <= 43) | (c_size >= 78 & c_size <= 83) | (c_size >= 118 & c_size <= 123)"

        s5_pm5 = self._prep_dummy_iv_df(self.df.query(pm5_q).copy())
        s5_pm3 = self._prep_dummy_iv_df(self.df.query(pm3_q).copy())
        s4_pm5 = self._prep_dummy_iv_df(other.df.query(pm5_q).copy())
        s4_pm3 = self._prep_dummy_iv_df(other.df.query(pm3_q).copy())

        run = self._run_dummy_iv
        cols = [f'({i})' for i in range(1, 13)]
        models = dict(zip(cols, [
            run(s5_pm5, 'avgverb', True),   # (1)  5th reading +/-5
            run(s5_pm3, 'avgverb', True),   # (2)  5th reading +/-3 with tipuach
            run(s5_pm3, 'avgverb', False),  # (3)  5th reading +/-3 no tipuach
            run(s5_pm5, 'avgmath', True),   # (4)  5th math +/-5
            run(s5_pm3, 'avgmath', True),   # (5)  5th math +/-3 with tipuach
            run(s5_pm3, 'avgmath', False),  # (6)  5th math +/-3 no tipuach
            run(s4_pm5, 'avgverb', True),   # (7)  4th reading +/-5
            run(s4_pm3, 'avgverb', True),   # (8)  4th reading +/-3 with tipuach
            run(s4_pm3, 'avgverb', False),  # (9)  4th reading +/-3 no tipuach
            run(s4_pm5, 'avgmath', True),   # (10) 4th math +/-5
            run(s4_pm3, 'avgmath', True),   # (11) 4th math +/-3 with tipuach
            run(s4_pm3, 'avgmath', False),  # (12) 4th math +/-3 no tipuach
        ]))

        rows = [
            'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Segment 1 (36-45)', '  ',
            'Segment 2 (76-85)', '   ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        param_map = {
            'classize':   ('Class size',            ''),
            'tipuach':    ('Percent disadvantaged',  ' '),
            'seg1_dummy': ('Segment 1 (36-45)',      '  '),
            'seg2_dummy': ('Segment 2 (76-85)',      '   '),
        }
        for col, res in models.items():
            params = res.params
            ses = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                if var in params.index:
                    grid.at[row_coef, col] = self._fmt_coef(params[var])
                    grid.at[row_se,   col] = self._fmt_se(ses[var])
            grid.at['Root MSE', col] = f'{np.sqrt(float(res.resids.T @ res.resids) / res.df_resid):.2f}'

        for col in ['(1)', '(2)', '(4)', '(5)', '(7)', '(8)', '(10)', '(11)']:
            grid.at['N', col] = f'{int(models[col].nobs):,}'

        return grid

    def format_dummy_iv_table(self, grid):
        cols = [f'({i})' for i in range(1, 13)]
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title='DUMMY-INSTRUMENT RESULTS FOR DISCONTINUITY SAMPLES')
            .tab_spanner(label='5th grade', columns=[f'({i})' for i in range(1, 7)])
            .tab_spanner(label='4th grade', columns=[f'({i})' for i in range(7, 13)])
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(1, 4)],  id='rc_5')
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(4, 7)],  id='m_5')
            .tab_spanner(label='Reading comprehension', columns=[f'({i})' for i in range(7, 10)], id='rc_4')
            .tab_spanner(label='Math',                  columns=[f'({i})' for i in range(10, 13)],id='m_4')
            .tab_spanner(label='+/- 5 Sample', columns=['(1)'],         id='pm5_rc5')
            .tab_spanner(label='+/- 3 Sample', columns=['(2)', '(3)'],  id='pm3_rc5')
            .tab_spanner(label='+/- 5 Sample', columns=['(4)'],         id='pm5_m5')
            .tab_spanner(label='+/- 3 Sample', columns=['(5)', '(6)'],  id='pm3_m5')
            .tab_spanner(label='+/- 5 Sample', columns=['(7)'],         id='pm5_rc4')
            .tab_spanner(label='+/- 3 Sample', columns=['(8)', '(9)'],  id='pm3_rc4')
            .tab_spanner(label='+/- 5 Sample', columns=['(10)'],        id='pm5_m4')
            .tab_spanner(label='+/- 3 Sample', columns=['(11)', '(12)'],id='pm3_m4')
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=cols)
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[1, 3, 5, 7]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_dummy_iv_table(self, other):
        grid = self.build_dummy_iv_grid(other)
        return self.format_dummy_iv_table(grid)

    # ------------------------------------------------------------------
    # Pooled interaction table
    # ------------------------------------------------------------------

    def _run_interaction_iv(self, df, outcome, with_interaction, grade4_dummy=False):
        """Manual 2SLS: stage-1 classize ~ p_size + controls, then interact fitted value with tipuach."""
        df = df.copy()
        extra = ['grade4'] if grade4_dummy else []
        base_controls = ['tipuach', 'c_size'] + extra
        needed = [outcome, 'classize', 'p_size'] + base_controls
        df = df[needed].dropna()

        # Stage 1: classize on p_size + controls
        X1 = sm.add_constant(df[base_controls + ['p_size']])
        fs = sm.OLS(df['classize'], X1).fit()
        df['classize_hat'] = fs.fittedvalues
        df['classize_hat_pd'] = df['classize_hat'] * df['tipuach']
        df['classize_pd'] = df['classize'] * df['tipuach']

        # Stage 2 OLS: use fitted classize [+ fitted interaction] + controls
        if with_interaction:
            stage2_vars = ['classize_hat', 'classize_hat_pd'] + base_controls
        else:
            stage2_vars = ['classize_hat'] + base_controls
        X2 = sm.add_constant(df[stage2_vars])
        ss = sm.OLS(df[outcome], X2).fit()

        # Structural residuals: substitute true classize and interaction back in
        if with_interaction:
            fitted_structural = (
                ss.params['const']
                + ss.params['classize_hat']    * df['classize']
                + ss.params['classize_hat_pd'] * df['classize_pd']
                + sum(ss.params[v] * df[v] for v in base_controls)
            )
        else:
            fitted_structural = (
                ss.params['const']
                + ss.params['classize_hat'] * df['classize']
                + sum(ss.params[v] * df[v] for v in base_controls)
            )
        resids = df[outcome] - fitted_structural
        rmse = float(np.sqrt((resids ** 2).sum() / (len(df) - len(ss.params))))

        # Package result for the grid builder
        class _IVResult:
            pass
        r = _IVResult()
        r.params = ss.params.rename({'classize_hat': 'classize', 'classize_hat_pd': 'classize_pd'})
        r.std_errors = ss.bse.rename({'classize_hat': 'classize', 'classize_hat_pd': 'classize_pd'})
        r.rmse = rmse
        r.nobs = len(df)
        return r

    def build_pooled_interaction_grid(self, other):
        # Stack grades for pooled models, adding grade4 dummy
        pooled = pd.concat([
            self.df.assign(grade4=0.0),
            other.df.assign(grade4=1.0),
        ], ignore_index=True)

        run = self._run_interaction_iv
        cols = [f'({i})' for i in range(1, 9)]
        models = dict(zip(cols, [
            run(self.df,  'avgverb', with_interaction=True),   # (1) 5th reading
            run(self.df,  'avgmath', with_interaction=True),   # (2) 5th math
            run(other.df, 'avgverb', with_interaction=True),   # (3) 4th reading
            run(other.df, 'avgmath', with_interaction=True),   # (4) 4th math
                run(pooled,   'avgverb', with_interaction=False, grade4_dummy=True),  # (5) pooled reading, no interaction
            run(pooled,   'avgverb', with_interaction=True,  grade4_dummy=True),  # (6) pooled reading, with interaction
            run(pooled,   'avgmath', with_interaction=False, grade4_dummy=True),  # (7) pooled math, no interaction
            run(pooled,   'avgmath', with_interaction=True,  grade4_dummy=True),  # (8) pooled math, with interaction
        ]))

        rows = [
            'Regressors',
            'Class size', '',
            'Percent disadvantaged', ' ',
            'Grade 4', '  ',
            'Enrollment', '   ',
            'Interaction',
            'Class size*PD', '    ',
            'Root MSE', 'N',
        ]
        grid = pd.DataFrame('', index=rows, columns=cols)

        param_map = {
            'classize':     ('Class size',            ''),
            'tipuach':      ('Percent disadvantaged', ' '),
            'grade4':       ('Grade 4',               '  '),
            'c_size':       ('Enrollment',             '   '),
            'classize_pd':  ('Class size*PD',          '    '),
        }
        for col, res in models.items():
            params = res.params
            ses    = res.std_errors
            for var, (row_coef, row_se) in param_map.items():
                if var in params.index:
                    grid.at[row_coef, col] = self._fmt_coef(params[var])
                    grid.at[row_se,   col] = self._fmt_se(ses[var])
            grid.at['Root MSE', col] = f'{res.rmse:.2f}'

        # N: individual for cols 1-4, centred per 2-col block for pooled
        for col in ['(1)', '(2)', '(3)', '(4)']:
            grid.at['N', col] = f'{int(models[col].nobs):,}'
        grid.at['N', '(5)'] = f'{int(models["(5)"].nobs):,}'
        grid.at['N', '(7)'] = f'{int(models["(7)"].nobs):,}'

        return grid

    def format_pooled_interaction_table(self, grid):
        cols = [f'({i})' for i in range(1, 9)]
        return (
            GT(grid.reset_index().rename(columns={'index': 'Regressors'}))
            .tab_header(title='POOLED ESTIMATES AND MODELS WITH PERCENT DISADVANTAGED INTERACTION TERMS')
            .tab_spanner(label='5th grade',        columns=['(1)', '(2)'])
            .tab_spanner(label='4th grade',        columns=['(3)', '(4)'])
            .tab_spanner(label='Pooled estimates', columns=[f'({i})' for i in range(5, 9)])
            .tab_spanner(label='Reading',          columns=['(5)', '(6)'], id='pool_read')
            .tab_spanner(label='Math',             columns=['(7)', '(8)'], id='pool_math')
            .cols_label(**{
                '(1)': 'Reading\n(1)', '(2)': 'Math\n(2)',
                '(3)': 'Reading\n(3)', '(4)': 'Math\n(4)',
                '(5)': '(5)', '(6)': '(6)', '(7)': '(7)', '(8)': '(8)',
            })
            .cols_label(Regressors='')
            .cols_align(align='left',   columns='Regressors')
            .cols_align(align='center', columns=cols)
            .tab_style(style=style.text(style='italic'), locations=loc.body(rows=[0, 9]))
            .tab_style(style=style.css('padding-left: 20px'), locations=loc.body(rows=[1, 3, 5, 7, 10]))
            .tab_options(
                table_border_top_style='double',
                table_border_bottom_style='double',
                heading_border_bottom_style='solid',
                heading_border_bottom_width='2px',
            )
        )

    def custom_pooled_interaction_table(self, other):
        grid = self.build_pooled_interaction_grid(other)
        return self.format_pooled_interaction_table(grid)
