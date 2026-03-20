import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import interp1d

from .dataframe_analysis import ProcessedData

class PlotGenerator(ProcessedData):
    """Generate publication-quality plots for class size analysis."""

    def __init__(self, df):
        """Initialize with processed data."""
        super().__init__(df)

    @staticmethod
    def _residualize(target, controls):
        """Remove linear effects of controls from target variable via OLS residuals."""
        return sm.OLS(target, controls).fit().resid
    
    @staticmethod
    def _apply_plot_styling(ax):
        """Remove top and right spines for cleaner appearance."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_maimonides_sawtooth(self):
        """Visualize actual vs predicted class size under Maimonides' discontinuity rule."""
        e = np.linspace(self.df['c_size'].min(), self.df['c_size'].max(), 1000)
        plt.figure(figsize=(10, 6))
        
        actual_means = self.df.groupby('c_size')['classize'].mean()
        plt.plot(actual_means.index, actual_means.values, 
                color='black', linewidth=1, label='Actual class size')
        plt.plot(e, self.maimonides_rule(e), 
                color='blue', linestyle='--', linewidth=1.5, label='Maimonides Rule')
        
        for y in [20.5, 27, 30.3, 32, 33.5, 40]:
            plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        
        plt.xlabel('Enrollment count')
        plt.ylabel('Class size')
        plt.title(self.label)
        plt.legend(frameon=True, loc='lower right')
        self._apply_plot_styling(plt.gca())
        plt.tight_layout()
        plt.show()

    def plot_scores_vs_predicted_size(self):
        """Show relationship between test scores and predicted class size across enrollment bins."""
        enroll_bin = self.enrollment_bins(self.df['c_size'])
        bin_centers = np.sort(enroll_bin.dropna().unique())
        bin_means = self.df.groupby(enroll_bin, observed=False)['avgverb'].mean()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        line1, = ax1.plot(bin_means.index, bin_means.values, color='black', label='Average test scores')
        ax1.set_xlabel('Enrollment count')
        ax1.set_ylabel('Average reading score')
        ax1.set_ylim(68, 80)
        ax1.set_xticks(range(5, 170, 20))

        ax2 = ax1.twinx()
        line2, = ax2.plot(bin_centers, self.maimonides_rule(bin_centers), color='blue', linestyle='--', label='Predicted class size')
        ax2.set_ylabel('Average size function')
        ax2.set_ylim(5, 40)

        for y in [20.5, 27, 30.3, 33.5, 40]:
            ax2.axhline(y=y, color='gray', linestyle=':', linewidth=0.8)

        ax1.legend(handles=[line1, line2], frameon=True, loc='lower right')
        plt.title(self.label)
        plt.tight_layout()
        plt.show()

    def plot_residual(self):
        """Plot residualized test scores vs residualized predicted class size."""
        valid = self.df[['avgverb', 'tipuach', 'c_size']].dropna().index
        controls = sm.add_constant(self.df.loc[valid, ['tipuach', 'c_size']])
        bin_means = pd.DataFrame({
            'bin': self.enrollment_bins(self.df.loc[valid, 'c_size']),
            'y_resid': self._residualize(self.df.loc[valid, 'avgverb'], controls),
            'f_resid': self._residualize(self.maimonides_rule(self.df.loc[valid, 'c_size']), controls),
        }).groupby('bin', observed=False)[['y_resid', 'f_resid']].mean()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        line1, = ax1.plot(bin_means.index, bin_means['y_resid'], color='black', label='Average test scores')
        ax1.set_ylabel('Reading score residual')
        ax1.set_xlabel('Enrollment count')
        ax1.set_ylim(-5, 5)
        ax1.set_xticks(range(5, 170, 20))

        ax2 = ax1.twinx()
        line2, = ax2.plot(bin_means.index, bin_means['f_resid'], color='blue', linestyle='--', label='Predicted class size')
        ax2.set_ylabel('Size-function residual')
        ax2.set_ylim(-15, 15)

        ax1.legend(handles=[line1, line2], loc='lower right', frameon=True)
        plt.title(self.label)
        plt.tight_layout()
        plt.show()

    def plot_cdf_by_instrument(self):
        """Plot CDF of class size in ±5 discontinuity sample, split by Maimonides Rule.</br>
        
        Separates data by whether predicted class size is >= 32 (high) or < 32 (low)."""
        # Filter to ±5 discontinuity sample
        query_str = "(c_size >= 36 & c_size <= 45) | (c_size >= 76 & c_size <= 85) | (c_size >= 116 & c_size <= 125)"
        disc_df = self.df.query(query_str).copy()
        
        # Compute Maimonides rule predicted class size
        disc_df['p_size'] = self.maimonides_rule(disc_df['c_size'])
        
        # Create binary instrument: 1 if p_size >= 32, 0 otherwise
        disc_df['high_instrument'] = (disc_df['p_size'] >= 32).astype(int)
        
        # Split by instrument
        low_data = disc_df[disc_df['high_instrument'] == 0]['classize'].dropna().sort_values()
        high_data = disc_df[disc_df['high_instrument'] == 1]['classize'].dropna().sort_values()
        
        # Compute empirical CDFs handling duplicates via unique values
        low_sorted = np.sort(low_data.values)
        high_sorted = np.sort(high_data.values)
        
        low_unique, low_counts = np.unique(low_sorted, return_counts=True)
        high_unique, high_counts = np.unique(high_sorted, return_counts=True)
        
        low_cdf_vals = np.cumsum(low_counts) / len(low_data)
        high_cdf_vals = np.cumsum(high_counts) / len(high_data)
        
        # Create smooth interpolations with linear kind
        low_interp = interp1d(low_unique, low_cdf_vals, kind='linear', fill_value='extrapolate', bounds_error=False)
        high_interp = interp1d(high_unique, high_cdf_vals, kind='linear', fill_value='extrapolate', bounds_error=False)
        
        # Generate smooth curves
        x_smooth = np.linspace(15, 41, 500)
        low_smooth = np.clip(low_interp(x_smooth), 0, 1)
        high_smooth = np.clip(high_interp(x_smooth), 0, 1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(x_smooth, low_smooth, linestyle='--', linewidth=1.5, color='blue', 
                label='Maimonides Rule < 32')
        ax.plot(x_smooth, high_smooth, linestyle='-', linewidth=1.5, color='black', 
                label='Maimonides Rule >= 32')
        
        ax.set_xlabel('Class size')
        ax.set_ylabel('CDF')
        ax.set_xlim(15, 41)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(15, 42))
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        self._apply_plot_styling(ax)
        ax.set_title(f"{self.label.replace('th Grade', 'th Grade')}")
        
        plt.tight_layout()
        plt.show()
