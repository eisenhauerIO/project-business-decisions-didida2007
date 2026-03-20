import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from .dataframe_analysis import ProcessedData

class PlotGenerator(ProcessedData):

    def __init__(self, df):
        super().__init__(df)

    @staticmethod
    def _residualize(target, controls):
        return sm.OLS(target, controls).fit().resid


    
    def plot_maimonides_sawtooth(self):

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
        
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    


    def plot_scores_vs_predicted_size(self):

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