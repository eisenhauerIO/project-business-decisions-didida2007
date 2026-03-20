import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from great_tables import GT, style, loc


from auxiliary.tables import TableGenerator
from auxiliary.plots import PlotGenerator

df_grade4 = pd.read_stata(r"F:\python-files\econ-481\project-business-decisions-didida2007\data\final4.dta")
df_grade5 = pd.read_stata(r"F:\python-files\econ-481\project-business-decisions-didida2007\data\final5.dta")
