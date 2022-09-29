import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import probplot
from scipy.stats import t as t_dist
from scipy.stats import normaltest
from scipy.stats import kstest
from scipy.stats import ttest_1samp
import statsmodels.api as sm


def viz_normal_dist(df):
    """
    Plot normal distributions and actual distributions of data
    
    Params:
        df: pd.DataFrame
    
    """
    
    for i in range(len(df.columns)):
        # Normal distribution
        x_list = np.linspace(df[df.columns[i]].min(), df[df.columns[i]].max(), 100)
        y_list = norm.pdf(x_list, loc=df[df.columns[i]].mean(), scale=df[df.columns[i]].std())
        plt.plot(x_list, y_list)
        plt.title("Normal distributions vs actual distributions of " + df.columns[i])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        df[df.columns[i]].hist(bins=100, density=True)
        plt.show()

def viz_probplot_norm(df):
    """
    Visualize probability plot of the variables
    
    Params:
        df: pd.DataFrame
        
    """
    
    for i in range(len(df.columns)):
        # Plot the probplot
        probplot(df[df.columns[i]].dropna(), dist="norm", fit=True, plot=plt);
        plt.title("Probplot of " + df.columns[i])
        plt.show()

def viz_t_dist(df):
    """
    Plot t-distributions and actual distributions of data
    
    Params:
        df: pd.DataFrame
    """
    # T-distribution
    for i in range(len(df.columns)):
        x_list = np.linspace(df[df.columns[i]].min(), df[df.columns[i]].max(), 100)
        params = t_dist.fit(df[df.columns[i]].dropna())
        display(params)
        dof, loc, scale = params
        y_list = t_dist.pdf(x_list, dof, loc, scale)
        plt.plot(x_list, y_list);
        plt.title("T-distributions and actual distributions of " + df.columns[i])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        df[df.columns[i]].hist(bins=100, density=True);
        plt.show()
    
def viz_probplot_t(df):
    """
    Visualize probability plot of the variables
    
    Params:
        df: pd.DataFrame
    """
    class my_t:
        def __init__(self,dof):
            self.dof = dof

        def fit(self, x):
            return t_dist.fit(x)
    
        def ppf(self,x, loc=0, scale=1):
            return t_dist.ppf(x, self.dof, loc, scale)

    # T-distribution
    for i in range(len(df.columns)):
        x_list = np.linspace(df[df.columns[i]].min(), df[df.columns[i]].max(), 100)
        params = t_dist.fit(df[df.columns[i]].dropna())
        display(params)
        dof, loc, scale = params
        y_list = t_dist.pdf(x_list, dof, loc, scale)
        sm.qqplot(df[df.columns[i]], dist=my_t(dof), line="s");
        

def test_normal_dist(df):
    """
    Test normality for the variables
    
    Params:
        df: pd.DataFrame

    """

    for i in range(len(df.columns)):
        if normaltest(df[df.columns[i]])[1] >= 0.05:
            display("We cannot reject the null hypothesis that " + df.columns[i] + " comes from normal distributions")
        elif normaltest(df[df.columns[i]])[1] < 0.05:
            display("We can reject the null hypothesis that " + df.columns[i] + " comes from normal distributions")
        else:
            pass


def test_t_dist(df):
    """
    Test t-distributions for the variables
    
    Params:
        df: pd.DataFrame

    """

    def cdf(x):
        return t_dist.cdf(x, dof, loc, scale)
    for i in range(len(df.columns)):
        dof, loc, scale = t_dist.fit(df[df.columns[i]].dropna())
        if kstest(df[df.columns[i]].dropna(), cdf)[1] >= 0.05:
            display("We cannot reject the null hypothesis that " + df.columns[i] + " comes from t-distributions")
        elif kstest(df[df.columns[i]].dropna(), cdf)[1] < 0.05:
            display("We can reject the null hypothesis that " + df.columns[i] + " comes from t-distributions")
        else:
            pass


def test_t_dist_1sample(df):
    """
    Test t-dist 1-sample mean
    
    """
    
    for i in range(len(df.columns)):
        if ttest_1samp(df[df.columns[i]], 0)[1] >= 0.05:
            display("We cannot reject the null hypothesis that the average of "+ df.columns[i] +" is 0.")
        elif ttest_1samp(df[df.columns[i]], 0)[1] < 0.05:
            display("We can reject the null hypothesis that the average of "+ df.columns[i] +" is 0.")
        else:
            pass
        

