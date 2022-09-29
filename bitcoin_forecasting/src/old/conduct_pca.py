import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 205)
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

    
def generate_pca(n_dim, df, df_name):
    """
    After standardizing original variables, fit and transform them
    Check the contribution rate in PCA, plot it, and see which variables are important in the principal components
    Generate new variables with PCA
    
    Params:
    n_dim: int
    df: pandas.DataFrame
    df_name: str (Assign the name of dataframe)
    """
    
    # Create new variables
    pca = PCA(n_components=n_dim)
    
    # Standardize the variables
    normalized = StandardScaler().fit_transform(df.values)
    
    # Fit and transform
    dim_pca = pca.fit_transform(normalized) 
    dim_pca = pd.DataFrame(dim_pca)
    dim_pca.columns = list(map(str, list(range(n_dim))))
    dim_pca.index = df.index
    dim_pca.columns = df_name + "_pca_" + dim_pca.columns
    
    # Contribution rate
    plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
    np.set_printoptions(precision=3, suppress=True) 
    print('Explained variance ratio: {}'.format(pca.explained_variance_ratio_))
    print('Accumulated contribution rate', pca.explained_variance_ratio_.sum())
    
    return dim_pca