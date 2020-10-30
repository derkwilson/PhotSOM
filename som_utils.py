import numpy as np
import pandas as pd
from astropy.io import fits
import wpca
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit


def rms(data_list):
    data_list = np.array(data_list)
    return np.sqrt(sum(data_list ** 2) / len(data_list))



def compute_colorAB(f1, sig1, f2, sig2):
    delm = -2.5 * np.log10(f1 / f2)
    sigm = np.abs(2.5 * 0.434 * np.sqrt((sig1 / f1) ** 2 + (sig2 / f2) ** 2))
    return (delm, sigm)



def make_redshift_cut(df, cutoff=2.0):
    return df.iloc[np.where(df['zspec'] <= cutoff)]



def load_data(data_dir):

    if '.fits' in data_dir:
        data = pd.DataFrame(fits.open(data_dir)[1].data)
        data = pd.DataFrame(np.array(data).byteswap().newbyteorder(), columns=list(data))
    elif '.csv' in data_dir:
        data = pd.read_csv(data_dir, sep=',', header=0)

    #take care of missing data
    data[data == -99.]  = np.nan
    data[data == -999.] = np.nan
    data[data == 0.]    = np.nan
    return data




def compute_color_columns(dataframe):
    columns = [col for col in list(dataframe) if (('FLUX' in col) and ('ERR' not in col))]

    for i in range(len(columns)-1):
        dataframe['('+columns[i]+'/'+columns[i+1]+')'], dataframe['('+columns[i]+'/'+columns[i+1]+')_ERR'] = compute_colorAB(
            dataframe[columns[i]].values,
            dataframe[columns[i]+'_ERR'].values,
            dataframe[columns[i+1]].values,
            dataframe[columns[i+1]+'_ERR'].values )
    return dataframe





def add_error_columns(dataframe, error_fraction=0.1):
    columns = list(dataframe)
    for col in columns:
        if ('ERR' not in col) and (col+'_ERR' not in list(dataframe)):
            dataframe[col+'_ERR'] = error_fraction * dataframe[col]
    return dataframe




def resample_data(df, seed=0):
    np.random.seed(seed)
    import copy
    resampled_df = copy.deepcopy(df)
    labels = list(resampled_df)
    for i in range(len(labels)):
        l = labels[i]
        if ('_ERR' not in l) and ('zspec' not in l) and ('FLUX' in l):
            resampled_df[l] = np.random.uniform(np.nan_to_num(df[l].values - df[l+'_ERR'].values), \
                                                np.nan_to_num(df[l].values + df[l+'_ERR'].values))
            resampled_df[l][np.isnan(df[l].values)] = np.nan
    return resampled_df



def subset_data(df, cutoff=4, seed=0):
    np.random.seed(seed)
    import copy
    subset_df = copy.deepcopy(df)
    labels = list(subset_df)
    photom_labels = [labels[i] for i in range(len(labels)) if (('_ERR' not in labels[i]) and ('zspec' not in labels[i]) and ('FLUX' in labels[i]))]

    num_phot_features = len(photom_labels)
    num = int(np.ceil(np.random.uniform(num_phot_features-cutoff, num_phot_features)))
    print('Subset Num:')
    print(num)
    np.random.shuffle(photom_labels)
    subset_labels = photom_labels[0:num]
    for k in range(len(photom_labels)):
        if photom_labels[k] not in subset_labels:
            print(photom_labels[k])
            subset_df.drop([photom_labels[k], photom_labels[k]+'_ERR'], axis=1, inplace=True)
    return subset_df




def reorder_column(df, idx, column_name):
    cols = list(df)
    cols.insert(idx, cols.pop(cols.index(column_name)))
    df = df.ix[:, cols]
    return df
