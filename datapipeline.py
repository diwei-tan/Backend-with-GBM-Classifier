import pandas as pd
import numpy as np

from functools import reduce
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from constants import sqr_, id_, ind_bool, ind_ordered


class Datapipeline():

    def __init__(self):
        self.train_preprocess = _compose(_correct_labels)
        self.data_preprocess = _compose(_handle_missing_v18q1,
                                        _handle_missing_v2a1,
                                        _handle_missing_rez_esc,
                                        _drop_all_data_square_variables)
        self.heads_preprocess = _compose(_drop_household_level_redundant_variables,
                                         _add_hhsize_diff_to_heads,
                                         _change_heads_elec_variables_to_one_var,
                                         _drop_heads_area2,
                                         _change_heads_booleans_to_ordinals,
                                         _add_heads_total_for_walls_roofs_floors,
                                         _add_heads_warning,
                                         _add_heads_bonus,
                                         _add_heads_per_captia_features)
        self.ind_preprocess = _compose(_drop_ind_redundant_variables,
                                       _change_ind_booleans_to_ordinals,
                                       _add_ind_constructed_features)
        self.ind_agg_preprocess = _compose(_drop_ind_agg_redundant_values)
    
    def preprocess_data(self, train, test):
        """
        This function takes in the train and test and pass the information through a
        preprocessing pipeline to clean the raw dataset and return a cleaned dataset
        with additional engineered features as well.

        :param train, test
        :return: final: final cleaned household level dataset. points for training have
                 target label, where else points for testing do not have target
                 ind: individual level data, which might be useful for plotting, etc.
        """
        train = self.train_preprocess(train)

        # Add null Target column to test
        test['Target'] = np.nan
        data = train.append(test, ignore_index = True)

        data = self.data_preprocess(data)

        # household level data preprocess
        heads = data.loc[data['parentesco1'] == 1, :]
        heads = self.heads_preprocess(heads)

        # individual level data preprocess
        ind = data[id_ + ind_bool + ind_ordered]
        ind = self.ind_preprocess(ind)

        # individual level data aggregation
        ind_agg = _create_aggregation_from_ind(ind)
        ind_agg = self.ind_agg_preprocess(ind_agg)

        #combine household and aggregate level data to form final cleaned dataset
        final = heads.merge(ind_agg, on = 'idhogar', how = 'left')
        
        return final, ind, test

    def preprocess_test(self, test):
        """
        This function takes in the train and test and pass the information through a
        preprocessing pipeline to clean the raw dataset and return a cleaned dataset
        with additional engineered features as well.

        :param train, test
        :return: final: final cleaned household level dataset. points for training have
                 target label, where else points for testing do not have target
                 ind: individual level data, which might be useful for plotting, etc.
        """
        # Add null Target column to test
        test['Target'] = np.nan
        data = test

        data = self.data_preprocess(data)

        # household level data preprocess
        heads = data.loc[data['parentesco1'] == 1, :]
        heads = self.heads_preprocess(heads)
        
        # individual level data preprocess
        ind = data[id_ + ind_bool + ind_ordered]
        ind = self.ind_preprocess(ind)

        # individual level data aggregation
        ind_agg = _create_aggregation_from_ind(ind)
        ind_agg = self.ind_agg_preprocess(ind_agg)

        #combine household and aggregate level data to form final cleaned dataset
        if heads.empty:
            final = ind_agg
        else:
            final = heads.merge(ind_agg, on = 'idhogar', how = 'left')
        
        return final, ind, test


def _compose(*functions):
    return reduce(lambda f, g: lambda x: g(f(x)),
                  functions,
                  lambda x: x)

def _correct_labels(train):
    # Groupby the household and figure out the number of unique values
    all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]

    # Iterate through each household
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])
        
        # Set the correct label for all members in the household
        train.loc[train['idhogar'] == household, 'Target'] = true_target

    return train

def _handle_missing_v18q1(data):
    data['v18q1'] = data['v18q1'].fillna(0)
    return data

def _handle_missing_v2a1(data):
    # Fill in households that own the house with 0 rent payment
    data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0
    # Create missing rent payment column
    data['v2a1-missing'] = data['v2a1'].isnull()
    return data

def _handle_missing_rez_esc(data):
    # If individual is over 19 or younger than 7 and missing years behind, set it to 0
    data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0
    # Add a flag for those between 7 and 19 with a missing value
    data['rez_esc-missing'] = data['rez_esc'].isnull()
    # any value > 5 should be 5
    data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5

    return data    

def _drop_all_data_square_variables(data):
    data = data.drop(columns = sqr_)
    return data

def _drop_household_level_redundant_variables(heads):
    heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
    return heads

def _add_hhsize_diff_to_heads(heads):
    heads['hhsize-diff'] = heads['tamviv'] - heads['hhsize']
    return heads

def _change_heads_elec_variables_to_one_var(heads):
    elec = []

    # Assign values
    for _, row in heads.iterrows():
        if row['noelec'] == 1:
            elec.append(0)
        elif row['coopele'] == 1:
            elec.append(1)
        elif row['public'] == 1:
            elec.append(2)
        elif row['planpri'] == 1:
            elec.append(3)
        else:
            elec.append(np.nan)
            
    # Record the new variable and missing flag
    heads['elec'] = elec
    heads['elec-missing'] = heads['elec'].isnull()

    # Remove the electricity columns
    heads = heads.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])

    return heads

def _drop_heads_area2(heads):
    heads = heads.drop(columns = 'area2')
    return heads

def _change_heads_booleans_to_ordinals(heads):
    # walls
    heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]),
                               axis = 1)
    heads = heads.drop(columns = ['epared1', 'epared2', 'epared3'])

    # roof
    heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]),
                              axis = 1)
    heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])
    
    # floor
    heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]),
                               axis = 1)
    heads = heads.drop(columns = ['eviv1', 'eviv2', 'eviv3'])

    return heads

def _add_heads_total_for_walls_roofs_floors(heads):
    heads['walls+roof+floor'] = heads['walls'] + heads['roof'] + heads['floor']
    return heads

def _add_heads_warning(heads):
    heads['warning'] = 1 * (heads['sanitario1'] + 
                         (heads['elec'] == 0) + 
                         heads['pisonotiene'] + 
                         heads['abastaguano'] + 
                         (heads['cielorazo'] == 0))
    return heads

def _add_heads_bonus(heads):
    # Owns a refrigerator, computer, tablet, and television
    heads['bonus'] = 1 * (heads['refrig'] + 
                        heads['computer'] + 
                        (heads['v18q1'] > 0) + 
                        heads['television'])
    return heads

def _add_heads_per_captia_features(heads):
    heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']
    heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']
    heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']
    heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
    return heads

def _drop_ind_redundant_variables(ind):
    ind = ind.drop(columns = 'male')
    return ind

def _change_ind_booleans_to_ordinals(ind):
    ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)
    ind = ind.drop(columns = ['instlevel1', 'instlevel2', 'instlevel3',
                              'instlevel4', 'instlevel5', 'instlevel6',
                              'instlevel7', 'instlevel8', 'instlevel9'])
    return ind

def _add_ind_constructed_features(ind):
    ind['escolari/age'] = ind['escolari'] / ind['age']
    ind['inst/age'] = ind['inst'] / ind['age']
    ind['tech'] = ind['v18q'] + ind['mobilephone']
    return ind

def _create_aggregation_from_ind(ind):
    # Define custom function
    range_ = lambda x: x.max() - x.min()
    range_.__name__ = 'range_'

    # Group and aggregate
    ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])

    # Rename the columns
    new_col = []
    for c in ind_agg.columns.levels[0]:
        for stat in ind_agg.columns.levels[1]:
            new_col.append(f'{c}-{stat}')
    ind_agg.columns = new_col

    return ind_agg

def _drop_ind_agg_redundant_values(ind_agg):
    # Create correlation matrix
    corr_matrix = ind_agg.corr()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

    ind_agg = ind_agg.drop(columns = to_drop)
    
    return ind_agg