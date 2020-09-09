import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import COLORS, POVERTY_MAPPING

class Plotter:
    def __init__(self, data, ind_data):
        self.data = data
        self.ind_data = ind_data
        
        #plot basic styles
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18
        plt.rcParams['patch.edgecolor'] = 'k'
    
    def plot_poverty_breakdown(self):
        # Labels for training
        labels = self.data.loc[(self.data['Target'].notnull()) & (self.data['parentesco1'] == 1),
                                ['Target', 'idhogar']]

        # Value counts of target
        label_counts = labels['Target'].value_counts().sort_index()

        f, ax = plt.subplots(figsize=(8, 6))

        # Bar plot of occurrences of each label
        label_counts.plot.bar(figsize = (8, 6), 
                              color = COLORS.values(),
                              edgecolor = 'k', linewidth = 2)

        # Formatting
        plt.xlabel('Poverty Level')
        plt.ylabel('Count')
        plt.xticks([x - 1 for x in POVERTY_MAPPING.keys()], 
                   list(POVERTY_MAPPING.values()), rotation = 30)
        plt.title('Poverty Level Breakdown')
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_household_level_data_corr(self):
        heads = self.data.loc[self.data['parentesco1'] == 1, :]
        train_heads = heads.loc[heads['Target'].notnull(), :].copy()
        variables = ['Target', 'dependency', 'warning', 'walls+roof+floor', 'meaneduc',
                    'floor', 'r4m1', 'overcrowding']

        # Calculate the correlations
        corr_mat = train_heads[variables].corr().round(2)

        f, ax = plt.subplots(figsize=(12, 12))

        # Draw a correlation heatmap
        plt.rcParams['font.size'] = 18
        plt.figure(figsize = (12, 12))
        sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 
                    cmap = "RdYlGn_r", annot = True)
        plt.title('Household Features Correlation to Target')
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_home_condition_to_target(self):
        f, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(x = 'Target', y = 'walls+roof+floor', data = self.data)
        plt.title('House Condition by Target')
        plt.xticks([0, 1, 2, 3], POVERTY_MAPPING.values())
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_warning_to_target(self):
        f, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(x = 'Target', y = 'warning', data = self.data)
        plt.title('House Condition by Target')
        plt.xticks([0, 1, 2, 3], POVERTY_MAPPING.values())
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_schooling_to_target(self):
        f, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(x = 'Target', y = 'meaneduc', data = self.data)
        plt.title('House Condition by Target')
        plt.xticks([0, 1, 2, 3], POVERTY_MAPPING.values())
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image
    
    def plot_overcrowding_to_target(self):
        f, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(x = 'Target', y = 'overcrowding', data = self.data)
        plt.title('House Condition by Target')
        plt.xticks([0, 1, 2, 3], POVERTY_MAPPING.values())
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_female_head_to_target(self):
        head_gender = self.ind_data.loc[self.ind_data['parentesco1'] == 1, ['idhogar', 'female']]
        temp_data = self.data.merge(head_gender, on = 'idhogar', how = 'left').rename(columns = {'female': 'female-head'})

        f, ax = plt.subplots(figsize=(8, 8))

        sns.boxplot(x = 'Target', y = 'meaneduc', hue = 'female-head', data = self.data)
        plt.title('Average Education by Target and Female Head of Household', size = 16)
        plt.xticks([0, 1, 2, 3], POVERTY_MAPPING.values())
        plt.tight_layout()

        # return as bytes image
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image