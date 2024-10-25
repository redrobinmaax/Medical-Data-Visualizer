#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Categorical Plot
def draw_cat_plot():
    # Melt the DataFrame to get categorical data
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group by 'cardio', 'variable', 'value' and get the counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    
    # Rename 'size' to 'total'
    df_cat = df_cat.rename(columns={'size': 'total'})
    
    # Draw the categorical plot
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').fig
    
    return fig

# Heatmap
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(corr)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw the heatmap
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    
    return fig

