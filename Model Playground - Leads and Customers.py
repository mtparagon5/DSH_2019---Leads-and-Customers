# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import sklearn

# +
# ----- taking json file and making it smaller in size (originally orient=records ==> file too large for git) -----#
# df = pd.read_json('data\\cleaned_df_numerical_columns_only.json')
# df.to_json('data\\cleaned_df_numerical_columns_only.json')
# -

df = pd.read_json('data\\cleaned_df_numerical_columns_only.json')

df


