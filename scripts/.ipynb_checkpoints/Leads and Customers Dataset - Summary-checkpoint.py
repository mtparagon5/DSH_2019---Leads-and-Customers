# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py
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

# # Leads and Customers Dataset - Summary
# Matthew Thomas
# <br>
# mt.paragon5@gmail.com

# ## In Summary:
#     - Addresses, Birthdates, Usernames, and Emails have been removed due to lack of trust in their integrity
#     - Industry, Company Size, and Acquisition Channels have been reduced to their respective 'ranked' columns
#     - Days Since Signup and Score have been replaced by columns with normalized values
#         - this may or may not be necessary but all values will now be 0.0, 1.0, or between 0.0 and 1.0
#     - Job Title was dropped
#         - Originally I picked out unique job titles e.g., marketing, advertising, director, manager, etc. and added 
#             binary columns for them but since Directors stood out the most, I decided to disregard the job titles. 
#     - Changed Is_Manager column to is_director and from a boolean value to a binary 0 or 1
#     
# #### Notes on the final DataFrame:
#     - The final DataFrame consists of 12 columns, each of which being a float value of 0, 1, or between the two
#         - company_size                    float64
#         - completed_form                  float64
#         - visited_pricing                 float64
#         - registered_for_webinar          float64
#         - attended_webinar                float64
#         - converted                       float64
#         - is_male                         float64
#         - is_director                     float64
#         - score_normalized                float64
#         - days_since_signup_normalized    float64
#         - acquisition_channel_numeric     float64
#         - industry_type_numeric           float64

# +
# %config IPCompleter.greedy=True
# %matplotlib inline

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
# -

leads_and_customers = '..\\data\\leads-and-customers.csv'
lead_scoring_fields = '..\\data\\Lead-Scoring-Fields.csv'

df_lac = pd.read_csv(leads_and_customers)
df_lsf = pd.read_csv(lead_scoring_fields)

# #### Review of Columns

df_lac.columns

# #### checking types

df_lac.dtypes

# ### Renaming column names

df_lac.columns = map(str.lower, df_lac.columns)
df_lac.columns = df_lac.columns.str.replace(' ', '_')
df_lac.columns = df_lac.columns.str.replace('-', '_to_')
df_lac.columns = df_lac.columns.str.replace('+', '_plus')

# ### ...

# ### Column manipulation
#  - change birthdate to datetime
#  - added the following columns:
#     - is_male
#     - score_normalized
#     - days_since_signup_normalized
#     - ranked value for company size (0-6) -- got this from Jon
#     - is_director since is_manager is boolean
#     
#  - I originally added binary columns for various industry types, job types, etc. but since directors seem to have the greatest impact on whether there is a conversion, I'm not going to worry about these. If we want to add back those classifications, they're in my exploration notebook.
#  
#  - I'm not interested in the following columns because they seem to be randomly generated:
#      - address
#      - email
#      - username
#      - birthdate
#      

score_max = df_lac['score'].max()
days_since_signup_max = df_lac['days_since_signup'].max()

# +
df_lac['is_male'] = df_lac.sex.apply(lambda x: 1 if x=='M' else 0)
df_lac['is_director'] = df_lac.job_title.apply(lambda x: 1 if 'Director' in x else 0)
df_lac['score_normalized'] = df_lac.score.apply(lambda x: x / score_max)
df_lac['days_since_signup_normalized'] = df_lac.days_since_signup.apply(lambda x: x / days_since_signup_max)

# Got this from Jon
df_lac['company_size'] = np.nan
df_lac.loc[df_lac['company_size_1_to_10'] == 1,'company_size'] = 0
df_lac.loc[df_lac['company_size_11_to_50'] == 1,'company_size'] = 1
df_lac.loc[df_lac['company_size_51_to_100'] == 1,'company_size'] = 2
df_lac.loc[df_lac['company_size_101_to_250'] == 1,'company_size'] = 3
df_lac.loc[df_lac['company_size_251_to_1000'] == 1,'company_size'] = 4
df_lac.loc[df_lac['company_size_1000_to_10000'] == 1,'company_size'] = 5
df_lac.loc[df_lac['company_size_10001_plus'] == 1,'company_size'] = 6

df_lac['acquisition_channel_numeric'] = np.nan
df_lac.loc[df_lac['acquisition_channel'] == 'Organic Search','acquisition_channel_numeric'] = 0
df_lac.loc[df_lac['acquisition_channel'] == 'Cold Email','acquisition_channel_numeric'] = 1
df_lac.loc[df_lac['acquisition_channel'] == 'Paid Search','acquisition_channel_numeric'] = 2
df_lac.loc[df_lac['acquisition_channel'] == 'Cold Call','acquisition_channel_numeric'] = 3
df_lac.loc[df_lac['acquisition_channel'] == 'Paid Leads','acquisition_channel_numeric'] = 4

# -


# ### Dropping columns that...
#     - aren't numeric
#     - that were reduced to/replaced by 'ranked' columns
#     - that aren't normalized

# +
df_lac_numeric = df_lac.select_dtypes(exclude=['object', 'bool', 'datetime64[ns]'])
reduced_columns = [
    'acquisition_channel_cold_call',
    'acquisition_channel_cold_email',
    'acquisition_channel_organic_search',
    'acquisition_channel_paid_leads',
    'acquisition_channel_paid_search',
    'company_size_1_to_10',
    'company_size_1000_to_10000',
    'company_size_10001_plus',
    'company_size_101_to_250',
    'company_size_11_to_50',
    'company_size_251_to_1000',
    'company_size_51_to_100'
]
unwanted_columns = [
    'score',
    'days_since_signup',
]
dropped_columns = reduced_columns + unwanted_columns

df_lac_numeric = df_lac_numeric.drop(columns=dropped_columns)
df_lac_numeric.columns
# -

len(df_lac_numeric.columns)

# ### Do we need industry columns?
#     - if we remove all industry columns, we'd be removing financials, which seemed to have some impact on conversion rate
#     - Can we remove industries without removing financials?
#     - Perhaps I'll just reduce them to a 'ranked' column

df_lac_numeric.columns

# ### With 'ranked' Industry columns...

df_lac_numeric['industry_type_numeric'] = np.nan
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'industry_financial_services','industry_type_numeric'] = 0
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'industry_furniture','industry_type_numeric'] = 1
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'industry_heavy_manufacturing','industry_type_numeric'] = 2
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'scandanavion_design','industry_type_numeric'] = 3
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'transportation','industry_type_numeric'] = 4
df_lac_numeric.loc[df_lac_numeric['industry_type_numeric'] == 'internet','industry_type_numeric'] = 5

df_lac_numeric.columns

industry_columns = [
    'industry_financial_services',
    'industry_furniture',
    'industry_heavy_manufacturing',
    'scandanavion_design',
    'transportation',
    'internet'
]
df_lac_numeric = df_lac_numeric.drop(columns=industry_columns)
df_lac_numeric.columns

len(df_lac_numeric.columns)

# ### Changing columns from int to float

df_lac_numeric.dtypes

df_lac_numeric = df_lac_numeric.astype('float64')

df_lac_numeric.dtypes


