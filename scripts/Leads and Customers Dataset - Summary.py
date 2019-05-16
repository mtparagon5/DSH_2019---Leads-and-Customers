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

import seaborn as sns; sns.set(style="ticks", color_codes=True)

style.use('ggplot')
# -

leads_and_customers = '..\\data\\leads-and-customers.csv'
lead_scoring_fields = '..\\data\\Lead-Scoring-Fields.csv'

df_lac = pd.read_csv(leads_and_customers)
df_lsf = pd.read_csv(lead_scoring_fields)

# #### Review of Columns

df_lac.columns

# +
# df_lac.industry
# -

# #### checking types

# +
# df_lac.dtypes
# -

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

# +
#  to normalize score and days_since_signup
# score_max = df_lac['score'].max()
# days_since_signup_max = df_lac['days_since_signup'].max()

# df_lac['score_normalized'] = df_lac.score.apply(lambda x: x / score_max)
# df_lac['days_since_signup_normalized'] = df_lac.days_since_signup.apply(lambda x: x / days_since_signup_max)

# +
# Got this from Jon
df_lac['company_size'] = np.nan
df_lac.loc[df_lac['company_size_1_to_10'] == 1,'company_size'] = 0
df_lac.loc[df_lac['company_size_11_to_50'] == 1,'company_size'] = 1
df_lac.loc[df_lac['company_size_51_to_100'] == 1,'company_size'] = 2
df_lac.loc[df_lac['company_size_101_to_250'] == 1,'company_size'] = 3
df_lac.loc[df_lac['company_size_251_to_1000'] == 1,'company_size'] = 4
df_lac.loc[df_lac['company_size_1000_to_10000'] == 1,'company_size'] = 5
df_lac.loc[df_lac['company_size_10001_plus'] == 1,'company_size'] = 6

# industry column already existed as categorical
# # reducing industry columns to a single categorical column
# df_lac['industry'] = ''
# df_lac.loc[df_lac['industry_furniture'] == 1,'industry'] = 'furniture'
# df_lac.loc[df_lac['industry_financial_services'] == 1,'industry'] = 'financial'
# df_lac.loc[df_lac['industry_heavy_manufacturing'] == 1,'industry'] = 'heavy manufacturing'
# df_lac.loc[df_lac['scandanavion_design'] == 1,'industry'] = 'scandanavion design'
# df_lac.loc[df_lac['transportation'] == 1,'industry'] = 'transportation'
# df_lac.loc[df_lac['internet'] == 1,'industry'] = 'internet'

# changing is_manager to is_director
df_lac['is_director'] = False
df_lac.loc[df_lac['is_manager'] == 0,'is_director'] = False
df_lac.loc[df_lac['is_manager'] == 1,'is_director'] = True

# changing completed_form to bool column
df_lac['completed_form_bool'] = False
df_lac.loc[df_lac['completed_form'] == 0,'completed_form_bool'] = False
df_lac.loc[df_lac['completed_form'] == 1,'completed_form_bool'] = True

# changing visited_pricing to bool column
df_lac['visited_pricing_bool'] = False
df_lac.loc[df_lac['visited_pricing'] == 0,'visited_pricing_bool'] = False
df_lac.loc[df_lac['visited_pricing'] == 1,'visited_pricing_bool'] = True

# changing registered_for_webinar to bool column
df_lac['registered_for_webinar_bool'] = False
df_lac.loc[df_lac['registered_for_webinar'] == 0,'registered_for_webinar_bool'] = False
df_lac.loc[df_lac['registered_for_webinar'] == 1,'registered_for_webinar_bool'] = True

# changing attended_webinar to bool column
df_lac['attended_webinar_bool'] = False
df_lac.loc[df_lac['attended_webinar'] == 0,'attended_webinar_bool'] = False
df_lac.loc[df_lac['attended_webinar'] == 1,'attended_webinar_bool'] = True

# changing attended_webinar to bool column
df_lac['is_male'] = False
df_lac.loc[df_lac['sex'] == 'F','is_male'] = False
df_lac.loc[df_lac['sex'] == 'M','is_male'] = True



# +
# df_with_dummies = pd.get_dummies(df_lac, drop_first=True) ## commented out due to amount of time it was taking on my pc
# -





df_lac[['completed_form_bool', 'completed_form']].head()

# ### Dropping columns that...
#     - that were reduced to/replaced by 'ranked' columns
#     - that weren't reliable e.g., address, birthdate, etc.
#     - that were the binary form of an already categorical column e.g., industry, acquisition channel

# +
## commented out since using reduced categorical columns
# df_lac_numeric = df_lac.select_dtypes(exclude=['object', 'bool', 'datetime64[ns]'])
reduced_columns = [
    # changed to ranked column
    'company_size_1_to_10',
    'company_size_1000_to_10000',
    'company_size_10001_plus',
    'company_size_101_to_250',
    'company_size_11_to_50',
    'company_size_251_to_1000',
    'company_size_51_to_100',
    # changed to is_director
    'is_manager',
    # these were converted to bool columns
    'attended_webinar',
    'registered_for_webinar',
    'visited_pricing',
    'completed_form',
    # changed sex to is_male
    'sex'
]
unwanted_columns = [
    #unreliable or redundant
    'address',
    'birthdate',
    'mail',
    'name',
    'username',
    'job_title',
    'company_size',
    # keeping only acquisiton_channel since it is by itself a categorical column
    'acquisition_channel_cold_call',
    'acquisition_channel_cold_email',
    'acquisition_channel_organic_search',
    'acquisition_channel_paid_leads',
    'acquisition_channel_paid_search',
    # keeping industry column only since it's already categorical
    'industry_financial_services',
    'industry_furniture',
    'industry_heavy_manufacturing',
    'scandanavion_design',
    'transportation',
    'internet',
]
dropped_columns = reduced_columns + unwanted_columns

df_lac_categorical = df_lac.drop(columns=dropped_columns)
# -

df_lac_categorical.head()

# ### Some Exploration

df_ctg = df_lac_categorical



# #### NOTE: 
#     - registered for webinar bool col seems closeley related to converted

# +
df_ctg[['converted', 
         'completed_form_bool', 
         'visited_pricing_bool', 
         'registered_for_webinar_bool',
         'attended_webinar_bool']].astype(float).hist()

plt.tight_layout()
plt.show()
# -

cols_to_drop = [
    'acquisition_channel',
    'industry'
]
df_floats = df_ctg.drop(columns=cols_to_drop).astype(float)

g = sns.pairplot(df_floats)

# #### NOTE:
#     - score seems to be much more related to attended webinar than registered for webinar
#     - score is obviously related to converted since everyone over score of ~150 converted
#     - as noted above, converted hist and registered for webinar hist seem similar in distribution
#     - agreed, days since seems unreliable since it seems to have a very even/unrealistic distribution

# dropping days_since_signup since it seems unreliable
df_ctg = df_ctg.drop(columns=['days_since_signup'])

df_ctg.head()

output_file = '..\\data\\cleaned_df_categorical_columns.json'

df_ctg.to_json(output_file, orient='records')


