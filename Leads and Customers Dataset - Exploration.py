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

# # Leads and Customers Dataset - Exploration
# matthew thomas
# <br>
# mt.paragon5@gmail.com

# +
# %config IPCompleter.greedy=True
# %matplotlib inline

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# -

leads_and_customers = 'data\\leads-and-customers.csv'
lead_scoring_fields = 'data\\Lead-Scoring-Fields.csv'

df_lac = pd.read_csv(leads_and_customers)
df_lsf = pd.read_csv(lead_scoring_fields)

# #### checking for null / na values

# +
# df_lac.isnull().sum()
# df_lac.isna().sum()
# -

# #### checking types

df_lac.dtypes

# #### rename column names

df_lac.columns = map(str.lower, df_lac.columns)
df_lac.columns = df_lac.columns.str.replace(' ', '_')
df_lac.columns = df_lac.columns.str.replace('-', '_to_')
df_lac.columns = df_lac.columns.str.replace('+', '_plus')

# #### column manipulation
#  - change birthdate to datetime
#  - add age_seconds column (int); not sure if useful
#  - added year_month column to investigate trends in scores by age groups
#  - added the following columns:
#     - age_seconds
#     - bd_year_month
#     - bd_month
#     - is_male
#     - is_manager
#     - is_director
#     - is_vp
#     - is_admin
#     - is_chief_officer
#     - is_advertising
#     - is_sales
#     - is_marketing
#     - is_other
#     - is_web_ind
#     - is_financials_ind
#     - is_furniture_ind
#     - is_transportation_ind
#     - is_manufacturing_ind
#     - is_scandesign_ind
#     - score_normalized
#     - zip_code
#

score_max = df_lac['score'].max()
days_since_signup_max = df_lac['days_since_signup'].max()

# +
df_lac.birthdate = pd.to_datetime(df_lac.birthdate)
df_lac['age_seconds'] = df_lac.birthdate.map(dt.datetime.toordinal)
df_lac['bd_year_month'] = df_lac.birthdate.apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
df_lac['bd_month'] = df_lac.birthdate.apply(lambda x: '{month}'.format(month=int(x.month)) if len(str(x.month)) > 1 else '{0}{month}'.format(0, month=int(x.month)))
df_lac['is_male'] = df_lac.sex.apply(lambda x: 1 if x=='M' else 0)
df_lac['is_manager'] = df_lac.job_title.apply(lambda x: 1 if 'Manager' in x else 0)
df_lac['is_director'] = df_lac.job_title.apply(lambda x: 1 if 'Director' in x else 0)
df_lac['is_vp'] = df_lac.job_title.apply(lambda x: 1 if 'VP' in x else 0)
df_lac['is_admin'] = df_lac.job_title.apply(lambda x: 1 if 'Administrator' in x else 0)
df_lac['is_chief_officer'] = df_lac.job_title.apply(lambda x: 1 if ('Chief' in x and 'Officer' in x) else 0)
df_lac['is_advertising'] = df_lac.job_title.apply(lambda x: 1 if 'Advertising' in x else 0)
df_lac['is_sales'] = df_lac.job_title.apply(lambda x: 1 if 'Sales' in x else 0)
df_lac['is_marketing'] = df_lac.job_title.apply(lambda x: 1 if 'Marketing' in x else 0)
df_lac['is_other'] = df_lac.job_title.apply(lambda x: 1 if ('Advertising' not in x and 'Marketing' not in x and 'Advertising' not in x) else 0)
df_lac['is_web_ind'] = df_lac.industry.apply(lambda x: 1 if 'Web' in x else 0)
df_lac['is_financials_ind'] = df_lac.industry.apply(lambda x: 1 if 'Financial' in x else 0)
df_lac['is_furniture_ind'] = df_lac.industry.apply(lambda x: 1 if 'Furniture' in x else 0)
df_lac['is_transportation_ind'] = df_lac.industry.apply(lambda x: 1 if 'Transportation' in x else 0)
df_lac['is_manufacturing_ind'] = df_lac.industry.apply(lambda x: 1 if 'Manufacturing' in x else 0)
df_lac['is_scandesign_ind'] = df_lac.industry.apply(lambda x: 1 if 'Scandanavion' in x else 0)
df_lac['score_normalized'] = df_lac.score.apply(lambda x: x / score_max)
df_lac['days_since_signup_normalized'] = df_lac.days_since_signup.apply(lambda x: x / days_since_signup_max)
df_lac['zip_code'] = df_lac.address.apply(lambda x: ''.join(c for c in x[-10:] if c.isdigit())[:5])
df_lac['score_normalized'] = df_lac.score.apply(lambda x: x / score_max)

# df_lac.bd_year_month = pd.to_datetime(df_lac.birthdate, format='%Y-%m')


# +
df_grouped_by_ym = df_lac.set_index('bd_year_month').groupby('bd_year_month')
df_grouped_by_m = df_lac.set_index('bd_month').groupby('bd_month')
# df_grouped_by_zip = df_lac.groupby('zip_code')

df_score_grouped_by_ym = df_grouped_by_ym['score']
df_score_grouped_by_m = df_grouped_by_m['score']
# df_score_grouped_by_zip = df_grouped_by_zip['score']
# -

df_ym_described = df_score_grouped_by_ym.describe()
df_m_described = df_score_grouped_by_m.describe()
# df_zip_described = df_score_grouped_by_zip.describe()

'zip_min: {0} -- zip_max: {1}'.format(df_lac['zip_code'].min(), df_lac['zip_code'].max())


def plot_df_described(df_described_to_plot):

    fig, axs = plt.subplots(7,1,figsize=(15,25))

    axs[0].plot(df_described_to_plot.index, df_described_to_plot['max'])
    axs[1].plot(df_described_to_plot.index, df_described_to_plot['75%'])
    axs[2].plot(df_described_to_plot.index, df_described_to_plot['50%'])
    axs[3].plot(df_described_to_plot.index, df_described_to_plot['25%'])
    axs[4].plot(df_described_to_plot.index, df_described_to_plot['min'])
    axs[5].plot(df_described_to_plot.index, df_described_to_plot['mean'])
    axs[6].plot(df_described_to_plot.index, df_described_to_plot['std'])

    axs[0].set_ylabel('max')
    axs[1].set_ylabel('75%')
    axs[2].set_ylabel('50%')
    axs[3].set_ylabel('25%')
    axs[4].set_ylabel('min')
    axs[5].set_ylabel('mean')
    axs[6].set_ylabel('std')


    plt.show()

# ### A look at the described score that's groupedby month of birth
#
# #### Observation:
# - customers born in June seem to have lower average scores

# ----- plots of the score df groupedby month  ----- #
plot_df_described(df_m_described)

# ### A look at the described score that's groupedby year_month of birth
#
# #### Observation:
# - customers seem to be evenly distributed across birth year/birth month
# - not so much for birth month only

# ----- plots of the score df groupedby year_month  ----- #
plot_df_described(df_ym_described)

df_ym_described.boxplot(column=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

# +
fig, axs = plt.subplots(2,2,figsize=(15,10))

bin_count = (df_ym_described['count'].count()) // 8

axs[0,0].plot(df_ym_described['count'])
axs[0,1].hist(df_ym_described[df_ym_described['count']<100]['count'], bins=bin_count)
axs[1,0].plot(df_ym_described[df_ym_described['count']>100]['count'])
axs[1,1].hist(df_ym_described[df_ym_described['count']>100]['count'], bins=bin_count)

plt.show()
# -

df_ym_described[df_ym_described['count']<100]

# +
# ----- just a comparison distribution of the months of birth ----- #

fig, axs = plt.subplots(2,1,figsize=(15,10))

axs[0].plot(df_m_described['count'])
axs[1].hist(df_m_described['count'], bins=12)


plt.show()
# -

# ### A look at the described score that's groupedby zip_code
#
# #### observation: 
#  - zip codes seem to be from 00000 to 99998; weird since there are 100000 records -- Q: is there something wrong with my code?
#      - the mean of the zip codes for both 0 and 1 conversion rates are both ~ 49,960 (when considered an integer)
#      - that, combined with the fact the bdays are to recent, the data could have been created
#  - the max count of any given zip seems to be 4

# +
# df_zip_described.index.unique()

# +
# df_zip_described.boxplot(column=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

# +
# ----- plots of the score df groupedby zip_code  ----- #
# plot_df_described(df_zip_described)

# +
# df_lac.dtypes
# -

# #### df of each column with number value and its mean resampled by year

df_resampled = (
    df_lac.set_index('birthdate')
        .resample('y')
        .agg({'score': 'mean', 
                'days_since_signup':'mean',
                'completed_form':'mean',
                'visited_pricing':'mean',
                'registered_for_webinar':'mean',
                'attended_webinar':'mean',
                'converted':'mean',
                'is_manager':'mean',
                'acquisition_channel_cold_call':'mean',
                'acquisition_channel_cold_email':'mean',
                'acquisition_channel_organic_search':'mean',
                'acquisition_channel_paid_leads':'mean',
                'acquisition_channel_paid_search':'mean',
                'company_size_1_to_10':'mean',
                'company_size_1000_to_10000':'mean',
                'company_size_10001_plus':'mean',
                'company_size_101_to_250':'mean',
                'company_size_11_to_50':'mean',
                'company_size_251_to_1000':'mean',
                'company_size_51_to_100':'mean',
                'industry_financial_services':'mean',
                'industry_furniture':'mean',
                'industry_heavy_manufacturing':'mean',
                'scandanavion_design':'mean',
                'transportation':'mean',
                'internet':'mean',
                'score':'mean',
                'age_seconds':'mean',
                'is_male':'mean',
                'is_director':'mean',
                'is_vp':'mean',
                'is_admin':'mean',
                'is_chief_officer':'mean',
                'is_advertising':'mean',
                'is_sales':'mean',
                'is_marketing':'mean',
                'is_other':'mean',
                'is_web_ind':'mean',
                'is_financials_ind':'mean',
                'is_furniture_ind':'mean',
                'is_transportation_ind':'mean',
                'is_manufacturing_ind':'mean',
                'is_scandesign_ind':'mean',
                'score_normalized':'mean',
                'days_since_signup_normalized':'mean'})
)
df_resampled

# ### A look at the df grouped by converted
#  - creating an aggregated df of the original df grouped by converted
#  - adding a row to show the difference in mean values
#      - a positive difference shows a greater chance of converting than not

df_groupedby_converted = (
    df_lac.groupby('converted')
        .agg({'score': 'mean', 
                'days_since_signup':'mean',
                'completed_form':'mean',
                'visited_pricing':'mean',
                'registered_for_webinar':'mean',
                'attended_webinar':'mean',
                'is_manager':'mean',
                'acquisition_channel_cold_call':'mean',
                'acquisition_channel_cold_email':'mean',
                'acquisition_channel_organic_search':'mean',
                'acquisition_channel_paid_leads':'mean',
                'acquisition_channel_paid_search':'mean',
                'company_size_1_to_10':'mean',
                'company_size_1000_to_10000':'mean',
                'company_size_10001_plus':'mean',
                'company_size_101_to_250':'mean',
                'company_size_11_to_50':'mean',
                'company_size_251_to_1000':'mean',
                'company_size_51_to_100':'mean',
                'industry_financial_services':'mean',
                'industry_furniture':'mean',
                'industry_heavy_manufacturing':'mean',
                'scandanavion_design':'mean',
                'transportation':'mean',
                'internet':'mean',
                'score':'mean',
                'age_seconds':'mean',
                'is_male':'mean',
                'is_director':'mean',
                'is_vp':'mean',
                'is_admin':'mean',
                'is_chief_officer':'mean',
                'is_advertising':'mean',
                'is_sales':'mean',
                'is_marketing':'mean',
                'is_other':'mean',
                'is_web_ind':'mean',
                'is_financials_ind':'mean',
                'is_furniture_ind':'mean',
                'is_transportation_ind':'mean',
                'is_manufacturing_ind':'mean',
                'is_scandesign_ind':'mean',
                'score_normalized':'mean',
                'days_since_signup_normalized':'mean'})
)
df_groupedby_converted

# +
df_converted_diff = df_groupedby_converted.diff()
df_converted_diff = df_converted_diff.rename(index={1: 'diff'}).drop(index=0)
df_aggregated_gb_converted = pd.concat([df_groupedby_converted, df_converted_diff])

df_aggregated_gb_converted
# -

# ### A look at the fields with conversion rates of various thresholds
#
# #### Observations: 
#   - there seems to be a threshold of ~ 14 days since signed up and conversion rate
#       - conversion drops below 50% after ~13.9 days
#       
#       
#   - conversion rate < 50%:
#      - See below
#      
#      
# - conversion rate > 50%:
#      - score
#      - days_since_signup
#      - completed_form
#      - visited_pricing
#      - registered_for_webinar
#      - internet
#      - is_male
#      - is_other
#      - score_normalized
#      
#
# - conversion rate > 75%:
#      - days_since_signup
#      - days_since_signup
#      - completed_form
#      - visited_pricing
#      
#
#  

df_aggregated_gb_converted_T = df_aggregated_gb_converted.T
df_zero_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[:]==0]
df_all_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[:]==1]
df_lt50_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[1]<.5]
df_gt50_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[1]>.5]
df_gt75_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[1]>.75]

df_zero_converted.dropna()

df_all_converted.dropna()

df_gt75_converted.drop(['age_seconds', 'score', 'days_since_signup']).plot(kind='barh')
plt.tight_layout()
plt.show()

df_diff_zero_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T['diff']==0]
df_diff_all_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T['diff']==1]
df_diff_pos_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T['diff']>0]
df_diff_neg_converted = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T['diff']<0]

# ### A look at the fields with lt50% and gt50% failure to convert
#
# ##### Observations: 
#  - fields with a < 50% failure to convert rate WITH a > 5% chance of converting:
#      - registered_for_webinar
#      - attended_webinar
#      - acquisition_channel_organic_search
#      - acquisition_channel_paid_leads
#      - company_size_251_to_1000
#      - industry_financial_services
#      - is_director
#      - score_normalized
#
#
#  - fields with a < 50% failure to convert rate WITH a > 10% chance of converting:
#      - registered_for_webinar
#      - attended_webinar
#      - score_normalized

df_lt50_fail_to_convert = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[0]<.50]
df_gt50_fail_to_convert = df_aggregated_gb_converted_T[df_aggregated_gb_converted_T[0]>.50]

df_gt50_fail_to_convert.drop(['age_seconds', 'score', 'days_since_signup']).plot(kind='barh')
plt.tight_layout()
plt.show()

df_lt50_fail_to_convert.plot(kind='barh')
plt.tight_layout()
plt.show()

df_gt10pct_pos_diff = df_lt50_fail_to_convert[df_lt50_fail_to_convert['diff']>0.09999]
df_gt10pct_pos_diff
df_gt10pct_pos_diff.plot(kind='barh')
plt.title('<50% Failure to Convert rate AND >10% difference in Conversion Rate')
plt.show()

df_gt5pct_pos_diff = df_lt50_fail_to_convert[df_lt50_fail_to_convert['diff']>0.04999]
df_gt5pct_pos_diff.plot(kind='barh')
plt.title('<50% Failure to Convert rate AND >5% difference in Conversion Rate')
plt.show()

# ### A look at the fields that show a positive vs negative mean value difference of the converted df
#
# #### Observations: 
#  - only age_seconds and score have relative differences > 1; rest are normalized
#  - is_male, is_director, and is_other are all positive changes in conversion rate
#  - registering for a webinar has a large positive difference
#
#  

# +
df_aggregated_diff_converted_T = df_aggregated_gb_converted.loc['diff'].T

df_positive_diff = df_aggregated_diff_converted_T[df_aggregated_diff_converted_T[:]>0]
df_negative_diff = df_aggregated_diff_converted_T[df_aggregated_diff_converted_T[:]<0]
df_neutral_diff = df_aggregated_diff_converted_T[df_aggregated_diff_converted_T[:]==0]

df_relative_means_lt1 = df_aggregated_diff_converted_T[df_aggregated_diff_converted_T[:]<1]
df_relative_means_gt1 = df_aggregated_diff_converted_T[df_aggregated_diff_converted_T[:]>1]

df_positive_diff = df_positive_diff.drop(['score', 'age_seconds'])
# -

df_neutral_diff.index

# +
fig, axs = plt.subplots(1,2,figsize=(15,10))

axs[0].barh(df_relative_means_lt1.index, df_relative_means_lt1)
axs[0].set_title('relative differences only (value < 1)')
axs[1].barh(df_relative_means_gt1.index, df_relative_means_gt1)
axs[1].set_title('relative differences only (value > 1)')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    
plt.tight_layout()
plt.show()

# +
fig, axs = plt.subplots(2,2,figsize=(15,15))

axs[0,0].barh(df_negative_diff.index, df_negative_diff)
axs[0,0].set_title('negative difference in score')
axs[1,0].barh(df_positive_diff.index, df_positive_diff)
axs[1,0].set_title('positive difference in score')
axs[0,1].plot(df_neutral_diff.index, df_neutral_diff)
axs[0,1].set_title('neutral difference in score')


for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    
plt.tight_layout()
plt.show()

# -

df_positive_diff

df_negative_diff

df_neutral_diff

# +
# ---- playground for finding unique job_titles and industry terms ---- #

# s = ""
# for c in df_lac.job_title.unique():
#     s += c + " "
    
    
# words = s.split()
# uwords = []
# for w in words:
#     if w not in uwords:
#         uwords.append(w)
        
# uwords
# s
# df_lac.job_title.unique()

# df_lac.industry.unique()

# df_lac.internet.unique()
# -

# ------- done graphing so I don't care how the months are sorted now in the xlabels from earlier ------ #
df_lac['bd_month'] = df_lac.bd_month.apply(lambda x: int(x))

# #### Clean Up:
# - removing non-number value columns
# - creating new df

to_drop = ['address', 
           'birthdate', 
           'mail', 
           'name', 
           'sex', 
           'username', 
           'acquisition_channel', 
           'job_title', 
           'company_size', 
           'industry', 
           'bd_year_month',
           'zip_code']

df_numerical = df_lac.drop(to_drop, axis=1)

df_numerical.max()

df_numerical
df_numerical.to_json('data\\cleaned_df_numerical_columns_only.json')

# ### A test to see differences when duplicate usernames are dropped
#
# #### Observations: 
#  - verly little difference in df grouped by conversion rate
#

df_test = df_lac.groupby(["username"]).filter(lambda df:df.shape[0] == 1)

df_test

df_test.count()

test_df_groupedby_converted = (
    df_test.groupby('converted')
        .agg({'score': 'mean', 
                'days_since_signup':'mean',
                'completed_form':'mean',
                'visited_pricing':'mean',
                'registered_for_webinar':'mean',
                'attended_webinar':'mean',
                'is_manager':'mean',
                'acquisition_channel_cold_call':'mean',
                'acquisition_channel_cold_email':'mean',
                'acquisition_channel_organic_search':'mean',
                'acquisition_channel_paid_leads':'mean',
                'acquisition_channel_paid_search':'mean',
                'company_size_1_to_10':'mean',
                'company_size_1000_to_10000':'mean',
                'company_size_10001_plus':'mean',
                'company_size_101_to_250':'mean',
                'company_size_11_to_50':'mean',
                'company_size_251_to_1000':'mean',
                'company_size_51_to_100':'mean',
                'industry_financial_services':'mean',
                'industry_furniture':'mean',
                'industry_heavy_manufacturing':'mean',
                'scandanavion_design':'mean',
                'transportation':'mean',
                'internet':'mean',
                'score':'mean',
                'age_seconds':'mean',
                'is_male':'mean',
                'is_director':'mean',
                'is_vp':'mean',
                'is_admin':'mean',
                'is_chief_officer':'mean',
                'is_advertising':'mean',
                'is_sales':'mean',
                'is_marketing':'mean',
                'is_other':'mean',
                'is_web_ind':'mean',
                'is_financials_ind':'mean',
                'is_furniture_ind':'mean',
                'is_transportation_ind':'mean',
                'is_manufacturing_ind':'mean',
                'is_scandesign_ind':'mean',
                'score_normalized':'mean',
                'days_since_signup_normalized':'mean'})
)
test_df_groupedby_converted

# +
test_df_converted_diff = test_df_groupedby_converted.diff()
test_df_converted_diff = test_df_converted_diff.rename(index={1: 'diff'}).drop(index=0)
test_df_aggregated_gb_converted = pd.concat([test_df_groupedby_converted, test_df_converted_diff])

test_df_aggregated_gb_converted
# -

# original -- comparing to df where duplicate usernames are dropped
df_aggregated_gb_converted

test_df_aggregated_gb_converted_T = test_df_aggregated_gb_converted.T
test_df_zero_converted = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[:]==0]
test_df_all_converted = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[:]==1]
test_df_lt50_converted = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[1]<.5]
test_df_gt50_converted = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[1]>.5]
test_df_gt75_converted = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[1]>.75]

test_df_lt50_fail_to_convert = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[0]<.50]
test_df_gt50_fail_to_convert = test_df_aggregated_gb_converted_T[test_df_aggregated_gb_converted_T[0]>.50]

test_df_gt10pct_pos_diff = test_df_lt50_fail_to_convert[test_df_lt50_fail_to_convert['diff']>0.09999]
test_df_gt10pct_pos_diff
test_df_gt10pct_pos_diff.plot(kind='barh')
plt.title('<50% Failure to Convert rate AND >10% difference in Conversion Rate')
plt.show()

test_df_gt5pct_pos_diff = test_df_lt50_fail_to_convert[test_df_lt50_fail_to_convert['diff']>0.04999]
test_df_gt5pct_pos_diff.plot(kind='barh')
plt.title('<50% Failure to Convert rate AND >5% difference in Conversion Rate')
plt.show()


