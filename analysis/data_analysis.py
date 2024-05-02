import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('../data/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../data/test.csv', index_col='PassengerId')

target = 'Survived'


def check_nans(df: pd.DataFrame) -> dict:
    """
    Returns the percentage of missing values in a given column
    but only if they exist
    """
    columns = list(df.columns)

    cols_w_nans = dict()

    for col in columns:
        n_nans = df.loc[df[col].isna(), col].shape[0]
        if n_nans > 0:
            cols_w_nans[col] = n_nans / df.shape[0]

    return dict(sorted(cols_w_nans.items(), key=lambda item: item[1], reverse=True))


'''
Check for Nan values
'''
print("Columns with missing values:")
print("\ttrain_df:")
print("\t\t", check_nans(train_df))
print("\ttest_df")
print("\t\t", check_nans(test_df))
print()

'''
Check for duplicates
'''
print("Columns with duplicated values:")
print("\ttrain_df:")
print("\t\t", train_df[train_df[test_df.columns].duplicated()])
print("\ttest_df")
print("\t\t", test_df[test_df[test_df.columns].duplicated()])
print()

'''
Plot for quantitative (numerical) predictors distribution
'''
numeric_columns = test_df.select_dtypes(include=np.number).columns
categorical_columns = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1).select_dtypes(include='object').columns

fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(12, len(numeric_columns) * 5))

for i, col in enumerate(numeric_columns):
    train_hist = sns.histplot(train_df[col], ax=axes[i, 0], kde=True)
    axes[i, 0].set_title(f'Train: {col} Distribution')
    sns.histplot(test_df[col], ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'Test: {col} Distribution')

plt.tight_layout()
plt.savefig('quantitative.png')
plt.show()

'''
Plot for qualitative (categorical) predictors distribution
'''
fig, axes = plt.subplots(len(categorical_columns), 2, figsize=(12, len(categorical_columns) * 5))

for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=train_df, ax=axes[i, 0], order=train_df[col].value_counts().index)
    axes[i, 0].set_title(f'Train: {col}')

    sns.countplot(x=col, data=test_df, ax=axes[i, 1], order=test_df[col].value_counts().index)
    axes[i, 1].set_title(f'Test: {col}')

plt.tight_layout()
plt.savefig('qualitative.png')
plt.show()

'''
Plot for target (label) distribution
'''
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
sns.countplot(train_df, x=target, ax=ax[0])
ax[0].set_title("Target distribution")
ax[1].pie(train_df[target].value_counts(),  labels=train_df[target].unique(), autopct='%1.1f%%')
ax[1].set_title("Target distribution")
plt.savefig('target.png')
fig.show()

'''
Plot for correlation matrix
'''
# Clear from junk
train_df['pre_ticket'] = train_df['Ticket'].str.extract(r'(.*?) ')
test_df['pre_ticket'] = test_df['Ticket'].str.extract(r'(.*?) ')
preprocessed_tickets = train_df['Ticket'].str.extract(r' ([0-9]+)').dropna()
train_df.loc[preprocessed_tickets.index, 'Ticket'] = preprocessed_tickets.to_numpy()
preprocessed_tickets = test_df['Ticket'].str.extract(r' ([0-9]+)').dropna()
test_df.loc[preprocessed_tickets.index, 'Ticket'] = preprocessed_tickets.to_numpy()
train_df.loc[train_df['Ticket'] == 'LINE', 'Ticket'] = -1
test_df.loc[test_df['Ticket'] == 'LINE', 'Ticket'] = -1

train_df['Ticket'] = train_df['Ticket'].astype(int)
corr_matrix = train_df[[*numeric_columns, 'Ticket', 'Survived']].corr()
mask = np.triu(np.ones_like(corr_matrix))
plt.figure()
sns.heatmap(corr_matrix, mask=mask, annot=True)
plt.title('Correlation matrix')
plt.savefig('correlation.png')
plt.show()
