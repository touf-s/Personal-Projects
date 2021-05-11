import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic.csv')
# Sibsp: total # of siblings+ spouse
# Parch: total count of Parents +  children
# Using all features, predict if the passengers survived or not
print(train)

# Step 1: Search for null values using heatmap visualization
print(train.info())  # Displays num non-null values and dtype of features
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()  # All null values are displayed in yellow
# Around 20% of age data is missing. Can be reasonably replaced with some sort of imputation
# Cabin column is missing too much data, so will be later dropped or changed

# Step 2: Extract info from count visualization (single or multiple features)
# Understand num features with histograms, understand cat features with value counts
# Look at numeric and categorical values separately
df_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
for i in df_num:
    sns.displot(df_num[i].dropna(), kde=False, bins=40)
    plt.show()
sns.heatmap(df_num.corr())  # Shows correlation between the num features
plt.show()
pivot_table = pd.pivot_table(train, index='Survived', values=['Age', 'SibSp', 'Parch', 'Fare'])
print(pivot_table)  # Shows average of features with respect to survival status


sns.countplot(x='Survived', data=train)
plt.show()
sns.countplot(x='Survived', hue='Sex', data=train)
plt.show()
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
plt.show()

sns.distplot(train['Age'].dropna(), kde=False, bins=40)  # Histogram of count vs age. If kde was not false, would give density vs age
plt.show()  # dropna() function drops the null values

sns.countplot(x='SibSp', data=train)  # count of people who had sibling/spouse
plt.show()

# Step 3: Data Cleaning
# We want to fill in the missing age data
# Maybe we can find a relationship between th Pclass and average age
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
plt.show()
# We cann see that the higher class passengers are older on average.
# We can use these averages to impute


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)  # Axis = 1 applies function to each column
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# Now we will drop the cabin feature, too many null values
train.drop('Cabin', axis=1, inplace=True)  # Drops all cabin values
train.dropna(inplace=True)  # Drops nulls in Embarked feature, which is categorical
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Step 4: Converting Categorical Features:
embark = pd.get_dummies(train['Embarked'], drop_first=True)  # drop_first removes the first column
# In output: 1 0 means Q, 0 1 means S, 0 0 means P. First column (P) is not required, so it is excluded
sex = pd.get_dummies(train['Sex'], drop_first=True)
print(embark)
print(sex)
# Now, categorical features (sex and embarked) are converted to int
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)  # Drop all categorical features
train = pd.concat([train, sex, embark], axis=1)  # Concatenates train with updated sex and embark columns
print(train.head())

# Step 5: Scale data

train_max = train.max()
train = train.divide(train_max)
print(train.describe)




# Step 5: Classify dataset into independent and dependent features
# print(train.drop('Survived', axis=1).head()) # dataframe with only independent (input) features
# print(train['Survived'].head) # our output
# Step 6: Build a Logistic Regression Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# store all columns except last one as inputs in x
x = train.iloc[:, 0:-1].values
# store last column as output in y
y = train.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)


logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test, predictions)
print(accuracy)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

