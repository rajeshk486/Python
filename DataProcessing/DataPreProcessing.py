# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:16:46 2017

@author: Kartz
"""

# Ananlysis of Credit Card Data using Supervised and Unsupervised Methods.


# Importing the Necessary Packages


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the Data
credit = pd.DataFrame(pd.read_csv("/home/hadoop/Downloads/UCI_Credit_Card.csv"))

# Reviewing the Data

credit.head()

credit.keys()

credit.describe()

# Dropping unwanted columns

credit = credit.drop('ID', axis=1)
credit.keys()

# Summary of the data - Visual Exploratory Analysis reference from Kaggle -
# Gender (Feature name :'SEX' ) - legend 1-Male, 2-Female

men = credit['SEX'][credit['SEX'] == 1].count()
men_default_yes = credit['SEX'][(credit['SEX'] == 1) & (credit['default.payment.next.month'] == 1)].count()
men_default_no = credit['SEX'][(credit['SEX'] == 1) & (credit['default.payment.next.month'] == 0)].count()

women = credit['SEX'][credit['SEX'] == 2].count()
women_default_yes = credit['SEX'][(credit['SEX'] == 2) & (credit['default.payment.next.month'] == 1)].count()
women_default_no = credit['SEX'][(credit['SEX'] == 2) & (credit['default.payment.next.month'] == 0)].count()

M = (men, men_default_yes, men_default_no)
F = (women, women_default_yes, women_default_no)
N = (1, 2, 3)

plt.figure()
ax = plt.subplot(111)
w = np.min(np.diff(N)) / 3
ax.bar(N, M, w, color='b', align='center', label='Male')
ax.bar(N + w, F, w, color='r', align='center', label='Female')
ax.set_ylabel('Number of Accounts')
ax.set_xticks(N + w)
ax.set_xticklabels(('Total', 'Default_payment_yes', 'Default_payment_no'))
ax.legend()
plt.show()

# identifying any outliers in the data.
plt.figure()
sns.boxplot(x='SEX', hue='default.payment.next.month', data=credit)
plt.show()

# Gender (Feature name :'EDUCATION' )
# legend:1=graduate school, 2=university, 3=high school,0=unknown 4=others, 5=unknown, 6=unknown
# since there are unknowns we are taking the count of knowns to plot and all unknowns to a single value
grad = credit['EDUCATION'][credit['EDUCATION'] == 1].count()
grad_default_yes = credit['EDUCATION'][(credit['EDUCATION'] == 1) & (credit['default.payment.next.month'] == 1)].count()
grad_default_no = credit['EDUCATION'][(credit['EDUCATION'] == 1) & (credit['default.payment.next.month'] == 0)].count()

Uni = credit['EDUCATION'][credit['EDUCATION'] == 2].count()
uni_default_yes = credit['EDUCATION'][(credit['EDUCATION'] == 2) & (credit['default.payment.next.month'] == 1)].count()
uni_default_no = credit['EDUCATION'][(credit['EDUCATION'] == 2) & (credit['default.payment.next.month'] == 0)].count()

high = credit['EDUCATION'][credit['EDUCATION'] == 3].count()
high_default_yes = credit['EDUCATION'][(credit['EDUCATION'] == 3) & (credit['default.payment.next.month'] == 1)].count()
high_default_no = credit['EDUCATION'][(credit['EDUCATION'] == 3) & (credit['default.payment.next.month'] == 0)].count()

Others = credit['EDUCATION'][(credit['EDUCATION'] == 0) | (credit['EDUCATION'] > 3)].count()
Others_default_yes = credit['EDUCATION'][
    (credit['EDUCATION'] == 0) | (credit['EDUCATION'] > 3) & (credit['default.payment.next.month'] == 1)].count()
Others_default_no = credit['EDUCATION'][
    (credit['EDUCATION'] == 0) | (credit['EDUCATION'] > 3) & (credit['default.payment.next.month'] == 0)].count()

Total = (grad, Uni, high, Others)
def_yes = (grad_default_yes, uni_default_yes, high_default_yes, Others_default_yes)
def_no = (grad_default_no, uni_default_no, high_default_no, Others_default_no)

N = (1, 2, 3, 4)
w = np.min(np.diff(N)) / 4

plt.figure()
ax = plt.subplot(111)

ax.bar(N, Total, w, color='b', align='center', label='Total')
ax.bar(N + w, def_no, w, color='r', align='center', label='Df_pay_yes')
ax.bar(N + 2 * w, def_no, w, color='g', align='center', label='Df_pay_no')

ax.set_ylabel('Number of Accounts')
ax.set_xticks(N + w)
ax.set_xticklabels(('Grad School', 'University', 'High School', 'Others'))
ax.legend()
plt.show()

# identifying any outliers in the data.
plt.figure()
sns.boxplot(x='EDUCATION', hue='default.payment.next.month', data=credit)
plt.show()

# Gender (Feature name :'AGE' )
# since age data is not categorical as like other columns for easier plot we are taking ranges from 20-80 we split it as adults(21-40),Elder(41-60),
# we are not considering older people as they are in the outliers explained in box plot below Old61( and above)


adults = credit['AGE'][(credit['AGE'] <= 40)].count()
adults_default_yes = credit['AGE'][(credit['AGE'] <= 40) & (credit['default.payment.next.month'] == 1)].count()
adults_default_no = credit['AGE'][(credit['AGE'] <= 40) & (credit['default.payment.next.month'] == 0)].count()

elder = credit['AGE'][(credit['AGE'] >= 41) & (credit['AGE'] <= 60)].count()
elder_default_yes = credit['AGE'][
    (credit['AGE'] >= 41) & (credit['AGE'] <= 60) & (credit['default.payment.next.month'] == 1)].count()
elder_default_no = credit['AGE'][
    (credit['AGE'] >= 41) & (credit['AGE'] <= 60) & (credit['default.payment.next.month'] == 0)].count()

Old = credit['AGE'][(credit['AGE'] >= 61)].count()
old_default_yes = credit['AGE'][(credit['AGE'] >= 61) & (credit['default.payment.next.month'] == 1)].count()
old_default_no = credit['AGE'][(credit['AGE'] >= 61) & (credit['default.payment.next.month'] == 0)].count()

Total = (adults, elder, Old)
def_yes = (adults_default_yes, elder_default_yes, old_default_yes)
def_no = (adults_default_no, elder_default_no, old_default_no)

N = (1, 2, 3)
w = np.min(np.diff(N)) / 3

plt.figure()
ax = plt.subplot(111)

ax.bar(N + w, Total, w, color='b', align='center', label='Total')
ax.bar(N + 2 * w, def_no, w, color='r', align='center', label='Df_pay_yes')
ax.bar(N + 3 * w, def_no, w, color='g', align='center', label='Df_pay_no')

ax.set_ylabel('Number of Accounts')
ax.set_xticks(N + w)
ax.set_xticklabels(('Adults(age<40)', 'Elder(Age 41-60)', 'Old(Age>60)'))
ax.set_title('Feature_name:Age')
ax.legend()
plt.show()

# identifying any outliers in the data.
plt.figure()
plt.boxplot(x='AGE', data=credit)
plt.xlabel('AGE')
plt.show()

# Gender (Feature name :'MARRIAGE' )
# legend:1=married, 2=single, 3=others
Married = credit['MARRIAGE'][credit['MARRIAGE'] == 1].count()
Married_default_yes = credit['MARRIAGE'][
    (credit['MARRIAGE'] == 1) & (credit['default.payment.next.month'] == 1)].count()
Married_default_no = credit['MARRIAGE'][(credit['MARRIAGE'] == 1) & (credit['default.payment.next.month'] == 0)].count()

Single = credit['MARRIAGE'][credit['MARRIAGE'] == 2].count()
Single_default_yes = credit['MARRIAGE'][(credit['MARRIAGE'] == 2) & (credit['default.payment.next.month'] == 1)].count()
Single_default_no = credit['MARRIAGE'][(credit['MARRIAGE'] == 2) & (credit['default.payment.next.month'] == 0)].count()

Other = credit['MARRIAGE'][credit['MARRIAGE'] == 3].count()
Other_default_yes = credit['MARRIAGE'][(credit['MARRIAGE'] == 3) & (credit['default.payment.next.month'] == 1)].count()
Other_default_no = credit['MARRIAGE'][(credit['MARRIAGE'] == 3) & (credit['default.payment.next.month'] == 0)].count()

M = (Married, Married_default_yes, Married_default_no)
S = (Single, Single_default_yes, Single_default_no)
O = (Other, Other_default_yes, Other_default_no)
N = (1, 2, 3)

plt.figure()
ax = plt.subplot(111)
w = np.min(np.diff(N)) / 3
ax.bar(N, M, w, color='b', align='center', label='Married')
ax.bar(N + w, S, w, color='r', align='center', label='Single')
ax.bar(N + 2 * w, O, w, color='g', align='center', label='Other')

ax.set_ylabel('Number of Accounts')
ax.set_xticks(N + w)
ax.set_xticklabels(('Total', 'Default_payment_yes', 'Default_payment_no'))
ax.set_title('Feature_name:Marriage')
ax.legend()
plt.show()

# identifying any outliers in the data.
plt.figure()
sns.boxplot(x='MARRIAGE', hue='default.payment.next.month', data=credit)
plt.show()

# Payment status (Feature name :'PAY_' )
# Payment Default Prediction Neural Network - https://www.kaggle.com/mahyar511/payment-default-prediction-neural-network
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0

plt.suptitle('Fig.6 : Payment Status', fontweight="bold", fontsize=22)
for cn in features[5:11]:
    ax = plt.subplot(gs[i])
    delay = np.zeros(12)
    delay_default_yes = np.zeros(12)
    delay_default_no = np.zeros(12)
    for j in np.arange(0, 12):
        delay[j] = credit[cn][credit[cn] == j - 2].count()
        delay_default_yes[j] = credit[cn][(credit[cn] == j - 2) & (credit['default.payment.next.month'] == 1)].count()
        delay_default_no[j] = credit[cn][(credit[cn] == j - 2) & (credit['default.payment.next.month'] == 0)].count()
    month = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.bar(month, delay, color='c', alpha=0.5, label='Total')
    plt.bar(month, delay_default_yes, color='k', alpha=0.5, label='Default_yes')
    plt.bar(month, delay_default_no, color='r', alpha=0.5, label='Default_no')

    plt.xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               ['0 Bill', 'Duly', 'Partial', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontweight="bold", size=10)
    ax.set_xlabel('Delay (month)')
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('Payment status in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# identifying any outliers in the data.
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0

plt.suptitle('Fig.6 : Payment Status', fontweight="bold", fontsize=22)
for cn in features[5:11]:
    ax = plt.subplot(gs[i])
    plt.boxplot(cn, data=credit)
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('outliers in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# Payment status (Feature name :'Bill_AMT' )
# Payment Default Prediction Neural Network - https://www.kaggle.com/mahyar511/payment-default-prediction-neural-network
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0
bins = 25
plt.suptitle('Fig.6 : BILL AMT Status', fontweight="bold", fontsize=22)
for cn in features[11:17]:
    ax = plt.subplot(gs[i])
    plt.hist(credit[cn], bins=bins, color='lightblue', label='Total', alpha=1)
    plt.hist(credit[cn][credit['default.payment.next.month'] == 1], bins=bins, color='k', label='Default', alpha=0.5)

    plt.xlabel('Amount (NT dollar)')
    plt.ylabel('Number of Accounts')
    ax.set_yscale('log', nonposy='clip')
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('Bill Amount status in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# identifying any outliers in the data.
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0

plt.suptitle('Fig.6 : Bill_AMT Status', fontweight="bold", fontsize=22)
for cn in features[11:17]:
    ax = plt.subplot(gs[i])
    plt.boxplot(cn, data=credit)
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('outliers in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# Payment status (Feature name :'Pay_amt' )
# Payment Default Prediction Neural Network - https://www.kaggle.com/mahyar511/payment-default-prediction-neural-network
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0
bins = 25
plt.suptitle('Fig.6 : Pay AMT Status', fontweight="bold", fontsize=22)
for cn in features[17:23]:
    ax = plt.subplot(gs[i])
    plt.hist(credit[cn], bins=bins, color='lightblue', label='Total', alpha=1)
    plt.hist(credit[cn][credit['default.payment.next.month'] == 1], bins=bins, color='k', label='Default', alpha=0.5)

    plt.xlabel('Amount (NT dollar)')
    plt.ylabel('Number of Accounts')
    ax.set_yscale('log', nonposy='clip')
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('pay Amount status in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# identifying any outliers in the data.
import matplotlib.gridspec as gridspec

features = list(credit.columns)
plt.figure(figsize=(14, 10))

gs = gridspec.GridSpec(3, 2)
i = 0

plt.suptitle('Fig.6 : Bill_AMT Status', fontweight="bold", fontsize=22)
for cn in features[17:23]:
    ax = plt.subplot(gs[i])
    plt.boxplot(cn, data=credit)
    months = ['Spetember', 'August', 'July', 'June', 'May', 'April']
    ax.set_xlabel(months[i])
    ax.set_title('outliers in ' + months[i], fontweight="bold", size=12)
    ax.legend()
    i += 1

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()


# count of data in each column
def total_count(x):
    data = dict()
    for i in x.unique():
        data[i] = np.sum(x == i)
    return data


education = total_count(credit['EDUCATION'])
print(
    'Count for Levels of education 1=graduate school, 2=university, 3=high school,0=unknown 4=others, 5=unknown, 6=unknown: \n',
    education)
sex = total_count(credit['SEX'])
print('Count for 1-Male 2-Female: \n', sex)
marital = total_count(credit['MARRIAGE'])
print('Count for marital status 1=married, 2=single, 0,3=others: \n', marital)

features1 = list(credit.columns)


def payment_count(x):
    data = dict()
    for cn in features1[5:11]:
        data[cn] = total_count(credit[cn])
    payment = pd.DataFrame(data)
    return payment


pay = payment_count(credit)
print('count of payment/due from April - september', pay)

# Clean up of Data and outliers
credit.keys()

# cleaning outliers from education
credit_new = credit.drop(credit[(credit['EDUCATION'] == 0) | (credit['EDUCATION'] >= 4)].index)
# cleaning outliers from age >60
credit_new = credit_new.drop(credit_new[(credit_new['AGE'] >= 60)].index)
# cleaning outliers from MARRAIGE >60
credit_new = credit_new.drop(credit_new[(credit_new['MARRIAGE'] == 0) | (credit_new['MARRIAGE'] >= 3)].index)
credit_new.describe()
# Drop from all pay_x categories
features2 = list(credit_new.columns)
for cn in features2[5:11]:
    credit_new = credit_new.drop(credit_new[(credit_new[cn] >= 3)].index)

# count of data in each column after data removal


education = total_count(credit_new['EDUCATION'])
print(
    'Count for Levels of education 1=graduate school, 2=university, 3=high school,0=unknown 4=others, 5=unknown, 6=unknown: \n',
    education)
sex = total_count(credit_new['SEX'])
print('Count for 1-Male 2-Female: \n', sex)
marital = total_count(credit_new['MARRIAGE'])
print('Count for marital status 1=married, 2=single, 0,3=others: \n', marital)

credit_new.shape
pay = payment_count(credit_new)
print('count of payment/due from April - september', pay)

# Modelling with Decision Tree and plotting

from sklearn.cross_validation import train_test_split


def decision_tree_plot(X, model):
    data = {1: X.values[:, 1:5], 2: X.values[:, 5:11], 3: X.values[:, 11:17], 4: X.values[:, 17:23],
            5: X.values[:, 5:23], 6: X.values[:, 1:23]}
    score = {}
    score1 = {}
    score2 = {}
    Result = {}
    for key in data:
        a = data[key]
        b = X.values[:, 23]
        X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.25)
        model.fit(a, b)
        Y_pred = model.predict(X_test)
        score[key] = model.fit(X_train, Y_train).score(X_test, Y_test)
        score1[key] = model.fit(X_train, Y_train).score(X_test, Y_pred)
        score2[key] = metrics.accuracy_score(Y_test, Y_pred)
    Result['test score'] = score
    Result['predicted score'] = score1
    Result['accuracy score'] = score2
    plt.plot(*zip(*sorted(Result['accuracy score'].items())), label='accuracy score')
    plt.plot(*zip(*sorted(Result['predicted score'].items())), label='predicted score')
    plt.plot(*zip(*sorted(Result['test score'].items())), label='test score')
    plt.ylabel('Score')
    plt.xlabel(('Categories'))
    plt.title('Initial Data:Score comparison : Model- Decision Tree')
    plt.legend()
    plt.show()
    return Result


model1 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=6, splitter='best')
credit_score = decision_tree_plot(credit, model1)
credit_new_score = decision_tree_plot(credit_new, model1)

model2 = LogisticRegression()

Logistic_score = decision_tree_plot(credit, model2)
Logistic_new_score = decision_tree_plot(credit_new, model2)

# List scores for analysis

print('Decision Tree \n', pd.DataFrame(credit_score))
print('Decision Tree \n', pd.DataFrame(credit_new_score))

print('Logistic Regression \n', pd.DataFrame(Logistic_score))
print('Logistic Regression \n', pd.DataFrame(Logistic_new_score))

# Decision Tree for the selected category

X = credit_new.values[:, 5:11]
Y = credit_new.values[:, 23]
from sklearn.cross_validation import train_test_split
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, splitter='best')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
tree.export_graphviz(model, out_file='decisionTree.dot', feature_names=credit_new.keys().values[5:11],
                     class_names=credit_new.keys().values[23], filled=True, rounded=True, special_characters=True)

# Logistic Regression Plotting

from sklearn.linear_model import LogisticRegression

X = credit_new.values[:, 5:11]
Y = credit_new.values[:, 23]
lm = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
print(lm.intercept_)
print(lm.coef_)
a = credit_new.keys()
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))
print(metrics.mean_squared_error(Y_test, predicted))
print(metrics.mean_absolute_error(Y_test, predicted))

# plotting regression line
a = credit_new['PAY_0'][(credit_new['default.payment.next.month'] == 1)]
b = credit_new['PAY_0'][(credit_new['default.payment.next.month'] == 0)]
ay = credit_new['default.payment.next.month'][(credit_new['default.payment.next.month'] == 1)]
by = credit_new['default.payment.next.month'][(credit_new['default.payment.next.month'] == 0)]
plt.scatter(a, ay, color='red', label='default_yes')
plt.scatter(b, by, color='green', label='default_no')
w0 = lm.intercept_
w1 = lm.coef_[0][0]
w2 = lm.coef_[0][1]
X = np.array([-2, 2])
plt.plot(X, (-w0 - w1 * X) / w2)
plt.title('Regression Line')
plt.xlabel('PAY_0')
plt.ylabel('default.payment.next.month')
plt.legend()
plt.ylim(-1, 2)
plt.show()


# Unsupervised methods
def unsupervised_plot(X, model):
    data = {1: X.values[:, 1:5], 2: X.values[:, 5:11], 3: X.values[:, 11:17], 4: X.values[:, 17:23],
            5: X.values[:, 5:23], 6: X.values[:, 1:23]}
    complete_score = {}
    Homogenity_score = {}
    Result = {}
    for key in data:
        a = scale(data[key])
        b = len(np.unique(X['default.payment.next.month']))
        sample, features = a.shape
        model.fit(a)
        complete_score[key] = metrics.completeness_score(X['default.payment.next.month'], model.labels_)
        Homogenity_score[key] = metrics.homogeneity_score(X['default.payment.next.month'], model.labels_)
    Result['complete_score'] = complete_score
    Result['Homogenity_score'] = Homogenity_score
    return Result


affinity = ('Euclidean', 'Hamming', 'Cosine')
from sklearn import cluster
from sklearn.preprocessing import scale

for i in affinity:
    model_un = cluster.AgglomerativeClustering(n_clusters=len(np.unique(credit['default.payment.next.month'])),
                                               linkage='average', affinity=i)
    credit_unsupervised = unsupervised_plot(credit, model_un)
    print(i, 'Supervised \n', pd.DataFrame(credit_unsupervised))
    credit_unsupervised = unsupervised_plot(credit_new, model_un)
    print(i, 'UnSupervised \n', pd.DataFrame(credit_unsupervised))

# Dendogram
from scipy.cluster.hierarchy import dendrogram, linkage

model = linkage(credit_new, 'ward')
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('PAY_0')
plt.ylabel('default.payment.next.month')
dendrogram(model, leaf_rotation=180., leaf_font_size=2., )
plt.show()

model = linkage()