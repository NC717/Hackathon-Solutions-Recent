import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, scale, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm


X_train = pd.read_csv("F:\Important books\Datasets\X_train.csv")
Y_train = pd.read_csv("F:\Important books\Datasets\Y_train.csv")

X_test = pd.read_csv("F:\Important books\Datasets\X_test.csv")
Y_test = pd.read_csv("F:\Important books\Datasets\Y_test.csv")
print X_train.head()

X_train[X_train.dtypes[(X_train.dtypes == "float64") | (X_train.dtypes == "int64")].index.values].hist(figsize=[11, 11])
plt.show()

min_max = MinMaxScaler()

X_train_minmax = min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                'Loan_Amount_Term', 'Credit_History']])
X_test_minmax = min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                              'Loan_Amount_Term', 'Credit_History']])

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_minmax, Y_train)

print accuracy_score(Y_test, clf.predict(X_test_minmax))

clf1 = LogisticRegression(C=1.0, penalty='l2')
clf1.fit(X_train_minmax, Y_train)

print accuracy_score(Y_test, clf1.predict(X_test_minmax))

X_train_scale = scale(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                'Loan_Amount_Term', 'Credit_History']])

X_test_scale = scale(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                'Loan_Amount_Term', 'Credit_History']])

Log = LogisticRegression(penalty='l2', C=1.0)
Log.fit(X_train_scale, Y_train)
print accuracy_score(Y_test, Log.predict(X_test_scale))

# Label Encoding
le = LabelEncoder()
for col in X_test.columns.values:
    if X_test[col].dtypes == 'object':
        data = X_train[col].append(X_test[col])
        le.fit(data.values)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])


print X_train.head()
# One hot encoding


support = svm.SVC(kernel='linear', C=1.0)
support.fit(X_train_scale, Y_train)
print accuracy_score(Y_test, support.predict(X_test_scale))



