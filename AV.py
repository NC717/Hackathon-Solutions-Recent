import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, scale, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


X_train = pd.read_csv("F:\Important books\Datasets\AV_train1.csv")
X_test = pd.read_csv("F:\Important books\Datasets\Av_test1.csv")

# Data Pre processing

#le = LabelEncoder()
#for col in X_train.columns.values:
 #   if X_train[col].dtypes == "object":
 #       data = X_train[col].append(X_train[col])
 #       le.fit(data)
 #       X_train[col] = le.transform(X_train[col])
 #       X_test[col] = le.transform(X_test[col])

# Used this instead of LabelEncoder
d = {'Male': 0, 'Femmale': 1}
e = {'No': 0, 'Yes': 1}
f = {'Graduate':0, 'Not Graduate':1}
g = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}

X_train['Gender'] = X_train['Gender'].map(d)
X_train['Self_Employed'] = X_train['Self_Employed'].map(e)
X_train['Education'] = X_train['Education'].map(f)
X_train['Property_Area'] = X_train['Property_Area'].map(g)
X_train['Married'] = X_train['Married'].map(e)

X_test['Gender'] = X_test['Gender'].map(d)
X_test['Self_Employed'] = X_test['Self_Employed'].map(e)
X_test['Education'] = X_test['Education'].map(f)
X_test['Property_Area'] = X_test['Property_Area'].map(g)
X_test['Married'] = X_test['Married'].map(e)

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

# Dataset Breaking

X_tra = X_train.ix[:, 1:12]
Y_tra = X_train.ix[:, -1:]
X_tes = X_test.ix[:, 1:13]
print X_tra.head()
print X_tes.head()
h = {'N': 0, 'Y': 1}
Y_tra['Loan_Status'] = Y_tra['Loan_Status'].map(h)

print Y_tra.head()

mm1 = Imputer(missing_values='NaN', strategy='mean',axis=0)
mm1 = mm1.fit(X_tes)
X_tes = mm1.transform(X_tes)
mm2 = mm1.fit(X_tra)
X_tra = mm2.transform(X_tra)

# Applying pca
pca = PCA(n_components=5, whiten=True).fit(X_tra)
pca1 = PCA(n_components=5, whiten=True).fit(X_tes)
X_ra = pca.transform(X_tra)
X_te = pca1.transform(X_tes)

X_ra = pd.DataFrame(X_ra)
X_te = pd.DataFrame(X_te)

print X_ra
print Y_tra

clf = RandomForestClassifier(n_estimators=1000)
cv = cross_val_score(clf, X_ra, Y_tra.values.ravel(), cv=15)
print cv
cv = pd.DataFrame(cv)
clf.fit(X_ra, Y_tra.values.ravel())
pdi = clf.predict(X_te)
loan_id = X_test['Loan_ID']
Out = DataFrame(pdi, loan_id)
cv.to_csv('ut.csv', index=False, header='False')









