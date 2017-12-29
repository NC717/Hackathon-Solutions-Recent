import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA


# Importing the data set
data = pd.read_csv("F:\Important books\datassets\Renewal_Prediction.csv")

# Data Pre processing

d = {'a': 0, 'b': 1}
data['C1'] = data['C1'].map(d)
d1 = {'u': 0, 'y': 1, 'l': 2, 't': 3}
data['C4']= data['C4'].map(d1)
d2 = {'g': 0, 'p': 1, 'gg': 2}
data['C5'] = data['C5'].map(d2)
d3 = {'c':0, 'd': 1, 'cc': 2, 'i': 3, 'j': 4, 'k': 5, 'm': 6, 'r': 7, 'q': 8, 'w': 9, 'x': 10, 'e': 11, 'aa':12, 'ff': 13}
data['C6'] = data['C6'].map(d3)
d4 = {'v': 0, 'h': 1, 'bb': 2, 'j': 3, 'n': 4, 'z': 5, 'dd': 6, 'ff': 7, 'o': 8}
data['C7'] = data['C7'].map(d4)
d5 = {'t':0, 'f':1}
data['C9'] = data['C9'].map(d5)
data['C10'] = data['C10'].map(d5)
data['C12'] = data['C12'].map(d5)
d6 = {'g':0,'p':1,'s': 2}
data['C13'] = data['C13'].map(d6)


# Dropping all unavailable values
data = pd.DataFrame(data)

data = data.replace(['?'], ['NaN'])
print data.head(10)

imp = Imputer(missing_values='NaN', strategy='median', axis =0)
imp = imp.fit(data)
dataN = imp.transform(data)

dataN = pd.DataFrame(dataN)
print dataN.head(10)
df = list(dataN.columns[:15])


# Selection of features and Variable to be predicted

X = dataN[df]
Y = data['Renewal']
X1 = np.array(X)

print X.head()

# Generating a correlation Matrix
corr = X.corr()
axes = plt.axes()
axes.legend("Heatmap Depicting a CORRELATION between various variables")
plt.imshow(corr, cmap='hot', interpolation='nearest')
plt.show()

for row in corr.columns:
    for col in corr.columns:
        if np.abs(corr[row][col]) > 0.4:
            if row is not col:
                print row, col


# Min Max scaler for dataframe X
minmax = MinMaxScaler()
X = minmax.fit_transform(X[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
X = pd.DataFrame(X)
print X.head(10)

# Perform Train test split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=111)


X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

y_train = np.array(y_train)
y_test = np.array(y_test)


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print len(predicted)
print metrics.accuracy_score(y_test, predicted)

scores = []
# Repeating same operation to improve accuracy
for i in range(100, 1000,100):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    predicted_rf = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, predicted_rf))

# Slight increase in the accuracy
print np.mean(scores)


# Principal component analysis
pca = PCA(n_components=6, whiten=True).fit(X)
X = pca.transform(X)
X = pd.DataFrame(X)
print  X.head()
# Perform Train test split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=111)


X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

y_train = np.array(y_train)
y_test = np.array(y_test)


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print len(predicted)
print metrics.accuracy_score(y_test, predicted)


scores = []
# Repeating same operation to improve accuracy
for i in range(100, 1000,100):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    predicted_rf = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, predicted_rf))

# Slight increase in the accuracy
print np.mean(scores)














