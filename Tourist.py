import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from  sklearn import linear_model
from matplotlib import pyplot as plt
import statsmodels.api as sm


file = pd.read_csv("F:\Important books\datassets\TSIC.csv")
file = pd.DataFrame(file)

min_max = MinMaxScaler()
file1 = min_max.fit_transform(file[['1992', '1993', '1993', '1994', '1995', '1996', '1997','1998', '1999', '2000',
                                    '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                                    '2011', '2012', '2013', '2014', '2015']])

data = pd.DataFrame(file1)
scale = StandardScaler()

X_train = data.ix[:, :23]
Y_train = data.ix[:, 24:]

X_train = scale.fit_transform(X_train.as_matrix())

mean = np.mean(file)
sd = np.std(file)
print mean + 1*sd, mean - 1*sd


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
bp = ax.boxplot(X_train)
fig.savefig('fig1.png', bbox_inches='tight')

lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
X_train = pd.DataFrame(X_train)
X_tra = X_train.ix[:, :23]
predictions = lm.predict(X_tra)
print predictions

predictions = pd.DataFrame(predictions)
predictions.to_csv('Tourist2.csv', index=False, header='2016')

est = sm.OLS(Y_train, X_train).fit()
est.summary()




