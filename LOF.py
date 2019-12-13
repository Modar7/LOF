
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



data = pd.read_csv(r'D:\4\Datasets\Mall_Customers.csv')

data=data.drop('CustomerID',axis=1)

data=data.rename(columns={'Spending Score (1–100)':'Spend_Score', 'Annual Income (k$)':'Income'})


# Convert categorical variable into dummy/indicator variables.
df=pd.get_dummies(data)


# Transform features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num2 = scaler.fit_transform(df)
num2 = pd.DataFrame(num2, columns = df.columns)


def LOF_plot(k):
    import seaborn as sns
    from sklearn.neighbors import LocalOutlierFactor
    var1, var2 = 1, 2
    clf = LocalOutlierFactor(n_neighbors=k, contamination=.1)
    y_pred = clf.fit_predict(df)
    LOF_Scores = clf.negative_outlier_factor_

    plt.title("Local Outlier Factor(LOF), K = {}".format(k))
    plt.scatter(df.iloc[:, var1], df.iloc[:, var2], color='k', s = 3., label ='Data points')
    radius = (LOF_Scores.max() - LOF_Scores) / (LOF_Scores.max() - LOF_Scores.min())
    plt.scatter(df.iloc[:, var1], df.iloc[:, var2], s=1000 * radius, edgecolors='r',
    facecolors ='none', label ='Outlier scores')
    plt.axis('tight')
    plt.ylabel("{}".format(df.columns[var1]))
    plt.xlabel("{}".format(df.columns[var2]))
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show();


LOF_plot(5)
LOF_plot(30)
LOF_plot(70)





