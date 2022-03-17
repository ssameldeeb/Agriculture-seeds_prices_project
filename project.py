import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


data = pd.read_csv("rice_wheat_corn_prices.csv")

print(data.shape)
print(data.columns.values)
print(data.info())
print(data.isna().sum())

data = data.dropna()
print(data.isna().sum().sum())

col = ["Price_rice_ton", "Price_wheat_ton", "Price_corn_ton",
       "Price_rice_ton_infl", "Price_wheat_ton_infl", "Price_corn_ton_infl"]

plt.figure(figsize=(15.5,9))
for (x,y) in zip(col,range(len(col))):
    plt.subplot(2,3,y+1)
    sns.distplot(data[x], color="g")
plt.show()

# col = ["Price_rice_ton_infl", "Price_wheat_ton_infl", "Price_corn_ton_infl"]

# plt.figure(figsize=(15.5,5))
# for (x,y) in zip(col,range(3)):
#     plt.subplot(1,3,y+1)
#     sns.distplot(data[x])
# plt.show()

print(data.dtypes)
print(data.head())

La = LabelEncoder()
data["Month"] = La.fit_transform(data["Month"])
print(data.dtypes)

sns.countplot(data["Month"])
plt.show()

plt.figure(figsize=(15,7))
print(data["Inflation_rate"].nunique())
sns.countplot(data["Inflation_rate"])
plt.show()

plt.figure(figsize=(14,7))
sns.heatmap(data.corr(), annot=True, cmap="hot")
plt.show()

data["category"] = 0
data.loc[(data["Price_rice_ton"] <= 200) & (data["Price_corn_ton"]<= 200) & (data["Price_wheat_ton"] <= 200) , "category"] = 1
data.loc[(data["Price_rice_ton"] <= 400) & (data["Price_corn_ton"]<= 400) & (data["Price_wheat_ton"] <= 400) &
         (data["Price_rice_ton"] > 200) & (data["Price_corn_ton"] > 200) & (data["Price_wheat_ton"] > 200), "category"] = 2
data.loc[(data["Price_rice_ton"] <= 1000) & (data["Price_wheat_ton"] <= 1000) &
         (data["Price_rice_ton"] > 400) & (data["Price_wheat_ton"] > 400), "category"] = 3
data.loc[data["category"] == 0 , "category"] = 4 

print(data["category"].value_counts())

x = data.drop("category", axis=1)
y = data["category"]

ss = StandardScaler()
x = ss.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle =True)
print(X_train.shape)



DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

print("_"*100)
print(DT.score(X_train, y_train))
print(DT.score(X_test, y_test))
print("_"*100)
y_pred = DT.predict(X_test)

# confusion_matrix
Cm = confusion_matrix(y_test,y_pred)
print(Cm)
sns.heatmap(Cm,annot=True, fmt="d", cmap="magma")
plt.show()

# The autput result
result = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# result.to_csv("The autput.csv",index=False)