import pandas as pd
import numpy as np

df=pd.read_csv('/content/superstore.csv')

df.head()

df.shape

df.describe()

import seaborn as sns

import matplotlib.pyplot as plt

# binary cross entropy = -yilog(yi_hat)-(1-yi)log(1-yi_hat), LF=sigmoid, cross_e = (0.2*0.4*0.7*0.8), log(a*b)=loga+logb, log of vals bet. 0 to 1 = neg, -log(a*b)
# ax+by+c=0
# perceptron loss= max(0,-yi.f(xi)), sum(wi.xi)+b, step_func

# softmax_reg = loss = categ cross_entr, sparse_cat_cross_en = -sum: range - j = 1 to k (yilog(yi_hat)), softmax - multiclass_classific.

df.select_dtypes(include='object')

df.info()

df.isnull().sum()

df.drop(columns=['记录数'],inplace=True)

sns.boxplot(df['Year'])

sns.boxplot(df)

for i in df.select_dtypes(exclude='object').columns:
  plt.figure(figsize=(6,6))
  sns.boxplot(x=df[i])
  plt.title(f'boxplot of {i}')
  plt.plot()

for i in df.select_dtypes(include='object'):
  x=df[i].unique()
  print(f'{i}:{x}')



df['Product.ID'].nunique()

for i in df.select_dtypes(include='object'):
  x=df[i].nunique()
  if x < 100:
   print(f'{i}:{x}')

sns.distplot(df['Discount'])

sns.distplot(df['Quantity'])

df['Quantity'].skew(),df['Discount'].skew()

df[['Discount','Quantity']].describe()

# Discount :

perc75=df['Discount'].quantile(0.75)
perc25=df['Discount'].quantile(0.25)
perc75,perc25

iqr=perc75-perc25
iqr

upper_limit=perc75+1.5*iqr
lower_limit=perc25-1.5*iqr
upper_limit,lower_limit

df[df['Discount']>upper_limit]

(df['Discount']>upper_limit).mean()*100

df['Discount']=np.where(df['Discount']>upper_limit,
                        upper_limit,
                        np.where(df['Discount']<lower_limit,
                                 lower_limit,
                                 df['Discount']
                                 )
                        )

df['Discount'][47632]

# Quantity:

q3=df['Quantity'].quantile(0.75)
q1=df['Quantity'].quantile(0.25)
q3,q1

iqr1=q3-q1
iqr1

u_l=q3+1.5*iqr1
l_l=q1-1.5*iqr1
u_l,l_l

df[df['Quantity']>u_l]

(df['Quantity']>u_l).mean()*100

df['Quantity']=np.where(df['Quantity']>u_l,
                        u_l,
                        np.where(df['Quantity']<l_l,
                                 l_l,
                                 df['Quantity']
                                 )
                        )

df['Quantity'][51258]

for i in df[['Discount','Quantity']]:
  plt.figure(figsize=(6,6)),
  sns.boxplot(x=df[i])
  plt.plot()

df.select_dtypes(include='object').columns

df['Order.Year']=pd.to_datetime(df['Order.Date']).dt.year
df['Order.Month']=pd.to_datetime(df['Order.Date']).dt.month
df['Order.Day']=pd.to_datetime(df['Order.Date']).dt.day

df['Ship.Year']=pd.to_datetime(df['Ship.Date']).dt.year
df['Ship.Month']=pd.to_datetime(df['Ship.Date']).dt.month
df['Ship.Day']=pd.to_datetime(df['Ship.Date']).dt.day

df.drop(columns=['Order.Date','Ship.Date','Product.Name','Customer.Name'],inplace=True)

df1=df.copy()

x=df1.drop(columns=['Segment'])
y=df1['Segment']

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,chi2,f_classif

ohe=OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')
le=LabelEncoder()
te=TargetEncoder()
ss=StandardScaler()
ms=MinMaxScaler()
fe1=VarianceThreshold(threshold=0.03)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

'''Category:3
Market:7
Order.Priority:4
Region:13
Segment:3
Ship.Mode:4
Sub.Category:17
Market2:6'''

cat_cols_train=x_train[['Category','Market','Region','Sub.Category','Market2']]
cat_cols_test=x_test[['Category','Market','Region','Sub.Category','Market2']]


from sklearn import set_config

ohe1=ohe.set_output(transform='pandas')

train_ohe=ohe1.fit_transform(cat_cols_train)
test_ohe=ohe1.transform(cat_cols_test)

train_ohe1=ohe.fit_transform(cat_cols_train)
test_ohe1=ohe.transform(cat_cols_test)


train_ohe2=pd.DataFrame(train_ohe1,columns=ohe.get_feature_names_out(['Category','Market','Region','Sub.Category','Market2']))

test_ohe2=pd.DataFrame(test_ohe1,columns=ohe.get_feature_names_out(['Category','Market','Region','Sub.Category','Market2']))

train_ohe2.shape,test_ohe2.shape

x_train.select_dtypes(include='object').columns

x_train['Ship.Mode'].unique()

oe1=OrdinalEncoder(categories=[['Low','Medium','High','Critical']])
oe2=OrdinalEncoder(categories=[['First Class','Second Class','Standard Class','Same Day']])

x_train['Order.Priority']=oe1.fit_transform(x_train[[ 'Order.Priority']])


x_test['Order.Priority']=oe1.transform(x_test[[ 'Order.Priority']])

x_train['Ship.Mode']=oe2.fit_transform(x_train[[ 'Ship.Mode']])
x_test['Ship.Mode']=oe2.transform(x_test[[ 'Ship.Mode']])

df['City'].nunique(),df['Country'].nunique(),df['Product.ID'].nunique(),df['Customer.ID'].nunique(),df['Order.ID'].nunique()

y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

df2=df.copy()

df2['Segment']=le.fit_transform(df2[['Segment']])

df2.groupby('City')['Segment'].mean()*100









x_train['City']=te.fit_transform(x_train[['City']],y_train)
x_test['City']=te.transform(x_test[['City']])

x_train['Country']=te.fit_transform(x_train[['Country']],y_train)
x_test['Country']=te.transform(x_test[['Country']])
x_train['Product.ID']=te.fit_transform(x_train[['Product.ID']],y_train)
x_test['Product.ID']=te.transform(x_test[['Product.ID']])
x_train['Customer.ID']=te.fit_transform(x_train[['Customer.ID']],y_train)
x_test['Customer.ID']=te.transform(x_test[['Customer.ID']])
x_train['Order.ID']=te.fit_transform(x_train[['Order.ID']],y_train)
x_test['Order.ID']=te.transform(x_test[['Order.ID']])




x_train.drop(columns=x_train.select_dtypes(include='object')).shape,train_ohe2.shape

train1=pd.concat([x_train.drop(columns=x_train.select_dtypes(include='object')).reset_index(),train_ohe2.reset_index(),],axis=1).drop(columns=['index'])



test1=pd.concat([x_test.drop(columns=x_test.select_dtypes(include='object')).reset_index(),test_ohe2.reset_index()],axis=1).drop(columns=['index'])

train1.shape,test1.shape

train_x=ss.fit_transform(train1)

test_x=ss.transform(test1)

train_x.shape,test_x.shape

fe1_train=fe1.fit_transform(train_x)
fe1_test=fe1.transform(test_x)

fe1.get_support()

fe2=SelectKBest(f_classif,k=25)

an_train=fe2.fit_transform(train_x,y_train)
an_test=fe2.transform(test_x)

fe2.get_support()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf1=DecisionTreeClassifier(max_depth=14,random_state=12)

clf1.fit(train1,y_train)

y_pred=clf1.predict(test1)

from sklearn.metrics import accuracy_score, precision_score

accuracy_score(y_test,y_pred)

precision_score(y_test,y_pred,average='macro')

cvs=cross_val_score(clf1,train1,y_train,cv=3)

cvs.mean()

clf2=RandomForestClassifier(bootstrap=True,n_estimators=200,random_state=4)

clf2.fit(an_train,y_train)

y_pred1=clf2.predict(an_test)

accuracy_score(y_test,y_pred1)



cvs.mean()




















