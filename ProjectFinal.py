import numpy as np 
import pandas as pd
import seaborn as sns 
import xlrd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder



df = pd.read_excel("prjct.xlsx")
print('Initial Table')
print(df)



x = df.groupby('bpref_no')
list = np.array(x['bpref_no'].unique())
df['monthly_consumption'] = df['base_consumption']//df['no_month_consumption']
df['month']=df['bill_month'].dt.month
df.drop('avg_ind', axis=1, inplace=True)
df



new_data = df[['meter_no','month','base_consumption','monthly_consumption','current_doc_amt']]
new_data['meter_no'].astype('category', copy=False)
new_data['month'].astype('category', copy=False)
new_data['current_doc_amt'].astype('int', copy=False)
print("Revised Table")
print(new_data)



plt.xlabel("Month")
plt.ylabel("Base Consumption")
plt.scatter(new_data['month'] , new_data['base_consumption'], color = 'green')
plt.show()



sns.swarmplot(x=new_data['base_consumption'], y=new_data['current_doc_amt'])
plt.show()


X = new_data[['meter_no', 'month']]
X = X.apply(LabelEncoder().fit_transform)
y = new_data[['base_consumption', 'current_doc_amt']] 



X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.1,shuffle = False)
lr = LinearRegression()
X_train.shape
test_base_consumption = y_test['base_consumption']
test_doc_amt = y_test['current_doc_amt']




lr.fit(X_train, y_train)
lr.intercept_
coeffcients = pd.DataFrame([X_train.columns,lr.coef_[0]]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})


y_pred = lr.predict(X_test)
print('Test values')
print(y_test)
print(y_pred)


plt.xlabel("Month")
plt.ylabel("Base Consumption")
plt.title("Prediction: month vs Base Consumption")
plt.scatter(X_test['month'] , y_test['base_consumption'], color = 'red')
plt.plot(X_test['month'] , y_pred[:,1:], color ='blue')
plt.show()

month = input("Enter a month")
meter_no = input("Enter a meter number")



month_list=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for i in range(len(month_list)):
    if month == month_list[i]:
        num = i
monthDF = pd.DataFrame({'meter_no':meter_no,'month':num}, index = [x for x in range(1)]) 
print(monthDF)
y_user_pred = lr.predict(monthDF)
print('The base_consumption would be: {}'.format( y_user_pred[:,0:]))


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print("Absolute error: {}".format(mean_absolute_error(y_test, y_pred)))
print('root mean squared error: {}'.format(np.sqrt(mse)))
print('r squared error: {}'.format(r_squared))


plt.plot(y_test, y_pred)
plt.xlabel("base_consumption")
plt.ylabel("Predicted Base Consumption")
plt.title("Predicted vs Test Consumption")
plt.show()





