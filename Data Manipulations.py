#Importing pandas library
import pandas as pd
#Loading data into a DataFrame
data_frame=pd.read_csv('/FilePath/..../Mall_Customers.csv’)
data_frame.head()
list(data_frame.columns)
data_frame.info()
data_frame.describe()
data_frame.describe(include='object')
data_frame.shape
data_frame.isnull().sum()
data_frame.isna().sum()/len(data_frame)*100
#Removing 4th indexed value from the data_frame
data_frame.drop(4, inplace = True)
data_frame.head()
data_frame.drop(data_frame.index[[1,3]], inplace=True)
data_frame.drop('CustomerID',axis=1, inplace=True)
data_frame.head()
data_frame.drop(['Gender', 'Age'],axis=1, inplace=True)
data_frame.head()
#Importing pandas library
#Loading data into a DataFrame
data_frame=pd.read_csv('/content/Mall_Customers.csv')
data_frame.rename({'CustomerID':"ID", 'Gender':"Sex", 'Annual Income
(k$)':"Salary"}, axis=1, inplace=True)
data_frame.head()
data_frame['Sex'] = data_frame['Sex'].map({'Male': 1 , 'Female': 2})
data_frame.head()
data_frame=pd.read_csv('/content/Mall_Customers.csv')
data_frame.rename({'CustomerID':"ID", 'Gender':"Sex", 'Annual Income
(k$)':"Salary"}, axis=1, inplace=True)
data_frame.head()
data_frame['Sex'].unique()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
label_encoder = preprocessing.LabelEncoder()
data_frame['Sex']= label_encoder.fit_transform(data_frame['Sex'])
data_frame.head()
data_frame.rename({' Spending Score (1-100)':"Spending_Score"}, axis=1,
inplace=True)
data_frame.head()                   
low_spenders_data= data_frame[data_frame.Spending_Score < 70]
low_spenders_data.describe()
high_earners_low_spenders_data = data_frame[(data_frame.Salary > 59.76) &
(data_frame.Spending_Score < 70)]
high_earners_low_spenders_data.describe()
#Creates a new column with all the values equal to 1
data_frame['NewColumn'] = 1
data_frame.head()
def satisfaction(value):
if value > 70:
return "Satisfied"
else:
return "Unsatisfied"                   
data_frame['Spending_Score'].apply(satisfaction)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
data_frame['Customer_Satisfaction'] = data_frame['Spending_Score'].apply(satisfaction)
data_frame.head()
#This will save the dataset without the raws indeces
data_frame.to_csv(r'/content/Prepared_Mall_Customers.csv', index=False)
import plotly.express as px
#lets load your prepared dataset
data = pd.read_csv('/content/Prepared_Mall_Customers.csv')
# Construct the histogram plot for the Spending_Score histogarm
Spending_Score_fig = px.histogram(data, x='Spending_Score')
# Display the plot
Spending_Score_fig.show()
# Construct the histogram plot for the Spending_Score histogarm
Customer_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction')
# Display the plot
Customer_Satisfaction_fig.show()
Age_Salary_Association_fig = px.scatter(x=data['Spending_Score'],
y=data['Salary'])
Age_Salary_Association_fig.show()
Spending_Score_Satisfaction_fig = px.histogram(data, x='Age',
color='Customer_Satisfaction')
Spending_Score_Satisfaction_fig.show()
Spending_Score_Satisfaction_fig = px.histogram(data, x='Age',
color='Customer_Satisfaction', barmode='overlay')
Spending_Score_Satisfaction_fig.show()
Spending_Score_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction',
color='Salary')
Spending_Score_Satisfaction_fig.show()
data['Sex'] = data['Sex'].map({1:'Male', 0:'Female'})
Spending_Score_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction',
color='Sex')
Spending_Score_Satisfaction_fig.show()
data['Sex'] = data['Sex'].map({1:'Male', 0:'Female'})
Spending_Score_Satisfaction_fig = px.histogram(data, x='Age',
color='Customer_Satisfaction', barmode="group")
Spending_Score_Satisfaction_fig.show()

Age_Salary_Satisfaction_fig = px.scatter(data, x="Age", y="Salary",
color="Customer_Satisfaction")
Age_Salary_Satisfaction_fig.show()

#let’s load your prepared dataset
data = pd.read_csv('/content/Prepared_Mall_Customers.csv')
data.describe().transpose()
#We used transpose to make the columns rows and the rows columns to twist
the table
# Drop unnecessary variables and rename your dataset
df = data.drop(columns=(['ID', 'NewColumn']))
df.describe()
Salary_fig = px.histogram(df, x='Salary')
Salary_fig.show()
Age_fig = px.box(df, x='Age')
Age_fig.show()
Salary_fig = px.box(df, x='Salary')
Salary_fig.show()
Age_Salary_Scatter_fig = px.scatter(x=df['Age'], y=df['Salary'])
Age_Salary_Scatter_fig.show()
def find_outliers_IQR(df):
q1=df.quantile(0.25)
q3=df.quantile(0.75)
IQR=q3-q1
outliers = df[((df<(q1-1.5*IQR))|(df>(q3+1.5*IQR)))]
return outliers
outliers = find_outliers_IQR(df['Salary'])
print("number of outliers: "+ str(len(outliers)))
outliers
df.drop(df.index[[199,198]], inplace=True)

outliers = find_outliers_IQR(df['Salary'])
print("number of outliers: "+ str(len(outliers)))
outliers
df.describe().transpose()
df.isnull( )
#To find the percentage of missing data per variable
df.isna().sum()/len(data_frame)*100
df_Complete_Case = data_frame.dropna()
df_Complete_Case
Mean_Salary = df['Salary'].mean()
Mean_Spending_Score = df['Spending_Score'].mean()
Mean_Age = df['Age'].mean()
df['Salary'].fillna(Mean_Salary, inplace=True)
df['Spending_Score'].fillna(Mean_Spending_Score, inplace=True)
df['Age'].fillna(Mean_Age, inplace=True)
#To find the percentage of missing data per variable
df.isna().sum()/len(df)*100
#This will save the imputed dataset without the row index
df.to_csv(r'/content/Clean_Mall_Customers.csv', index=False)
import pandas as pd
#let’s load your prepared dataset
df1 = pd.read_csv('/content/Mall_Customers.csv')
df1.head()
import pandas as pd
#let’s load your prepared dataset
df2 = pd.read_csv('/content/Mall_Customers_Additional.csv')
df2.head()
Merged_Mall_df = df1.merge(df2, on='CustomerID')
Merged_Mall_df.head()
Merged_Mall_df.to_csv("Merged_Mall_Data.csv", index=False)
import pandas as pd
#let’s load your prepared dataset
df = pd.read_csv('/content/Merged_Mall_Data.csv')
df.describe()
from sklearn.preprocessing import StandardScaler
#drop unnecessary numeric and non-numeric variables
df_numeric = df.drop(columns=(['CustomerID', 'Gender']))
ss = StandardScaler()
df_scaled = ss.fit_transform(df_numeric)
df_scaled
df_scaled = pd.DataFrame(df_scaled,columns = df_numeric.columns)
df_scaled.head()
df_scaled.describe()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_mms = mms.fit_transform(df_numeric)
df_mms
df_mms = pd.DataFrame(df_mms,columns = df_numeric.columns)
df_mms.head()
df_mms.describe()









































                   
                   
