#!/usr/bin/env python
# coding: utf-8

# <h1><font color='black'>
#     Question 1
#     </font></h1>
# <p></p>
# 
# 

# In[187]:


# Step 1

# import numpy as np #np is the standard convention for Numpy
import pandas as pd

# Step 2 & 3

users = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user",delimiter = '|')
users.head()


# In[188]:


# Step 4

print("Average Age Vs Occupation")
users[['occupation','age']].groupby('occupation').mean()
# users.groupby('occupation').age.mean()


# In[189]:


# Step 5

print("Male Ratio Vs Occupation In Descending Order %")

occupation_total = users.groupby(['occupation']).gender.count()
male_count = users.loc[users['gender'] == 'M']
male_in_occupation = male_count.groupby(['occupation', 'gender']).gender.count()
male_ratio = (male_in_occupation/occupation_total)*100
male_ratio.sort_values(ascending= False)


# In[190]:


# Step 6

print ("Occupation Vs. Minimum & Maximum Age")
#print(users.groupby('occupation').age.mean(['min', 'max']))

users.groupby('occupation').age.agg(['min', 'max'])
final = pd.DataFrame(users.groupby('occupation').age.agg(['min', 'max']))
final


# In[191]:


# Step 7

print("Mean Age Of Genders For All Occupations")

users.groupby(['occupation', 'gender']).age.mean()
final = pd.DataFrame(users.groupby(['occupation', 'gender']).age.mean())
final


# In[192]:


# Step 8 

occupation_gender = users.groupby(['occupation', 'gender']).gender.count()
count_occupation = users.groupby(['occupation']).gender.count()
final = round((occupation_gender/count_occupation)*100, 2).rename('%')
final = pd.DataFrame(final)
final


# <h1><font color='#749306'><font= "Courier New Bold">
#     Question 2
#     </font></h1>
# <p></p>

# In[193]:


# Step 1

import pandas as pd

# Step 2 & 3

euro12 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv", delimiter = ',')
euro12


# In[194]:


# Step 4

euro12[['Goals']]


# In[195]:


# Step 5

total= len(euro12.groupby('Team').groups)
print("Total Teams Participated In Euro 2022: ", total)

# Step 6

columns = euro12.shape[1]
print("Total Number Of Columns In Euro12 Dataset are: ", columns)


# In[196]:


# Step 7

print("Countries In Euro-12 With Total Red & Yellow Cards")

discipline = euro12[['Team','Yellow Cards','Red Cards']]
discipline


# In[197]:


# Step 8

discipline = discipline.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=False)
discipline


# In[198]:


# Step 9
 
yellow_cards_mean = euro12['Yellow Cards'].mean() 
print("Mean Yellow Cards Given Per Team: ", yellow_cards_mean)


# In[199]:


# Step 10

print("Teams Who Scored More Than 6 Goals")
euro12[euro12['Goals'] > 6]


# In[200]:


# Step 11

print("Teams That Start With 'G'")
euro12[euro12['Team'].str.startswith('G')] 


# In[201]:


# Step 12

print("First 7 Columns")
euro12.iloc[:, :7]


# In[202]:


# Step 13

print("All Columns Except Last 3")
euro12.iloc[:, :-3]


# In[203]:


# Step 14

print("Shooting Accuracy For Italy, England & Russia")
euro12.loc[euro12.Team.isin(['Italy', 'England', 'Russia'])][['Team', 'Shooting Accuracy']]


# <h1><font color='brown'><font= "Courier New Bold">
#     Question 3
#     </font></h1>
# <p></p>

# In[212]:


# Step 1

import numpy as np
import pandas as pd

# Step 2

ser1 = pd.Series(np.random.randint(1,5,(100)))
ser2 = pd.Series(np.random.randint(1,4,(100)))
ser3 = pd.Series(np.random.randint(10000,30000,(100)))

#Create DataFrame (df) by joinning the Series by column
#df = pd.DataFrame({'ser1':ser1,'ser2':ser2,'ser3':ser3})
#df = pd.DataFrame({ser1, ser2, ser3}, axis=1)

# Step 3

df = pd.DataFrame({'ser1':ser1,'ser2':ser2,'ser3':ser3})

# Step 4

df.rename(columns = {'ser1':'bedrs', 'ser2':'bathrs', 'ser3':'price_sqr_meter'}, inplace = True)

# Step 5
df2 = pd.DataFrame({'bigcolumn': pd.concat([ser1, ser2, ser3])})

# Step 6
print("Step 6: Yes It is Going Only Till 99 & Not 100")
df


# In[211]:


df2.index = pd.RangeIndex(start=0, stop=300)
df2


# <h1><font color='Yellow'><font= "Courier New Bold">
#     Question 4
#     </font></h1>
# <p></p>

# In[233]:


# Step 1

import pandas as pd    
import numpy as np
import datetime as dt

# Step 2 & 3
data = pd.read_csv('wind.txt', delimiter='\s+')
data.rename(columns={'Yr':'Year','Mo':"Month","Dy":"Day"}, inplace=True)
data


# In[234]:


# Step 4

data['Year'] += 1900
data


# In[238]:


# Step 5

data.index = pd.to_datetime(pd.concat([data['Year'], data['Month'], data['Day']], axis=1))
data.drop(['Year','Month','Day'], axis=1, inplace=True)
data


# In[243]:


# Step 6

data.isnull().sum()


# In[244]:


# Step 7

data.shape[0]-data.isnull().sum()


# In[249]:


# Total non null values

data.shape[0]-data.isnull().sum().sum()
print(" Total Non-Null Value Count: ", data.shape[0]-data.isnull().sum().sum())


# In[247]:


# Total null Values

data.isnull().sum().sum()
print(" Total Null Value Count: ", data.isnull().sum().sum())


# In[256]:


# Step 8 

a= round(data.mean().mean())
print("Total Mean For Windspeed Overall: ", a)


# In[258]:


# Step 9

loc_stats=pd.DataFrame()
loc_stats['min']=data.min()
loc_stats['max']=data.max()
loc_stats['mean']=data.mean()
loc_stats['std']=data.std()
loc_stats


# In[261]:


# Step 10

day_stats=pd.DataFrame()
day_stats['min']=data.min(axis=1)
day_stats['max']=data.max(axis=1)
day_stats['mean']=data.mean(axis=1)
day_stats['std']=data.std(axis=1)

day_stats


# In[262]:


# Step 11

data['date']=data.index
data['month']=data['date'].apply(lambda date:date.month)
data['year']=data['date'].apply(lambda date:date.year)
data['day']=data['date'].apply(lambda date:date.day)
january_winds=data.query('month==1')
january_winds
january_winds.loc[:,'RPT':'MAL'].mean()


# In[263]:


# Step 12

data.query('month == 1 and day == 1')


# In[264]:


# Step 13

data.query('day == 1')


# In[265]:


# Step 14

resample_week = data.resample('W').mean()
resample_week


# In[266]:


# Step 15

df_1961 = data[data.index < pd.to_datetime('1962-01-01')]
df_1961.resample('W').mean()
df_1961.resample('W').min()
df_1961.resample('W').max()
df_1961.resample('W').std()


# <h1><font color='voilet'><font= "Courier New Bold">
#     Question 5
#     </font></h1>
# <p></p>

# In[269]:


# Step 1, 2, 3  & 4

chipo=pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv',sep='\t')
chipo.head(10)


# In[270]:


# Step 5

print("Number of observation is:", len(chipo)) 


# In[271]:


# Step 6

print("The number of column in the dataset is:", len(chipo.columns))


# In[272]:


# Step 7

print("Column names: ", chipo.columns.values)   


# In[273]:


# Step 8

chipo.index 


# In[274]:


# Step 9

most_ordr_item = chipo.groupby(['item_name'])['item_name'].agg(['count']).sort_values(by='count', ascending=False).reset_index()
print("The most ordered item is:", most_ordr_item.loc[0][0])


# In[275]:


# Step 10

most_ordr_item.loc[0] 


# In[279]:


# Step 11

print("The Most Ordered Item In Choice_Desc: ", chipo.groupby('choice_description').agg({'quantity':'sum'}).sort_values(by='quantity', ascending=False).head(1).index[0])


# In[280]:


# Step 12

print("Total Items Ordered Were :", most_ordr_item['count'].sum()) 


# In[281]:


# Step 13

price_in_float = chipo.copy()
price_in_float['item_price']  = chipo['item_price'].str.replace("$","", regex=True).astype(float)   
print(price_in_float['item_price'])


# In[284]:


lambda_df = chipo.copy()
lambda_df['item_price'] = chipo['item_price'].apply(lambda x: float(x.replace("$",""))) 


# In[285]:


print(lambda_df['item_price'])


# In[286]:


# Step 14

print("Total Revenue : ", (lambda_df['item_price'] * lambda_df['quantity']).sum()) 


# In[288]:


# Step 15

print("Total Orders :", lambda_df['quantity'].sum()) 


# In[289]:


# Step 16

print("Average Revenue/ Order: ", (lambda_df['item_price'] * lambda_df['quantity']).mean()) 


# In[290]:


# Step 17

print("Different Items Sold : ", most_ordr_item.count()[0]) 


# <h1><font color='Orange'><font= "Courier New Bold">
#     Question 6
#     </font></h1>
# <p></p>

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\\Users\\romil\\Downloads\\usdata.csv", header=0, sep=",")
df


# In[14]:


df.plot(x='Year', y=['Marriages_per_1000', 'Divorces_per_1000'])


plt.title('Marriages and Divorces per capital in the U.S. between 1867 and 2011')
plt.ylabel('Marriages and Divorces per Capita')
plt.xlabel('Year')
plt.grid(axis='x')
plt.show()


# <h1><font color='green'><font= "Courier New Bold">
#     Question 7
#     </font></h1>
# <p></p>

# In[15]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=[15, 10])
marriage = [709000, 1667000, 2315000]
divorce = [56000, 385000, 944000]

X = np.arange(len(marriage))

plt.bar(X, marriage, color = 'black', width = 0.25)
plt.bar(X + 0.25, divorce, color = 'red', width = 0.25)
plt.legend(['Marriage', 'Divorce'])
plt.xticks([i + 0.25 for i in range(3)], ['1900', '1950', '2000'])
plt.title("Marriage Vs. Divorce Vertical Chart")
plt.xlabel('Marriages Vs. Divorce Per Capita In US For 1900, 1950, and 2000')
plt.ylabel('Total')
plt.show()


# <h1><font color='pink'><font= "Courier New Bold">
#     Question 8
#     </font></h1>
# <p></p>

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

kills = pd.read_csv('actor_kill_counts.csv')
kills


# In[21]:


kills.plot.barh(x='Actor', y='Count')
plt.ylabel('Actor')
plt.xlabel('Kill Count')
plt.grid(axis='x', linestyle = '--')
plt.show()


# <h1><font color='Black'><font= "Courier New Bold">
#     Question 9
#     </font></h1>
# <p></p>

# In[24]:


import matplotlib.pyplot as plt
import pandas as pd

roman_emperors = pd.read_csv('roman-emperor-reigns.csv')
assassinated_emperors = roman_emperors[
roman_emperors['Cause_of_Death'].apply(lambda x: 'assassinated' in x.lower())]

print(assassinated_emperors)
number_assassinated = len(assassinated_emperors)

print(number_assassinated)
other_deaths = len(roman_emperors) - number_assassinated

print(other_deaths)
emperor = assassinated_emperors["Emperor"]
cause_of_death = assassinated_emperors["Cause_of_Death"]
plt.pie(range(len(cause_of_death)), labels=emperor,autopct='%1.2f%%', startangle=50, radius=0.045 * 100,rotatelabels = 270)
fig = plt.figure(figsize=[13, 18])


# <h1><font color='Blue'><font= "Courier New Bold">
#     Question 10
#     </font></h1>
# <p></p>

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Relationship = pd.read_csv("arcade-revenue-vs-cs-doctorates.csv", header=0, delimiter=",")
print(Relationship)


# In[28]:


file_url = 'arcade-revenue-vs-cs-doctorates.csv'
df = pd.read_csv(file_url)
df.rename(columns = {'Total Arcade Revenue (billions)':'REVENUE','Computer Science Doctorates Awarded (US)':'AWARDS'}, inplace=True)
groups = df.groupby('Year')
for name, group in groups:
    plt.plot(group.REVENUE, group.AWARDS, marker='o', linestyle='', markersize=12, label=name)

plt.legend()
plt.xlabel("REVENUE")
plt.ylabel("AWARDS")
plt.show()


# In[ ]:




