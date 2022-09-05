#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


# Creating first variable for dataset.
covid_df = pd.read_csv("C:/Users/User/Downloads/DATASET/covid_19_india.csv")


# In[3]:


# First 10 rows of data.
covid_df.head(10)


# In[4]:


# Getting information regarding a dataset.
covid_df.info()


# In[5]:


# For statistical analysis of dataset (only for numerical coloumns).
covid_df.describe()


# In[6]:


# Creating 2nd variable for vaccination dataset.
vaccine_df = pd.read_csv("C:/Users/User/Downloads/DATASET/covid_vaccine_statewise.csv")


# In[7]:


vaccine_df.head(7)


# In[8]:


# Dropping unnecessary coloumns from 1st dataset.
covid_df.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"], inplace = True, axis=1)


# In[9]:


covid_df.head()


# In[10]:


# Changing the format of date coloumn.
covid_df['Date'] = pd.to_datetime(covid_df['Date'], format = '%Y-%m-%d')


# In[11]:


covid_df.head()


# In[12]:


# Active cases.
covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# In[13]:


# Creating pivot table by adding confirmed deaths and cured coloumns.
statewise = pd.pivot_table(covid_df, values = ["Confirmed","Deaths","Cured"], 
                           index = "State/UnionTerritory", aggfunc = max)


# In[14]:


# Recovery rate =
statewise["Recovery Rate"] = statewise["Cured"]*100/statewise["Confirmed"]


# In[15]:


# Mortality rate
statewise["Mortality Rate"] = statewise["Deaths"]*100/statewise["Confirmed"]


# In[16]:


# Sorting values acc to confirmed cases
statewise = statewise.sort_values(by = "Confirmed", ascending = False)


# In[17]:


# Plot pivot table 
statewise.style.background_gradient(cmap = "magma")


# In[18]:


# Top 10 Active cases states
top_10_with_active_cases = covid_df.groupby(by = 'State/UnionTerritory').max()[['Active_Cases', 'Date']].sort_values(by = ['Active_Cases'], ascending = False).reset_index()
fig = plt.figure(figsize=(15,10))
plt.title("Top 10 states with Active Covid Cases in INDIA", size = 20)
ax = sns.barplot(data = top_10_with_active_cases.iloc[:10], y = "Active_Cases", x = "State/UnionTerritory", linewidth = 2, edgecolor = "pink")

plt.xlabel("States")
plt.ylabel("Total Active Cases")
plt.show()


# In[19]:


# Top 10 stated acc to DEATHS
top_10_deaths = covid_df.groupby(by = 'State/UnionTerritory').max()[['Deaths', 'Date']].sort_values(by = 'Deaths',ascending = False).reset_index()
fig = plt.figure(figsize = (18,5))
plt.title("Top 10 states with most Deaths", size = 25)
ax = sns.barplot(data = top_10_deaths.iloc[:12], y = "Deaths", x = "State/UnionTerritory", linewidth = 2, edgecolor = "blue")
plt.xlabel("States")
plt.ylabel("Total Death Cases")
plt.show()


# In[20]:


# Trends of active cases
fig = plt.figure(figsize = (12,6))
ax = sns.lineplot(data = covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra','Karnataka','Kerala','Tamil Nadu','Uttar Pradesh']), x = 'Date', y = 'Active_Cases', hue = 'State/UnionTerritory'])
ax.set_title("Top 5 Affected States in INDIA", size = 16)


# In[21]:


# 2nd Dataset
vaccine_df.head()


# In[23]:


# Renaming updates on to vaccine date
vaccine_df.rename(columns = {'Updated On' : 'Vaccine_Date'}, inplace = True)


# In[25]:


vaccine_df.head(10)


# In[26]:


vaccine_df.info()


# In[27]:


# Sum of missing values for each column
vaccine_df.isnull().sum()


# In[29]:


# Dropping missing value columns
vaccination = vaccine_df.drop(columns = ['Sputnik V (Doses Administered)','AEFI','18-44 Years (Doses Administered)','45-60 Years (Doses Administered)','60+ Years (Doses Administered)'], axis = 1)


# In[30]:


vaccination.head()


# In[32]:


# Pie chart for vaccination for male v/s female
male = vaccination["Male(Individuals Vaccinated)"].sum()
female = vaccination["Female(Individuals Vaccinated)"].sum()
px.pie(names=["Male","Female"], values=[male, female], title = "Male and Female Vaccination.")


# In[33]:


# Dropping rows with state mentioned as INDIA
vaccine = vaccine_df[vaccine_df.State!='India']
vaccine


# In[35]:


# Renaming last column
vaccine.rename(columns = {"Total Individuals Vaccinated" : "Total"}, inplace = True)
vaccine.head()


# In[36]:


# Most vaccinatted states
max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total', ascending = False)[:5]
max_vac 


# In[37]:


# Plot for most vaccinated states
fig = plt.figure(figsize = (10,5))
plt.title("Top 5 Vaccinated States in INDIA", size = 20)
x = sns.barplot(data = max_vac.iloc[:10], y = max_vac.Total, x =max_vac.index, linewidth = 2,edgecolor = 'Orange')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show()


# In[38]:


#  Least vaccinated states
min_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
min_vac = min_vac.sort_values('Total', ascending = True)[:5]
min_vac  


# In[43]:


# plot for least vaccinated states
fig = plt.figure(figsize = (15,10))
plt.title("Least 5 Vaccinated States in INDIA", size = 20)
x = sns.barplot(data = min_vac.iloc[:10], y = min_vac.Total, x = min_vac.index, linewidth = 2, edgecolor = 'Orange')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show() 


# In[44]:


vaccine_df.head()


# In[45]:


covid_df.head()


# In[ ]:




