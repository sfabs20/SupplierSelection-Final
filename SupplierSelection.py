#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import altair as alt
import plotly.graph_objects as go
import pydeck as pdk


if __name__ == "__main__":

    st.set_page_config(page_title="Supplier Evaluation and Selection Tool", page_icon="✅", layout="wide")
    dataset = pd.read_excel('C:\\Users\\shruti.a.nigam\\Desktop\\final_data.xlsx')
    st.title(":bar_chart: Supplier Evaluation and Selection Tool ")
    #---- Filter Section ----- #
    with st.sidebar:
        st.sidebar.header("Menu")
        filter_criteria1 = st.expander("Filter for Suppliers Summary")
        with filter_criteria1:
            parts1 = st.multiselect("Parts", options=dataset["part_name"].unique(),default=dataset["part_name"].unique())
            year = st.multiselect("Year", options=dataset["year"].unique(), default=dataset["year"].unique())
            #price = st.slider("Price", min_value=float(0), max_value=float(max(dataset["unit_price"])))
            #lead_time = st.slider("Lead_Time", min_value=0, max_value=max(dataset["lead_time_in_days"]))
        filter_criteria2 = st.expander("Filter For Suppliers ranking")
        with filter_criteria2:
            parts2 = st.selectbox("Parts", options=dataset["part_name"].unique())
datasummary,ranking,report = st.tabs(["Suppliers summary","Supplier Ranking","Best Suppliers for each part"])
with datasummary:
    st.caption("This displays distribution of related KPI data for all suppliers")
    st.markdown("Suppliers Location")
    df_latlong = dataset[['latitude','longitude']]
    st.map(df_latlong,zoom=4, use_container_width=True)
    st.markdown('supplier by unit price')
    #st.bar_chart(data=dataset[dataset['part_name'].isin(parts1) & dataset['year'].isin(year)],x='supplier_name', y='unit_price')
    bar1=alt.Chart(dataset[dataset['part_name'].isin(parts1) & dataset['year'].isin(year)]).mark_bar().encode(
         x='supplier_name:O',
         y='unit_price:Q',
         color='part_name:N',
         column='part_name:N')
    st.altair_chart(bar1,use_container_width=False)
    st.markdown('supplier by ESG score')
    #st.bar_chart(data=dataset,x='supplier_name', y='esg_score')
    esg_score = pd.DataFrame(dataset.groupby(['supplier_name','part_name']).mean('esg_score'))
    bar2=alt.Chart(dataset[dataset['part_name'].isin(parts1) & dataset['year'].isin(year)]).mark_bar().encode(
         x='supplier_name:O',
         y='esg_score:Q',
         color='part_name:N',
         column='part_name:N')
    st.altair_chart(bar2)
    st.markdown('supplier by Lead time(in days)')
    lead_time = pd.DataFrame(dataset.groupby(['supplier_name','part_name']).mean('lead_time_in_days'))
    #st.bar_chart(data=dataset,x='supplier_name', y='lead_time_in_days')
    bar3=alt.Chart(dataset[dataset['part_name'].isin(parts1) & dataset['year'].isin(year)]).mark_bar().encode(
         x='supplier_name:O',
         y='lead_time_in_days:Q',
         color='part_name:N',
         column='part_name:N')
    st.altair_chart(bar3)
    #st.bar_chart(data=dataset,x='supplier_name', y='lead_time_in_days')
    #st.map(data=dataset['location_id'])
    total_sales = pd.DataFrame(dataset.groupby(['supplier_name','part_category','year','tier_level']).sum('total_sales'))
    bar4=alt.Chart(dataset[dataset['part_name'].isin(parts1) & dataset['year'].isin(year)]).mark_bar().encode(
       x='supplier_name:O',
       y='total_sales:Q',
       color='part_category:N',
       column='part_category:N')
    st.altair_chart(bar4)
 
    #scatter = px.scatter(dataset,x='unit_price', y='total_sales', title="unit_price and total_sales are highly correlated")
    #st.plotly_chart(scatter)


#weighted product calculation

with ranking:
  st.caption("This tab displays ranking of suppliers for the selected car part")
  from objective_weights_mcda.mcda_methods import VIKOR
  from objective_weights_mcda.additions import rank_preferences
  from objective_weights_mcda import correlations as corrs
  from objective_weights_mcda import normalizations as norm_methods
  from objective_weights_mcda import weighting_methods as mcda_weights


# In[3]:


  import numpy as np
  import pandas as pd
  import copy
  import matplotlib.pyplot as plt
  import seaborn as sns


# In[4]:


  criteria_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\criteria.csv", index_col = 'Cj')


# In[5]:


  data_tyre_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\Tyre.csv", index_col = 'Ai')
  data_fuelfilter_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\Fuel_Filter.csv", index_col = 'Ai')
  data_headlight_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\Head_light.csv", index_col = 'Ai')
  data_airfilter_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\AirFilter.csv", index_col = 'Ai')
  data_airbag_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\Airbag.csv", index_col = 'Ai')
  data_clutch_presentation = pd.read_csv("C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\Clutch.csv", index_col = 'Ai')
# In[6]:


# Load data from CSV
  filename1 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Airbag.csv"
  data1 = pd.read_csv(filename1, index_col = 'Ai')
  filename2 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Airfilter.csv"
  data2 = pd.read_csv(filename2, index_col = 'Ai')
  filename3 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Clutch.csv"
  data3 = pd.read_csv(filename3, index_col = 'Ai')
  filename4 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Fuel_Filter.csv"
  data4 = pd.read_csv(filename4, index_col = 'Ai')
  filename5 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Head_Light.csv"
  data5 = pd.read_csv(filename5, index_col = 'Ai')
  filename6 = "C:\\Users\\shruti.a.nigam\\Documents\\Supplier Selection\\dataset_Tyre.csv"
  data6 = pd.read_csv(filename6, index_col = 'Ai')
# Load decision matrix from CSV
  df_data1 = data1.iloc[:len(data1) - 1, :]
  df_data2 = data2.iloc[:len(data2) - 1, :]
  df_data3 = data3.iloc[:len(data3) - 1, :]
  df_data4 = data4.iloc[:len(data4) - 1, :]
  df_data5 = data5.iloc[:len(data5) - 1, :]
  df_data6 = data6.iloc[:len(data6) - 1, :]
# Criteria types are in the last row of CSV
  types1 = data1.iloc[len(data1) - 1, :].to_numpy()
  types2 = data2.iloc[len(data2) - 1, :].to_numpy()
  types3 = data3.iloc[len(data3) - 1, :].to_numpy()
  types4 = data4.iloc[len(data4) - 1, :].to_numpy()
  types5 = data5.iloc[len(data5) - 1, :].to_numpy()
  types6 = data6.iloc[len(data6) - 1, :].to_numpy()
# Convert decision matrix from dataframe to numpy ndarray type for faster calculations.
  matrix1 = df_data1.to_numpy()
  matrix2 = df_data2.to_numpy()
  matrix3 = df_data3.to_numpy()
  matrix4 = df_data4.to_numpy()
  matrix5 = df_data5.to_numpy()
  matrix6 = df_data6.to_numpy()
# Symbols for alternatives Ai
  list_alt_names1 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data1.shape[0] + 1)]
  list_alt_names2 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data2.shape[0] + 1)]
  list_alt_names3 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data3.shape[0] + 1)]
  list_alt_names4 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data4.shape[0] + 1)]
  list_alt_names5 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data5.shape[0] + 1)]
  list_alt_names6 = [r'$A_{' + str(i) + '}$' for i in range(1, df_data6.shape[0] + 1)]
# Symbols for columns Cj
  cols1 = [r'$C_{' + str(j) + '}$' for j in range(1, data1.shape[1] + 1)]
  cols2 = [r'$C_{' + str(j) + '}$' for j in range(1, data2.shape[1] + 1)]
  cols3 = [r'$C_{' + str(j) + '}$' for j in range(1, data3.shape[1] + 1)]
  cols4 = [r'$C_{' + str(j) + '}$' for j in range(1, data4.shape[1] + 1)]
  cols5 = [r'$C_{' + str(j) + '}$' for j in range(1, data5.shape[1] + 1)]
  cols6 = [r'$C_{' + str(j) + '}$' for j in range(1, data6.shape[1] + 1)]


# In[7]:


  print('Criteria types')


# In[8]:


  weights1 = mcda_weights.entropy_weighting(matrix1, types1)
  weights2 = mcda_weights.entropy_weighting(matrix2, types2)
  weights3 = mcda_weights.entropy_weighting(matrix3, types3)
  weights4 = mcda_weights.entropy_weighting(matrix4, types4)
  weights5 = mcda_weights.entropy_weighting(matrix5, types5)
  weights6 = mcda_weights.entropy_weighting(matrix6, types6)

  df_weights1 = pd.DataFrame(weights1.reshape(1, -1), index = ['Weights'], columns = cols1)
  df_weights2 = pd.DataFrame(weights2.reshape(1, -1), index = ['Weights'], columns = cols2)
  df_weights3 = pd.DataFrame(weights3.reshape(1, -1), index = ['Weights'], columns = cols3)
  df_weights4 = pd.DataFrame(weights4.reshape(1, -1), index = ['Weights'], columns = cols4)
  df_weights5 = pd.DataFrame(weights5.reshape(1, -1), index = ['Weights'], columns = cols5)
  df_weights6 = pd.DataFrame(weights6.reshape(1, -1), index = ['Weights'], columns = cols6)
  #st.write(df_weights1*100)
  #st.write(df_weights2*100)
  #st.write(df_weights3*100)
  #st.write(df_weights4*100)
  #st.write(df_weights5*100)
  #st.write(df_weights6*100)
# Create the VIKOR method object
  vikor = VIKOR(normalization_method=norm_methods.minmax_normalization)

# Calculate alternatives preference function values with VIKOR method
  pref1 = vikor(matrix1, weights1, types1)
  pref2 = vikor(matrix2, weights2, types2)
  pref3 = vikor(matrix3, weights3, types3)
  pref4 = vikor(matrix4, weights4, types4)
  pref5 = vikor(matrix5, weights5, types5)
  pref6 = vikor(matrix6, weights6, types6)
# rank alternatives according to preference values
  rank1 = rank_preferences(pref1, reverse = False)
  rank2 = rank_preferences(pref2, reverse = False)
  rank3 = rank_preferences(pref3, reverse = False)
  rank4 = rank_preferences(pref4, reverse = False)
  rank5 = rank_preferences(pref5, reverse = False)
  rank6 = rank_preferences(pref6, reverse = False)
  @st.experimental_memo
  def get_data() -> pd.DataFrame:
      return pd.read_excel("Supplierschoice.xlsx",index_col = 'Alternatives')

  df = get_data()
  df_results1 = pd.DataFrame(index = list_alt_names1)
  df_results2 = pd.DataFrame(index = list_alt_names2)
  df_results3 = pd.DataFrame(index = list_alt_names3)
  df_results4 = pd.DataFrame(index = list_alt_names4)
  df_results5 = pd.DataFrame(index = list_alt_names5)
  df_results6 = pd.DataFrame(index = list_alt_names6)
  df_results1['Pref'] = pref1
  df_results1['Rank'] = rank1
  df_results1['Alternatives'] = ['A1','A2','A3','A4','A5']
  df_results2['Pref'] = pref2
  df_results2['Rank'] = rank2
  df_results2['Alternatives'] = ['A1','A2','A3','A4','A5']
  df_results3['Pref'] = pref3
  df_results3['Rank'] = rank3
  df_results3['Alternatives'] = ['A1','A2','A3','A4']
  df_results4['Pref'] = pref4
  df_results4['Rank'] = rank4
  df_results4['Alternatives'] = ['A1','A2','A3','A4','A5']
  df_results5['Pref'] = pref5
  df_results5['Rank'] = rank5
  df_results5['Alternatives'] = ['A1','A2','A3','A4','A5']
  df_results6['Pref'] = pref6
  df_results6['Rank'] = rank6
  df_results6['Alternatives'] = ['A1','A2','A3','A4','A5']
  df1=pd.merge(df_results1, df, on='Alternatives',how='inner') 
  df2=pd.merge(df_results2, df, on='Alternatives',how='inner') 
  df3=pd.merge(df_results3, df, on='Alternatives',how='left') 
  df4=pd.merge(df_results4, df, on='Alternatives',how='inner') 
  df5=pd.merge(df_results5, df, on='Alternatives',how='inner') 
  df6=pd.merge(df_results6, df, on='Alternatives',how='inner') 
  if "Airbag" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df1,x='Name', y='Rank',use_container_width=True)
     p2.markdown('Airbag Suppliers Ranking')
     p2.write(df1)
  elif "Air Filter" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df2,x='Name', y='Rank',use_container_width=True)
     p2.markdown('Airfilter Suppliers Ranking')
     p2.write(df2)
  elif "Clutch" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df3,x='Name', y='Rank',use_container_width=True) 
     p2.markdown('Clutch Suppliers Ranking')
     p2.write(df3)   
  elif "Fuel Filter" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df4,x='Name', y='Rank',use_container_width=True)
     p2.markdown('Fuel Filter Suppliers Ranking') 
     p2.write(df4)    
  elif "Head Light" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df5,x='Name', y='Rank',use_container_width=True)
     p2.markdown('Head Light Suppliers Ranking')
     p2.write(df5)
  elif "Tyre" in parts2:
     p1,p2 = st.columns(2)
     p1.bar_chart(data=df6,x='Name', y='Rank',use_container_width=True)
     p2.markdown('Tyre Suppliers Ranking')
     p2.write(df6)
with report:
    st.caption("This displays top 3 suppliers for each part")  
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.markdown('Top 3 Airbag Suppliers')
    minrank1 = df1.nsmallest(3, 'Rank')
    top1 = minrank1['Name']
    m1.write(top1)
    m2.markdown('Top 3 Air Filter Suppliers')
    minrank2 = df2.nsmallest(3, 'Rank')
    top2 = minrank1['Name']
    m2.write(top2)
    m3.markdown('Top 3 Clutch Suppliers')
    minrank3 = df3.nsmallest(3, 'Rank')
    top3 = minrank1['Name']
    m3.write(top3)
    m4.markdown('Top 3 Fuel Filter Suppliers')
    minrank4 = df4.nsmallest(3, 'Rank')
    top4 = minrank1['Name']
    m4.write(top4)
    m5.markdown('Top 3 Head Light Suppliers')
    minrank1 = df5.nsmallest(3, 'Rank')
    top5 = minrank1['Name']
    m5.write(top5)
    m6.markdown('Top 3 Tyre Suppliers')
    minrank1 = df6.nsmallest(3, 'Rank')
    top6 = minrank1['Name']
    m6.write(top6)
    
    
  
  



