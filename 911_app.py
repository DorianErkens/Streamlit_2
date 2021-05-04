import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib as plt 
import plotly.express as px 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import DBSCAN
import time

left_column,right_column = st.beta_columns(2)
left_columns = st.title('Clustering for 911 emergencies call')
right_column = st.image("https://www.aphp.fr/sites/default/files/don_dambulances.jpg")
st.write('We will take a look at the different categories of 911 calls')


#DATA_PATH ="/Users/dorian.erkens/Desktop/Jedha_FS_Bootcamp/Final Project/Streamlit/911_emergencies/911.csv"
DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/911.csv"

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_PATH,nrows=nrows)
    data = data.rename(columns={
        "lat":"Latitude",
        "lng":"Longitude",
        'desc':"Description",
        'title':'Title',
        'twp':'Township',
        'addr':"Address"})
    data = data.drop('e',axis=1)
    return data
data = load_data(10000)

st.header("Exploratory Data Analysis")
if st.checkbox('Tick the box to display the first lines of the raw dataset'):
    st.subheader('Raw data')
    st.dataframe(data=data.head())

st.subheader("We can see how this all fits on the map")
#fig = plt.figure(figsize=(30,10))
fig = px.scatter_mapbox(data_frame=data, lat="Latitude", lon="Longitude",color="Title",mapbox_style="open-street-map",color_continuous_scale=px.colors.cyclical.IceFire, size_max=25)
st.plotly_chart(fig, use_container_width=True)

st.header('Unsupervised Machine Learning')
#Regarder comment optimiser le cache sur le focused dataset, le load prend pas mal de temps sur ce nouveau dataset
#Il semble que le cache ne fait pas effet
@st.cache
def focused_dataset(nrows):
    data_focused = data[['Latitude','Longitude','Title']]
    data_focused = pd.get_dummies(data_focused,drop_first=True)
    return data_focused
data_focused = focused_dataset(data)
st.write('We need to take a different look at the current dataset')
st.write("Namely, we will isolate geographical and title information, from which we will get dummy variables")
st.write(data_focused)

st.subheader("DB Scan Model")
st.write('We will now use the DB Scan algorithm to see whether there is a pattern coming from the different type\nof 911 calls within the city')
X = data_focused
#left_column,right_column = st.beta_columns(2)
#button = right_column.button('You can push the button to launch the training of the model')
#latest_iteration = st.empty()
#bar = st.progress(0)
#if button : 
#    'Starting the fit ...'
#    for i in range(100):
#        latest_iteration.text(f'Iteration{i+1}')
#        clustering = DBSCAN(eps=0.1,min_samples=100,metric='manhattan')
#        clusters = clustering.fit_predict(X)
#        bar.progress(i+1)
#        time.sleep(0.1)
#    '... and it is complete'"""
left_column,right_column = st.beta_columns(2)
button = right_column.button('You can push the button to launch the training of the model')
if button : 
    clustering = DBSCAN(eps=0.1,min_samples=100,metric='manhattan')
    clusters = clustering.fit_predict(X)
    st.write('Well done, fit is complete and you ran your first Unsupervised model !')

    clusters_reshp = []
    for element in clusters.flat:
        clusters_reshp.append(element)
    clusters_reshp = pd.Series(data=clusters_reshp).rename('clusters',axis=1)


    st.write('After fitting, we are now able to see interesting patterns in the geomap')
    data = data.join(clusters_reshp)
    fig = px.scatter_mapbox(data[data.clusters!=-1], lat="Latitude", lon="Longitude",color="Title",mapbox_style="open-street-map",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=25)
    st.plotly_chart(fig, use_container_width=True)
