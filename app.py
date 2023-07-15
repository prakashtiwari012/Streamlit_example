import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit_theme as stt

# Set page configuration
st.set_page_config(
    page_title="Wine Dataset Dashboard",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a predefined theme
#stt.set_theme({'primary': '#701b88'})

# Load the Wine dataset
wine = datasets.load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target

# Set dashboard title
st.title('Wine Dataset Dashboard')

# Sidebar selection
st.sidebar.title('Explore Wine Dataset')
selected_feature = st.sidebar.selectbox('Select a feature', wine.feature_names)
selected_target = st.sidebar.selectbox('Select a target', wine.target_names)

# Filter data based on selected feature and target
filtered_data = data[data['target'] == selected_target][[selected_feature, 'target']]

# Plotting
fig = px.scatter(data, x=selected_feature, y='target', color='target', width=800, height=500,
                 title=f'{selected_feature} vs Target')
fig.update_traces(marker=dict(size=8))
fig.update_layout(xaxis_title=selected_feature, yaxis_title='Target')
st.plotly_chart(fig)

# Show the dataset
st.subheader('Wine Dataset')
st.dataframe(data)

# Show the description of the selected feature
st.subheader('Description of Selected Feature')
st.write(wine.feature_names)
st.write(wine.DESCR.split('\n\n')[1])

# Show the description of the selected target
st.subheader('Description of Selected Target')
st.write(wine.target_names)
st.write(wine.DESCR.split('\n\n')[2])
