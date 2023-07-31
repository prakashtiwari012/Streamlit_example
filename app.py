import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit_theme as stt
import streamlit_authenticator as stauth
import streamlit as st
import yaml
from yaml.loader import SafeLoader
from PIL import Image


with open('auth_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

#hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def create_network_graph():
    # Create a simple example graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 1)])

    # Draw the graph using matplotlib
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, font_size=10, font_weight='bold', edge_color='gray')

    # Show the graph
    plt.title("Sample Network Graph")
    plt.tight_layout()
    st.pyplot()

@st.cache(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(image_name)




# Set page configuration
st.set_page_config(
    page_title="Wine Dataset Dashboard",
    page_icon="🍷",
    #page_icon="🚩",
    layout="wide",
    initial_sidebar_state="expanded",
)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    
    # Apply a predefined theme
    #stt.set_theme({'primary': '#701b88'})

    # Load the Wine dataset
    wine = datasets.load_wine()
    data = pd.DataFrame(wine.data, columns=wine.feature_names)
    data['target'] = wine.target

    # Set dashboard title
    st.title('Wine Dataset Dashboard')

    st.sidebar.image(load_image("capture.png"), use_column_width=True)

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

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
