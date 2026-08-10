import streamlit as st 

st.set_page_config(
    page_title="MOBIUS",
    page_icon=":dna",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={ 'About': "# This is an *extremely* cool app!" }
)

navigation_menu = {
    "Homepage": [ st.Page("homepage.py", title = "Homepage", icon="🏠", default=True) ],
    "Preparatory Steps": [
        st.Page("prep/dataprep.py", title = "Data Preparation", icon="📂"), 
        st.Page("prep/graph_building.py", title = "Network Preparation", icon="🌐") , 
        st.Page("ML/init_task.py", title = "Task Preparation", icon="🏗️"),
    ],
    "Network Analysis": [
        st.Page("analysis/load_net.py", title = "Load Zipped Networks", icon="⚜️") , 
        st.Page("analysis/analysis.py", title = "Network Analysis", icon="🕵️") , 
        st.Page("analysis/visualization.py", title = "Network Visualization", icon="👁️") , 
        st.Page("analysis/consistency_pruning.py", title = "Graph Pruning", icon="🪓"),
        st.Page("analysis/overlap.py", title = "Network Overlapping", icon="🧩")
    ],
    "Data-driven Analysis": [
        st.Page("ML/load_ml.py", title = "Load Zipped Task", icon="✨"), 
        st.Page("ML/features_handling.py", title = "Handle Features Sets", icon="🤌"),
        st.Page("ML/eda.py", title = "Exploratory Data Analysis", icon="📊"),
        st.Page("ML/fsel_traditional.py", title = "One-Shot Feature Selection", icon="🧮"),
        st.Page("ML/fsel_ensemble.py", title = "Ensemble Feature Selection", icon="👥"), 
        st.Page("ML/network_based_fs.py", title = "Network-based Feature Selection", icon="🕸️"), 
        st.Page("ML/evaluation.py", title = "Model Evaluation", icon="📈")
    ],
}


st.navigation({
    step: options 
        for step, options in navigation_menu.items() 
            if len(options) > 0
}).run()

