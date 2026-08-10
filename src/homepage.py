import streamlit as st 


st.info(
"""
    ### Welcome to MOBIUS

This is the homepage of the MOBIUS tool, where you can navigate through different sections for omics network analysis. 
This tool is designed to support multi-omics data integration and analysis using a combination of machine learning and complex network approaches.

Use the sidebar to access various functionalities, including data preparation, network analysis, and machine learning tasks.

It guides you step by step — from preparing your data, to analyzing networks, to selecting relevant features, and finally evaluating predictive models.

Below is a brief overview of the available modules and their purposes.

#### 🔧 Preparatory Steps

Before running any analysis, you need to set up your input data and tasks.

* 📂 Data Preparation: Import and structure your raw tabular data for further analysis.
* 🌐 Network Preparation: Build correlation networks from your tabular data stratifying samples by relevant metadata (e.g., disease status, cohort).
* 🏗️ Task Preparation: Define a binary classification tasks (e.g., case vs control) from your tabular data.


#### 🔗 Complex Network Analysis

These modules allow you to explore, manipulate, and compare networks.

* ⚜️ Load Zipped Networks: Import pre-packaged network files for quick analysis.
* 🧑‍🔬 Network Analysis: Perform structural and statistical analyses on networks.
* 👁️ Network Visualization: Interactively explore network structures.
* 🪓 Graph Pruning: Reduce network complexity by pruning edges below a given significance threshold.
* 🧩 Network Overlapping: Compare multiple networks to highlight shared and unique structures.

#### 🧠 Machine Learning Analysis

These modules focus on feature selection and predictive modeling.

* ✨ Load Zipped Task: Import a saved task (datasets + configurations).
* 🤌 Handle Features Sets: Organize and manipulate sets of candidate features.
* 📊 Exploratory Data Analysis (EDA): Get descriptive statistics, plots, and distributions to better understand your dataset.
* 🧮 Traditional Feature Selection: Apply standard statistical or ML-based methods for feature ranking.
* 👥 Ensemble Feature Selection: Combine multiple selection strategies for robust results.
* 🕸️ Network-based Feature Selection: Identify sets of highly relevant and low-redundant features by leveraging a multi-objective evolutionary algorithm running on the feature graph.
* 📈 Model Evaluation: Train and test predictive models, compare performance, and assess generalization.

""")




