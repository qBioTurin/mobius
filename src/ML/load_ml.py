import streamlit as st 
from typing import List, Dict
import zipfile
import graph_tool.all as gt 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

from tools.utils import  load_prepdata, HelpMessage, read_graph_from_zip
from tools.gtools import new_numeric_vp
from tools.ml_utils import ML_SessionState, TaskParameters, init_session, WilcoxonTest
from tools.graph_fs import FeatureGraphEnricher
from tools.ml_managers import DataLoaderManager
import tools.mongoml as mongoml

from tools.enums import GraphField

import plotly.express as px 
import logging


def viz_feature_importances(df: pd.DataFrame, m2viz: List[str], color_palette: Dict[str, str] = None):
    def view_all_omics( df, metric ) -> go.Figure:
        fig = px.bar(
            df, 
            y = df.index, 
            x = metric, 
            color="omic", 
            color_discrete_map = color_palette, 
            category_orders = category_orders)

        
        mediana = df[metric].median()
        # Aggiunta della linea orizzontale come Scatter (per comparire nella legenda)
        fig.add_trace(
            go.Scatter(
                y=df.index,  
                x=[mediana] * (len(df)),  
                mode='lines',
                name=f"Median={mediana:e}",  
                line=dict(color='orange', dash='dash', width=2),  
            )
        )
        mean = df[metric].mean()
        fig.add_trace(
            go.Scatter(
                y=df.index,  
                x=[mean] * (len(df)),  
                mode='lines',
                name=f"Mean={mean:e}",  
                line=dict(color='green', dash='dash', width=2),  
            )
        )
        
        fig.update_yaxes( categoryorder = sorting_mode )

        fig.update_layout(
            title = f"Feature importances w.r.t. {metric}", 
            yaxis_title = "Feature", 
            xaxis_title = metric, 
            legend_title = "Node type"
        )
        return fig 


    my_options = ["Feature ordering", "Ascending ordering", "Descending ordering"]
    chosen_setting = st.segmented_control("Univariate feature importances", options = my_options, default=my_options[0] )  #index = 0, horizontal = True )
    category_orders= { "omic": sorted(color_palette.keys())} if color_palette else None  #{"omic": avail_omics} ,


    st.radio("Single or multi-omics view", options = ["Multi-omics", "Single-omics"], key = "featimp_viz_mode", horizontal = True )

    try:
        chosen_index = my_options.index( chosen_setting )
    except ValueError:
        chosen_index = 0
    finally:
        match chosen_index:
            case 1:
                sorting_mode = "total ascending"
            case 2: 
                sorting_mode = "total descending"
            case _:
                sorting_mode = None
        

    metric_tabs = st.tabs( m2viz )
    for tab, metric in zip( metric_tabs, m2viz ):
        with tab:
            

            if st.session_state.featimp_viz_mode == "Multi-omics":
                fig = view_all_omics( df, metric )
            else:
                omic_types = sorted( df.omic.unique().tolist() )
                fig = make_subplots(cols=1, rows=df.omic.nunique(), shared_xaxes=True, 
                    subplot_titles=omic_types, vertical_spacing=0.1)
                median = df[metric].median()
                mean = df[metric].mean()
                
                for i, omic_type in enumerate( omic_types ):
                    _subdf = df[ df.omic == omic_type ]

                    fig.add_trace(
                        go.Scatter(
                            y=_subdf.index,  
                            x=[median] * (len(_subdf)),  
                            mode='lines',
                            name=f"Median={median:.3f}",  
                            line=dict(color='orange', dash='dash', width=2 ),
                            showlegend= (i==0)  
                        ), col=1, row=i+1
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=_subdf.index,  
                            x=[mean] * (len(_subdf)),  
                            mode='lines',
                            name=f"Median={mean:.3f}",  
                            line=dict(color='black', dash='dash', width=2 ),
                            showlegend= (i==0)  
                        ), col=1, row=i+1
                    )

                    trace = go.Bar( 
                        y = _subdf.index, 
                        x = _subdf[metric].tolist(), 
                        name=omic_type, 
                        orientation='h',
                        marker=dict(color=color_palette[omic_type]) )
                    fig.add_trace( trace, col = 1, row = i + 1).update_yaxes(categoryorder=sorting_mode)
                    
                
                fig.update_layout(
                    title = f"View of {metric} centrality", 
                    yaxis_title = "feature", 
                    legend_title = "Omic layer"
                )

            st.plotly_chart( fig, use_container_width = True )


def validate_ML_input():
    st.session_state["valid_input"] = False
    st.session_state[ ML_SessionState.INIT_SESSION.value ] = False

    if ( input_zip := st.session_state.zip_ml_data ):

        data, features, encodings = load_prepdata( input_zip )
        training_data = data.get("training_set")
        st.session_state["task_info"] = encodings.get("task_info")
        st.session_state["palette"] = encodings.get("colors")
        covariates = features.pop("covariates")

        if st.session_state.get("task_info") is None:
            st.session_state["error_message"] = """
            The provided ZIP file is missing task information.

            Please, make sure you have loaded the zipfile produced by the 'Task Preparation' component.
            """
            return

        task_info = TaskParameters(
            st.session_state["task_info"]["target"], 
            st.session_state["task_info"]["pos_class"], 
            st.session_state["task_info"]["neg_class"]
        )
        data_id = task_info.get_unique_study_identifier( feature_groups = features )
        g_metadata = mongoml.retrieve_graph_metadata_from_mongo( data_id )

        if not g_metadata:# is None:
            st.session_state["error_message"] = f"""
            No feature graph with ID '{data_id}' has been found in the database. 

            Please, make sure you have built the graph first from the 'Task Preparation' component."""
        else:
            st.session_state[ "valid_input" ] = training_data[ covariates ]
            st.session_state[ "whole_g_shape"] = g_metadata  
            if st.session_state.get("error_message"):
                del st.session_state["error_message"]


def load_ml_UI():
    
    if st.session_state[ ML_SessionState.INIT_SESSION.value ]:
        with st.container(border=True):
            task = st.session_state.get( ML_SessionState.SELECTED_TASK.value )
            feature_graph = st.session_state.get( ML_SessionState.FEATURE_GRAPH.value )
            
            tr_set = st.session_state[ ML_SessionState.DISCOVERY_SET.value ]

            pos_class, neg_class = task.get_class(True), task.get_class(False)
            awesome_map = dict( true = pos_class, false = neg_class )
            vals = { v_data.name: v_data for v_data in st.session_state[ ML_SessionState.VALIDATION_SETS.value ] }
            awesome_data = { tr_set.name: tr_set, **vals }
            awesome_classprops = {
                data_id: { awesome_map[str(k).lower()]: v for k, v in d.get_class_distribution().to_dict().items() } 
                    for data_id, d in awesome_data.items() 
            }

            props = awesome_classprops[ tr_set.name ]
            no_pos, no_neg = props[ pos_class ], props[ neg_class ]
           

            st.markdown(f"""
#### Task: binary classification
* Target covariate: **{task.target_cov}** -> **{task.pos_class}** vs **{task.neg_class}**
* Feature graph: **{feature_graph.num_vertices()}** vertices and **{feature_graph.num_edges()}** edges""")
            cols = st.columns(2)
    
            with cols[0]:
                fc = Counter( list(feature_graph.vp[ GraphField.FEATURE_OMIC.value ] ))
                df_fc = pd.DataFrame( fc.items(), columns = ["omic", "num_features"] )#.set_index("omic")

                fig = px.bar( 
                    df_fc, 
                    x="omic", 
                    y = "num_features", 
                    color="omic", color_discrete_map=st.session_state["my_palette"], barmode="stack", 
                    title="Number of features per omic")
                st.plotly_chart( fig )

                st.dataframe( df_fc.set_index("omic"), use_container_width=True)
            
            with cols[1]:
                

                awesome_df = pd.DataFrame( awesome_classprops ).T
                fig = px.bar( awesome_df, x = awesome_df.index, y = [neg_class, pos_class], title=f"Number of samples per class" )
                fig.update_layout(xaxis={"title": "Dataset"},  yaxis={"title": "No. samples"}, legend={"title": f"Target: {task.target_cov}"})
                st.plotly_chart( fig )


                st.markdown(f"""#### Class proportions in **{tr_set.name}**\n
* **{task.pos_class}**: {no_pos} samples ({no_pos/(no_pos+no_neg)*100:.2f}%)
* **{task.neg_class}**: {no_neg} samples ({no_neg/(no_pos+no_neg)*100:.2f}%)""")


            ### TODO: put it in a function (and in a container) for clarity and efficiency 
            with st.container(border=True):
                all_data = st.session_state[ ML_SessionState.ALL_DATA.value ]
                data_names = [ dataset.name for dataset in all_data]
                ## TODO: add data shapes in the names
                data_tabs = st.tabs( data_names )

                for tab, curr_data in zip( data_tabs, all_data ):
                    with tab:
                        st.markdown( f"## {curr_data.name }")
                        st.markdown( "### Summary")
                        st.dataframe( curr_data.get_full_matrix().describe().T, use_container_width=True )


            with st.container(border=True): 
                st.header("Feature scores")
                df = st.session_state[ "df_weights" ]
                l, r = st.columns(2)
                with l:
                    st.markdown("#### Raw Feature importance Scores")
                    st.dataframe( df, use_container_width = True  )
                with r:
                    st.markdown("#### Summary")
                    st.dataframe( df.describe(), use_container_width = True  )
                with st.container(border=True):
                    viz_feature_importances(df, df.columns.tolist()[1:], st.session_state.my_palette)


    else:
        with st.container(border = True):
            st.header("Training and test data")
            st.file_uploader( label = "Task ZIP file", type = "zip", key = "zip_ml_data", on_change=validate_ML_input)
            

            if (error := st.session_state.get("error_message")):
                st.error(error)
                st.stop()

            if isinstance(st.session_state.get("valid_input"), pd.DataFrame): #metadata is not None and metadata is not False:
                metadata = st.session_state.get("valid_input")
                task_info = st.session_state.get("task_info")
                default_target_cov, default_pos_class, default_neg_class = [ task_info[k] for k in ("target", "pos_class", "neg_class") ]
                avail_columns = metadata.columns.tolist()
                index_default_opt = avail_columns.index( default_target_cov )
                
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.selectbox( "Target", key="target_cov", options = avail_columns, index = index_default_opt )
                    set_of_values = list( metadata[ st.session_state.target_cov ].unique() )
                    if st.session_state.target_cov != default_target_cov:
                        default_pos_class = list() 
                        default_neg_class = list()


                with col2:
                    st.multiselect( "Positive class",key="pos_class", options=set_of_values,default=default_pos_class)
                with col3:
                    
                    st.multiselect( "Negative class",key="neg_class", options=set_of_values,default=default_neg_class) 

                avail_covariates_for_learn = [ cov for cov in avail_columns if cov != st.session_state.target_cov]
                st.pills(
                    "Select covariates can be **optionally** used for training ML algorithms", 
                    options=avail_covariates_for_learn, 
                    default = avail_covariates_for_learn,
                    selection_mode="multi",
                    key="ML_covariates")

                
                options = dict()

                for rec in st.session_state.whole_g_shape:
                    if rec:
                        creation_date = rec.get("creation_date")
                        n_nodes = rec.get("n_nodes")
                        n_edges = rec.get("n_edges")
                        p = rec.get("p")

                        if p:
                            options[ p ] = (
                                p, creation_date, n_nodes, n_edges, str(rec.get("_id")),
                                f"Created on {creation_date}: {n_nodes} nodes, {n_edges} edges (p={p})"
                            )


                sorted_options = dict( sorted( options.items(), key=lambda item: item[1][0], reverse=True ) )  # sort by p-value
                opt =  [f"adj.p <= {p}" for p in sorted_options.keys() ]  #list( sorted_options.keys() )
                cap = [ v[-1] for v in sorted_options.values() ]


                st.radio("Load graph", options=opt, captions=cap, horizontal=True, key="id_graph_from_db")
                

                with st.form("form_data_loading", border = False):
                    
                    if st.form_submit_button("Finalize data loading"):
                        with st.status("Finalizing data loading...") as status:
                            
                            the_task = TaskParameters(st.session_state.target_cov, 
                                st.session_state.pos_class, 
                                st.session_state.neg_class )
                            
                            st.write(f"Task initialized: {the_task}")
                
                            
                            index = list( sorted_options.keys() )[opt.index( st.session_state.id_graph_from_db)]
                            graph_file_id = sorted_options[ index ][4]
                            st.write(f"Selected: {index} => file ID: {graph_file_id}")
                    
                            
                            load_manager = DataLoaderManager(
                                input_zipfile=st.session_state.zip_ml_data, 
                                task = the_task, 
                                user_covariates = st.session_state.ML_covariates, 
                                feature_graph_id = graph_file_id
                            )
                            st.write(f"Initializing the session...")
                            load_manager.initialize_session( st.session_state )
                            st.success("Data loaded successfully!")

                            
                            feature_graph = st.session_state[ ML_SessionState.FEATURE_GRAPH.value ]  
                            
                            st.session_state["my_palette"] = st.session_state["palette"]
                            
                            if FeatureGraphEnricher.get_vertex_weights( feature_graph, st.session_state.target_cov ):
                                df_weights_normalized = FeatureGraphEnricher.get_dataframe_vertex_weights( feature_graph, st.session_state.target_cov )
                                st.session_state[ "df_weights" ] = df_weights_normalized
                                st.session_state[ ML_SessionState.FEATURE_GRAPH.value] = FeatureGraphEnricher.update_graph_weights( feature_graph, df_weights_normalized)

                                dfff = WilcoxonTest( st.session_state[ ML_SessionState.DISCOVERY_SET.value ], the_task ).get_wilcoxon_results()
                                dfff = dfff.loc[ df_weights_normalized.index.tolist() ]

                                new_numeric_vp( feature_graph, GraphField.WX_PVALUE, "double", dfff.p_value.to_numpy() )
                                st.rerun()
                            else:
                                status.update(label="The feature graph has not vertex weights (aka feature importances) yet. Please, compute vertex weights before proceeding!", state="error", expanded=True) 


st.title("Load Task")

HelpMessage.load_zip_task()

init_session( st.session_state, ML_SessionState.INIT_SESSION, False )

load_ml_UI()