import streamlit as st 
import pandas as pd 

from typing import Dict

from tools.gtools import GraphCoherenceCheck, get_input_graph_names
from tools.utils import build_tsv_archive, load_prepdata, write_prepdata, HelpMessage
from tools.enums import GraphField


ARE_DATA_OK = "data_are_ok"


@st.cache_resource
def get_vertex_weights( _gcc: GraphCoherenceCheck ) -> Dict[ str, pd.DataFrame ]:
    dfs_vweights = dict() 
    diff_set = { x.value for x in ( GraphField.FEATURE_NAME, GraphField.FEATURE_OMIC )}

    for g, condition in _gcc.get_graph_iterator():
        weight_set = set( g.vp.keys() ).difference( diff_set ) #['feature', 'omic'] )
        df = dfs_vweights[ condition ] = pd.DataFrame( index = list(g.vp[ GraphField.FEATURE_NAME.value ]))
        
        for wname in weight_set:
            df[wname] = g.vp[ wname ].a

    return dfs_vweights


@st.cache_resource
def coherence_check( g_archive, labels, sign_lb, padj_flag ):
    gcc = GraphCoherenceCheck( g_archive, labels, sign_lb, padj_flag )
    gcc.compute_local_bitmasks()
    big_df = gcc.extract_global_datatable()
    df_dict = gcc.extract_cohort_dependent_datatable()

    tsv_archive = build_tsv_archive( unstratified = big_df, **df_dict )
    graph_archive = gcc.save_filtered_graphs( zip_filename=None) ## generate temp file   #( f"filtered_graphs__p_{sign_lb}.zip" )
    vw_set = get_vertex_weights( gcc )

    return ( tsv_archive, graph_archive, vw_set )


def check_data_are_ok():
    if ARE_DATA_OK in st.session_state:
        del st.session_state[ ARE_DATA_OK ] 

    if "input_graph_archive" in st.session_state:
        input_graph_archive = st.session_state.input_graph_archive

        match load_prepdata( input_graph_archive ):
            case data, feature_sets, encodings:
                covariates = feature_sets.get("covariates")
                metadata = data.get("omics_data")[ covariates ]
                stratification_covariates = encodings.get("stratification_info")
                cov_inner = stratification_covariates.get("cov_internal")
                network_set = list( get_input_graph_names( input_graph_archive ) )

                if cov_inner is not None: 
                    label_alphabet = tuple( sorted( metadata[ cov_inner ].unique() )  )
                    st.session_state[ ARE_DATA_OK ] = (
                        stratification_covariates.get("cov_internal"), 
                        label_alphabet,
                        network_set
                    )
                else:
                    st.toast("Stratification covariate not found in metadata")

                print(f"Covariates: {covariates}")
                print(f"Metadata df:\n{metadata}\n")
                print(f"Stratification covariates: {stratification_covariates}")
            case _:
                pass 


st.title("Consistency & Significance Pruning")


HelpMessage.prune_nets()

st.file_uploader( 
    label = "Graph archive", 
    type = "zip", 
    key = "input_graph_archive", 
    on_change = check_data_are_ok
)


if st.session_state.get( ARE_DATA_OK ):
    strat_covariate, label_alphabet, network_set = st.session_state[ ARE_DATA_OK ]


    with st.form("GraphPruning"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(
f"""##### Dimensionality reduction:
* Stratification feature: **{strat_covariate}**
* Number of networks: {len(network_set)}
* List of networks:""") 
            st.write( network_set )
            

        with cols[1]:
            st.selectbox(f"Choose significance level", key="significance_level", options=[1, 0.05, 0.01, 0.001])
            use_padj = st.toggle("Use adjusted p-values", value=True, key="enable_padj")
            
        if st.form_submit_button("Reduce graph dimensionality"):
            graph_archive = st.session_state.input_graph_archive
            significance_lvl = st.session_state.significance_level

            sign_level_str = "NO" if significance_lvl == 1 else significance_lvl

            with st.status(f"Extracting data from graphs (significance lvl = {sign_level_str})...") as status:
                st.write(f"Computing coherence check with significance level = {significance_lvl}")
                tsv_zip, graph_zip, vw_set = coherence_check( graph_archive, label_alphabet, significance_lvl, use_padj )
                
                st.write("Saving data...")
                initial_data = load_prepdata( st.session_state.input_graph_archive )
                write_prepdata(
                    tsv_data=initial_data[0], 
                    feature_sets=initial_data[1],
                    encodings=initial_data[2],
                    zip_handle_append=graph_zip
                )
                
                st.session_state[ "zipped_files" ] = ( tsv_zip, graph_zip )
                
                ###TODO: salvare metadatai in graph_zip 

                status.update(label="Computation completed!", state="complete", expanded=False)


    if "zipped_files" in st.session_state: #False:
        significance_lvl = st.session_state.significance_level
        cached_data = st.session_state.zipped_files

        source_filename = st.session_state.input_graph_archive

        if all( cached_data ):
            with st.expander("Download data", expanded = False ):
                filename_networks = source_filename.name.split(".")[0] #st.session_state.graph_archive.name.split(".")[0]
                cached_tabs, cached_graph = cached_data 
                to_be_downloaded = [
                    ( cached_tabs, "Download zipped tables", f"tables_{filename_networks}_p_{significance_lvl}.zip"), 
                    ( cached_graph, "Download filtered graphs", f"{filename_networks}__p_{significance_lvl}.zip" )
                ]

                for col, (my_stuff, btn_title, filename_out) in zip( st.columns(2), to_be_downloaded):
                    with col:
                        with open( my_stuff, "rb" ) as fp:
                            st.download_button( btn_title, data = fp, file_name = filename_out, mime = "application/zip" )
