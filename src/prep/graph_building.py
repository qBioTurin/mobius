import streamlit as st 
import pandas as pd
import zipfile, json
from time import time 
from itertools import combinations
import plotly.express as px 
from tools.utils import load_zip_archive, build_graph_zip, load_prepdata, write_graph_in_zip, write_prepdata, HelpMessage
from tools.gtools import OmicsGraphBuilder

from tools.enums import Coefficients


def load_preprocessed_data():
    match load_prepdata( st.session_state.zip_preprocessed_data ):
        case data, feature_sets, encodings:
            df = data.get("omics_data")
            covariates = feature_sets.pop("covariates")
            st.session_state[ "ettore" ] = ( df, feature_sets, covariates )
            st.session_state["colorz"] = encodings.get("colors")
        case None:
            del st.session_state[ "ettore" ]

    #         # Caricare i dati JSON
                

st.title(f"Network preparation")

HelpMessage.network_build()


st.file_uploader( 
    label = "Preprocessed data", 
    type = "zip", 
    key = "zip_preprocessed_data", 
    on_change = load_preprocessed_data
)


if st.session_state.zip_preprocessed_data and "ettore" in st.session_state: # 
    df, feature_sets, covariates = st.session_state[ "ettore" ]
    metadata = df[ covariates ]
    omics_names = [ name for name in feature_sets.keys() ] # if name != "metadata" ]
    available_pairs = list( combinations( omics_names, 2 ) )
    col_df, col_barplot = st.columns(2)
    with col_df:
        df_features = pd.DataFrame(data = [
            (omic_type, feature) for omic_type, feature_list in feature_sets.items() for feature in feature_list ], 
            columns = ["omic_type", "feature_id"]).set_index("feature_id")
        st.dataframe( df_features, use_container_width=True )
    with col_barplot:
        df_features["count"] = 1
        st.plotly_chart( use_container_width=True, figure_or_data = px.bar( 
            data_frame = df_features, 
            x = "omic_type", y = "count", 
            color="omic_type", 
            color_discrete_map = st.session_state.colorz) )
        

    omics_data = {
        name: df[ feature_sets[ name ] ] for name in omics_names
    }


    with st.container( border = True ):
        st.header("Stratified graph-building")

        with st.form( key = "form_corr", border=True ):
            
            st.selectbox(
                "Select correlation function", 
                key = "f_corr_id",
                options = map( lambda x: x.value, Coefficients ),  #Coefficients, #["Spearman", "Pearson", "Xi"], 
                index = 0 )


            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("Omics interaction")
                st.pills(
                    "Intra-omics interactions", 
                    selection_mode="multi",
                    options = omics_names, 
                    default = omics_names,
                    key = "intra_interactions" )
                st.pills(
                    "Inter-omics interactions",
                     selection_mode="multi",
                    options = available_pairs, 
                    default = available_pairs, 
                    key = "inner_interactions" )

            with col_r:
                st.subheader("Stratification settings")
                st.selectbox( 
                        "External stratification covariate", 
                        key="f_ext_strat", 
                        options = covariates, 
                        index = 0 )
                st.selectbox( 
                        "Internal stratification covariate", 
                        key="f_inn_strat", 
                        options = [None] + covariates, 
                        index = 0 )

                
            if st.form_submit_button("Start building process"):# and False:
                f_ext_strat = st.session_state.f_ext_strat
                f_inn_strat = st.session_state.f_inn_strat
                func_corr_name = st.session_state.f_corr_id 

                inter_corr = st.session_state.inner_interactions
                intra_corr = st.session_state.intra_interactions


                usr_message = (
                    f"single-strat graph w.r.t. {f_ext_strat}" if f_inn_strat is None else 
                    f"double-strat graph w.r.t. {f_ext_strat} and {f_inn_strat}"
                )
                
                graph_dict = dict() 

                with st.status( f"Computing {usr_message}" ) as status: 
                    if f_inn_strat is None:
                        start_t = time() 

                        gbuilder = OmicsGraphBuilder( omics_data, func_corr_name )
                        gbuilder.init_vertices()
                        st.write( f"Computing edges" )
                        gbuilder.build_edges( metadata, f_ext_strat, intra_corr, inter_corr )
                        st.write("Saving weighted edges")
                        gbuilder.create_edges()

                        end_t = time() 
                        st.write(f"Graph built in {end_t-start_t:.3f} seconds ")

                        graph_dict[ f_ext_strat ] = gbuilder.graph
                        
                    else:
                        group_sizes = [ (cov, subdf.shape[0]) for cov, subdf in metadata.groupby( f_ext_strat ) ]
                        group_sizes.sort( key = lambda t: t[1], reverse=True )

                        for cov_value, sample_size in group_sizes:
                            subdf = metadata[ metadata[f_ext_strat] == cov_value ].copy()
                            curr_omics = { name: omic_table.loc[ subdf.index ].copy() for name, omic_table in omics_data.items() }

                            st.write(f"Computing graph for {f_ext_strat}='{cov_value}' from {sample_size} samples")

                            start_t = time() 
            
                            gbuilder = OmicsGraphBuilder( curr_omics, func_corr_name )
                            gbuilder.init_vertices()
                            gbuilder.build_edges( subdf, f_inn_strat, intra_corr, inter_corr )
                            gbuilder.create_edges()

                            end_t = time() 
                            st.write(f"Graph built in {end_t-start_t:.3f} seconds")

                            graph_dict[ cov_value ] = gbuilder.graph 

                    status.update(label="Computation completed!", state="complete", expanded=False) 

                    initial_data = load_prepdata( st.session_state.zip_preprocessed_data )
                    encodings = dict(
                        stratification_info = dict(cov_external=f_ext_strat, cov_internal = f_inn_strat), 
                        **initial_data[2]
                    )
                    st.session_state[ "out_zip" ] = out_zip = write_prepdata(
                        tsv_data = initial_data[0], 
                        feature_sets = initial_data[1],
                        encodings = encodings
                    )
                    with zipfile.ZipFile( out_zip, "a" ) as zip_ml_data:
                        for name, g in graph_dict.items():
                            write_graph_in_zip( zip_ml_data, g, f"graph_{name}.gt" )
                            

if st.session_state.get("out_zip") is not None:
    with open( st.session_state[ "out_zip" ], "rb" ) as fp:
        st.download_button(
            "Download graph archive", 
            data = fp,
            file_name = st.session_state[ "out_zip" ],
            mime = "application/zip"
        ) 
    
   
