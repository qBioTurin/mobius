from itertools import chain
import streamlit as st 


import pandas as pd 
import numpy as np 
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from collections import defaultdict

import logging
logging.basicConfig( level=logging.INFO )

from tools.utils import (
    SamplesMetadata, 
    nearZeroVar,
    write_prepdata, 
    generate_colors, 
    tab_files_extensions)
from tools.helpers import HelpMessage


SAMPLES_METADATA = "samples_metadata"
SAMPLES_LIST = "samples_list"
SAMPLES_OMICSDATA = "samples_omicsdata"
DATA_READY_TO_BE_DOWNLOADED = "your_awesome_data"


#####################################################################################
########################## UTILITY FUNCTIONS 
#####################################################################################
def filter_clinical_dataframe( df: pd.DataFrame ) -> pd.DataFrame:
    """ Adds a UI on top of a dataframe to let viewers filter columns """
    
    def categorical_filtering( df, column ) -> pd.DataFrame:
        val_columns = [str(x) for x in df[column].unique().tolist()]
        user_cat_input = right.multiselect(
            f"Values for {column}", val_columns, default=val_columns )

        return df[df[column].isin(user_cat_input)]


    modify = st.checkbox("Add filters" )

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                df = categorical_filtering( df, column )
                
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]

            else:
                df = categorical_filtering( df, column )

    return df

@st.fragment
def describe_metadata():
    numeric_only = st.checkbox("Numeric only")
    flag_numeric = None if numeric_only else "all"
    st.dataframe(  st.session_state[ SAMPLES_METADATA ].filt_data.describe( include = flag_numeric ) ) # include = "all" ) )


@st.cache_data
def load_dataframe( filename ) -> pd.DataFrame:
    if filename:
        ext = filename.name.split(".")[-1].lower()

        match ext:
            case "tsv" | "txt" | "csv":
                sep = "," if ext == "csv" else "\t"
                df = pd.read_csv( filename, sep = sep, index_col = 0, header = 0 )
            case "xlsx":
                df = pd.read_excel( filename, index_col = 0, header = 0 )

        for col in df.columns:
            if is_object_dtype( df[col] ):
                df[col] = df[col].astype( pd.StringDtype("pyarrow") )

        return df 


def load_clinicals():
    already_pres = SAMPLES_METADATA in st.session_state

    match ( clinical_covariates_filename := st.session_state.metadata_file ):
        case None:
            if already_pres:
                del st.session_state[ SAMPLES_METADATA ]
                logging.warning(f"Dataset deleted.")
                st.toast("Metadata deleted")

        case _ if not already_pres:
            df = load_dataframe( clinical_covariates_filename )
            st.session_state[ SAMPLES_METADATA ] = SamplesMetadata( df, df, df.columns.tolist() )
            st.session_state[ SAMPLES_LIST ] = df.index.tolist()
            st.toast(f"Metadata loaded: {df.shape}")


def load_omics_files():
    samples_set = set( st.session_state[ SAMPLES_METADATA ].filt_data.index.tolist() )
    omics_features = dict() 
    missing_samples = defaultdict(set)
    union_missing_samples = set() 
    df_list = list() 
    num_features = 0 
    dropped_features = dict()

    if ( omics_filelist := st.session_state.omics_files ):
        for omics_file in omics_filelist:
            curr_filename = omics_file.name

            if samples_set.intersection( (df := load_dataframe( omics_file )).columns.tolist() ):
                df = df.T

            if ( curr_missing_samples := samples_set.difference( df.index.tolist() ) ):
                missing_samples[ curr_filename ] = curr_missing_samples
                union_missing_samples = union_missing_samples.union( curr_missing_samples )

            num_features += df.shape[1]
            dropped_features[ curr_filename ] = curr_nzv = set( nearZeroVar( df ) )
            logging.critical(f"Dropping {len(curr_nzv)} out of {df.shape[1]} NearZeroVariance features from {curr_filename}")

            not_nzv_features = omics_features[ curr_filename ] = [c for c in df.columns.tolist() if c not in curr_nzv ]

            if len(not_nzv_features) != df.shape[1]:
                df = df[ not_nzv_features ]

            df_list.append( df )

        samples_with_omics = list( samples_set.difference( union_missing_samples ) ) 
        df_omics = pd.concat( df_list, axis = 1 ).loc[ samples_with_omics ]
        num_not_nzv_features = sum([ len(v) for v in omics_features.values() ])

        logging.critical(f"Features passing the NZV step: {num_not_nzv_features} out of {num_features}")

        st.session_state[ SAMPLES_OMICSDATA ] = ( 
            df_omics,
            omics_features, 
            missing_samples, 
            dropped_features )
        

st.title("Data preparation")

HelpMessage.data_preparation()

main_col_left, main_col_right = st.columns(2)
with main_col_left:
    st.header("Metadata and covariates")
    st.file_uploader( 
        label = "Tabular file containing samples' clinical data", 
        type = tab_files_extensions, 
        key = "metadata_file", 
        on_change = load_clinicals
    )

with main_col_right:
    st.header("Omics data")

    st.file_uploader( 
        label = "Tabular file containing samples' omics data", 
        type = tab_files_extensions, 
        accept_multiple_files = True,
        key = "omics_files",
        on_change = load_omics_files, 
        disabled = SAMPLES_METADATA not in st.session_state
    )


if SAMPLES_METADATA in st.session_state:

    with st.container(border=True):
        main_tab_metadata, main_tab_omics = st.tabs(["Metadata", "Omics data"])

        with main_tab_metadata:
            if SAMPLES_METADATA in st.session_state:
                metadata = st.session_state[ SAMPLES_METADATA ]

                col_data, col_edit = st.columns(2)

                with col_edit:
                    st.header("Handle metadata")
                    with st.container():
                        with st.form("change_index_form"):
                            avail_index_cols = [ 
                                col for col in metadata.filt_data.columns 
                                    if metadata.filt_data[col].value_counts().max() == 1 and metadata.filt_data[col].dtype in ("object", "string")
                            ]

                            st.pills(
                                "Select a column **without duplicated** to use as an index",
                                options = avail_index_cols,
                                key = "index_column", selection_mode="single"
                            )

                            if st.form_submit_button("Change index"):
                                new_index = st.session_state.index_column
                                if new_index:
                                    metadata.filt_data = metadata.filt_data.reset_index().set_index( new_index )
                                    metadata.raw_data = metadata.raw_data.reset_index().set_index( new_index )
                                    st.session_state[ SAMPLES_METADATA ] = SamplesMetadata(
                                        metadata.raw_data, metadata.filt_data, metadata.filt_data.columns.tolist() #metadata.covariates
                                    )
                                    logging.info(f"Index changed to {new_index}")
                                    st.rerun()
                                else:
                                    st.error("You cannot set an empty index")


                        with st.form("drop_columns_form"):
                            st.radio("Select the operation:", ["remove", "keep"], 
                                    captions = [ "Remove specific covariates", "Keep specific covariates"], 
                                    horizontal = True,
                                    key="mode_checkbox")
                            st.multiselect(
                                "Select covariates",
                                options =  st.session_state[ SAMPLES_METADATA ].covariates,
                                default = [],
                                key = "selected_clinicals"
                            )

                            if st.form_submit_button("Reduce dataframe"):
                                st.write(f"Covariates removed: {st.session_state.selected_clinicals}")

                                keep_mode = st.session_state.mode_checkbox == "keep" 
                                selected_ones = st.session_state.selected_clinicals

                                if keep_mode and selected_ones is False:
                                    st.error("You cannot.")
                                    
                                else:
                                    my_data = st.session_state[ SAMPLES_METADATA ]
                                    raw_data = my_data.filt_data
                                    old_shape = raw_data.shape 

                                    features_to_drop = set( my_data.covariates ).difference( selected_ones ) if keep_mode else selected_ones
                                    reduced_df = raw_data.drop( columns = list( features_to_drop ) )

                                    st.session_state[ SAMPLES_METADATA ] = SamplesMetadata(
                                        my_data.raw_data, reduced_df, reduced_df.columns.tolist()
                                    )

                                    logging.info(f"Dataframe modified from {old_shape} to {reduced_df.shape}")
                                    st.rerun()

                    with st.container(border =True):
                        fltrdf = filter_clinical_dataframe(  st.session_state[ SAMPLES_METADATA ].filt_data  )
                        enabled_filtering = st.session_state.get("filter_checkbox", False)
                        if st.button( "Confirm filtering", key = "btn_filter" ):
                            _old = st.session_state[ SAMPLES_METADATA ]
                            st.session_state[ SAMPLES_METADATA ] = SamplesMetadata( 
                                _old.raw_data, fltrdf, _old.covariates
                            )
            #                 
                with col_data:
                    st.header("View metadata")
                    with st.container(border=True):
                        tab_data, tab_describe = st.tabs(["Raw data", "Data summary"])
                        with tab_data:
                            st.dataframe(  st.session_state[ SAMPLES_METADATA ].filt_data  )
                        with tab_describe:
                            describe_metadata()
                            
         
        with main_tab_omics:
            omics_available = SAMPLES_OMICSDATA in st.session_state
            metadata_available = SAMPLES_METADATA in st.session_state

            match ( omics_available, metadata_available ):
                case (True, True):
                    omics_matrix, omics_features, missing_samples, nzv_features = st.session_state[ SAMPLES_OMICSDATA ]

                    for omics_name, omics_features in omics_features.items():
                        #TODO. add visualizations for omics data: barplots, histograms, etc.    
                        num_missing = len( missing_samples[omics_name] )
                        num_features = len( omics_features )
                        with st.expander(f"{omics_name} -> {num_features} features", expanded= False):

                            if nzv_features[omics_name]:


                                st.warning( f"Dropped {len(nzv_features[omics_name])} features with Near Zero Variance: {nzv_features[omics_name]}" )
                            if num_missing:
                                st.warning( f"Missing {len(missing_samples[omics_name])} samples:\n\n{missing_samples[omics_name]}" )
                            curr_df = pd.DataFrame( omics_matrix[omics_features] )
                            l, r = st.columns(2)
                            with l:
                                st.dataframe( curr_df.head(10) )
                            with r:
                                st.dataframe( curr_df.describe().T )
                                
                case (False, True):
                    st.warning("Please load omics data")
                case (_, False):
                    st.warning("Please load metadata first")


if SAMPLES_OMICSDATA in st.session_state and SAMPLES_METADATA in st.session_state:
    metadata = st.session_state[ SAMPLES_METADATA ].filt_data
    omics_matrix, omics_features, missing_samples, nzv_features = st.session_state[ SAMPLES_OMICSDATA ]
    my_df = pd.merge( metadata, omics_matrix, left_index = True, right_index = True )
    st.write( f"Shape of the final dataframe: {my_df.shape}")
    total_nf = 0 
    sorted_omics = sorted( omics_features.keys() )
    default_colors = [ 
        f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" 
            for color in generate_colors( len( sorted_omics ))]


    with st.form("form_alias_color"):
        for i, omic_id in enumerate( sorted_omics ):
            the_features = omics_features[omic_id]
            l, c, r = st.columns([0.4,0.1,0.5])
            with l:
                st.subheader( f"Omics layer: {omic_id}")
            with c:
                st.color_picker( f"Choose a color", key=f"color_{i}", value=default_colors[i] )
            with r:
                st.text_input( f"Set an alias for {omic_id}", value = omic_id, key=f"alias_{i}" )
                
            total_nf += len(the_features)
        st.write( f"Total number of features: {total_nf}" )

        if st.form_submit_button("Confirm aliases and colors"):
            st.dataframe( my_df.head(10), use_container_width=True )
            form_data = {
                omic_id: (st.session_state[f"alias_{i}"], st.session_state[f"color_{i}"])
                    # for omic_id in omics_features.keys()
                    for i, omic_id in enumerate( sorted_omics )
            }
            omics_features = {
                form_data[omic_id][0]: omics_features[omic_id]
                    for omic_id in omics_features.keys()
            }
            encodings = dict( colors = {
                omic_id: color_id for omic_id, color_id in form_data.values()
            })

            covariates_and_features = dict(
                covariates = metadata.columns.tolist(),
                **omics_features
            )
            

            st.session_state[ DATA_READY_TO_BE_DOWNLOADED ] = write_prepdata( 
                tsv_data = dict( omics_data = my_df), 
                feature_sets = covariates_and_features,
                encodings = encodings )
            

    if DATA_READY_TO_BE_DOWNLOADED in st.session_state:
        with open( st.session_state[ DATA_READY_TO_BE_DOWNLOADED ], "rb" ) as fp:
            st.download_button(
                "Download preprocessed data", 
                data = fp,
                file_name = "preprocessed_data.zip",
                mime = "application/zip"
            )
