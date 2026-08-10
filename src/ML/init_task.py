import streamlit as st 
from tools.utils import write_prepdata, HelpMessage
from tools.gtools import OmicsGraphBuilder, OmicsGraphFilter, gt 
from datetime import datetime

from tools.enums import Coefficients, enum
from itertools import combinations, chain
from typing import List, Dict 

from sklearn.preprocessing import StandardScaler

import plotly.express as px 
from tools.utils import (
    load_prepdata, tab_files_extensions)
import tools.ml_utils as mlu 
from tools.graph_fs import FeatureGraphEnricher
import tools.mongoml as mongoml

from time import time

import numpy as np, pandas as pd
from scipy.stats.mstats import gmean


CURRENT_STEP = "current_step"
THE_METADATA = "the_metadata"
THE_DATA = "the_data"
TRAIN_TEST = "train_test_data"
FEATURE_PER_OMIC = "feature_groups"
TMP_OMICS_DATA = "tmp_omics"
ZIPPED_DATA = "train_test_zip"
GRAPH_BUILT = "featuregraph_built"


class DataTransformation(enum.Enum):
    NONE = "None"
    LOG = "log(**x**+1)"
    SQRT = "sqrt(**x**)"
    CLR="clr(**x**)"

    @classmethod
    def get_transformation(cls) -> List[str]:
        return list( map( lambda x: x.value, cls ) )
    
    @classmethod
    def __get_function(cls, transformation: str):
        match cls( transformation ):
            case cls.LOG:
                return np.log1p
            case cls.SQRT:
                return np.sqrt
            case cls.CLR:
                delta = 1e-10
                return lambda x: np.log( (x+delta) / gmean(x +delta) ) 


            case _:
                return None
    
    @classmethod
    def get_transformed_data(cls, data: pd.DataFrame, transformation: str) -> pd.DataFrame:
        trans_data = data.copy()
        
        if (func := cls.__get_function( transformation )) is not None:
            trans_data = func( trans_data )

        return trans_data
    

    @classmethod
    def apply_transformation(cls, data: pd.DataFrame, features: List[str], transformation: str):
        if (func := cls.__get_function( transformation )) is not None:
            data[ features ] = data[features].apply( func )  #func( data[ features ] )
        return data.copy()


@st.fragment
def fragment_viz_tranformation(selected_omic: str):
    st.subheader(f"View {selected_omic} distributions")
    whole_featureset = np.array( list( st.session_state[ FEATURE_PER_OMIC ][ selected_omic ]))
    max_n_features = min( 50, len( whole_featureset ) )
    max_plottable_features = st.slider(
        "Max. number of features to plot", 
        min_value=1, 
        max_value=max_n_features,
        value= max_n_features // 3 + 2,
        step=1, 
        key=f"max_plottable_features_{selected_omic}" )
    plottable_features = np.random.choice(
        whole_featureset, 
        size = max_plottable_features, 
        replace=False ).tolist()

    selected_features = st.multiselect("Select features", options=plottable_features, default=plottable_features, key=f"selected_features_{selected_omic}")
    selected_transformation = st.pills("Select transformation", options=DataTransformation.get_transformation(), default="None", key=f"{selected_omic}_t_trans")
    selected_transformation = selected_transformation or "None"

    if len(selected_features) > 0:
        transformed_data = DataTransformation.get_transformed_data( 
            st.session_state[ THE_DATA ][ selected_features ], selected_transformation)
        
        if np.any( np.isnan( transformed_data.values ) ):
            st.warning("WARNING: Transformed data contains NaN values. Avoid this transformation unless you're looking for troubles.")

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.dataframe( transformed_data, use_container_width=True )
        with col2:
            fig = px.histogram( 
                data_frame = transformed_data, 
                x = selected_features, 
                barmode="overlay", 
                histnorm="probability density" )
            fig.update_traces(opacity=0.75)
            fig.update_layout(title_text=f"Distribution of {selected_omic} features")
            st.plotly_chart(fig, use_container_width=True)


def apply_data_transformations(
        data: pd.DataFrame,
        feature_groups: Dict[str, List[str]],
        transformation_map: Dict[str, str]) -> pd.DataFrame:
    
    new_data = data.copy()
    
    for omic_type, features in feature_groups.items():
        if (transformation := transformation_map.get( omic_type )) is not None:
            new_data = DataTransformation.apply_transformation( new_data, features, transformation )
    return new_data


def check_step( step: int ) -> bool:
    return st.session_state[ CURRENT_STEP ] == step


def prepare_train_test_data( ml_data, train_samples, test_samples ):
    train_data = ml_data.loc[ train_samples ].copy()
    test_data = ml_data.loc[ test_samples ].copy()

    return train_data, test_data


def wrap_validation_set(metadata_file, omics_files, target_cov: str, classes: List[str]):
    df_metadata = pd.read_csv( metadata_file, sep='\t', index_col=0, header=0)
    st.write(f"Target covariate: {target_cov}; classes: {classes}")
    df_metadata = df_metadata[ df_metadata[ target_cov ].isin( list( chain.from_iterable(classes) ) ) ].copy() 
    sample_set = set( df_metadata.index.tolist() )
    df_omics = list() 
    error = set() 


    for omics_file in omics_files:
        df = pd.read_csv( omics_file, sep='\t', index_col=0, header=0)

        if not sample_set.intersection( df.index.tolist() ):
            df = df.T

        if not sample_set.intersection( df.index.tolist() ):
            st.error(f"Validation set and omics data do not share any sample!")
            error.add(omics_file.name)
        else:
            sample_set = sample_set.intersection( df.index.tolist() )
            df_omics.append( df )

    if error:
        st.error(f"Something went wrong while loading the omics data from file(s) {error}")
        return None
    
    if len(df_omics) == 0:
        st.error(f"No omics data could be loaded!")
        return None
    
    if len(sample_set) == 0:
        st.error(f"No samples could be matched between metadata and omics data!")
        return None

    sample_set = list( sample_set )
    df_omics = pd.concat( [ df_metadata ] + df_omics, axis=1 ).loc[ sample_set ]

    # check whether the validation set contains (at least) the same features as the discovery set
    discovery_features = set( st.session_state[ THE_DATA ].columns.tolist() )
    validation_features = set( df_omics.columns.tolist() )

    if ( missing_features := discovery_features - validation_features ):
        st.error(f"Validation set misses the following {len(missing_features)} features: {missing_features}")
    else:
        return df_omics[ st.session_state[ THE_DATA ].columns ]


#### initialize page state
if CURRENT_STEP not in st.session_state:
    st.session_state[ CURRENT_STEP ] = 0

###
if TRAIN_TEST not in st.session_state:
    st.session_state[ TRAIN_TEST ] = dict() 

    
if FEATURE_PER_OMIC not in st.session_state:
    st.session_state[ FEATURE_PER_OMIC ] = dict() 

if "vset" not in st.session_state:
    st.session_state["vset"] = dict()


st.title("ML task initialization")


HelpMessage.init_task()

st.file_uploader( 
    label = "Preprocessed data", 
    type = "zip", 
    key = "zip_ml_input_data"
)


if st.session_state.zip_ml_input_data:
    match load_prepdata( st.session_state.zip_ml_input_data ):
        case data, feature_sets, encodings:
            
            assert (df := data.get("omics_data")) is not None 
            st.write("Number of samples:", df.shape[0])

            omics_counts = { omic_id: len(features) for omic_id, features in feature_sets.items() }
            omics_df = pd.DataFrame( data = list( omics_counts.items() ), columns = ["Omics", "Occurrences"]).set_index("Omics")
            st.dataframe( omics_df, use_container_width=True )
            
            
            st.session_state[ THE_DATA ] = df
            st.session_state[ THE_METADATA ] = df[ feature_sets.pop( "covariates" ) ].copy()
            st.session_state[ FEATURE_PER_OMIC ] = feature_sets
            st.session_state[ "__encodings" ] = encodings
            st.session_state[ "OMICS_FEATURES" ] = list( chain.from_iterable( feature_sets.values() ) ) 
        case _:
            st.error("Something went wrong while loading the data")


    split_data, viz_data = st.tabs( ["Split data", "Visualize data"] )


    with split_data:
        df_metadata = st.session_state[ THE_METADATA ] 
        st.title("Define the task")
        cl, cr = st.columns(2)
        with cl:
            n_samples, _ = df_metadata.shape
            st.dataframe( df_metadata, use_container_width=True )
        with cr:
            with st.container(border=True):
                df_metadata_value_counts = df_metadata[ st.session_state[ THE_METADATA ].columns.tolist() ].nunique()
                st.selectbox(
                    "Select target feature", key="target_covariate", 
                    options=df_metadata_value_counts[ df_metadata_value_counts > 1 ].index.tolist(), 
                    index=0)

                nested_cols = st.columns(2)
                with nested_cols[0]:
                    st.number_input(
                        "Select data percentage to be used as test set", 
                        key="tset_fraction", 
                        min_value=0, max_value=30, value=30, step=1, format="%d" )
                with nested_cols[1]:
                    st.multiselect(
                        "Sample stratification", 
                        key="sample_strata", 
                        options=df_metadata.columns.tolist())
                    
                values_cov = df_metadata[ st.session_state.target_covariate ].unique()   #set( df_metadata[ st.session_state.target_covariate ].values )
                with st.form(key="form_splitdata", border=False):
                    cols_classes = st.columns(2)
                    with cols_classes[0]:
                        st.multiselect("Select positive class", key="pos_class", options=values_cov, default=values_cov[0], help="Select one or more labels for the positive class.")
                    with cols_classes[1]:
                        st.multiselect("Select negative class", key="neg_class", options=values_cov, default=values_cov[1], help="Select one or more labels for the negative class.")

                    submit_pressed = st.form_submit_button("Split data")

            if submit_pressed:
                pos_class, neg_class = set(st.session_state.pos_class), set(st.session_state.neg_class)


                if (intersect := pos_class.intersection( neg_class )):
                    st.error(f"Positive and negative classes must be disjointed! Troubling labels: **{', '.join(intersect)}**")
                else:
                    interesting_labels = pos_class.union( neg_class )
                    target_cov = st.session_state.target_covariate
                    tset_fraction = st.session_state.tset_fraction / 100.
                    cov_strata = [ target_cov ] + st.session_state.sample_strata
                    df_metadata_subset = df_metadata[ df_metadata[ target_cov ].isin(interesting_labels) ].copy()
                    X_train, X_test = mlu.train_test_split__stratified( df_metadata_subset, cov_strata, size = tset_fraction)
                    train_samples, test_samples = X_train.index.tolist(), X_test.index.tolist()
                    st.info(f"Num. samples for training: {len(train_samples)}; num. samples for test: {len(test_samples)}. ")


                    ## saving set of indices for training and test sets
                    st.session_state[ TRAIN_TEST ] = dict( training_set = train_samples, test_set = test_samples )

                    st.session_state[ "task_data" ] = dict( 
                        tasktype = "classification", 
                        target = target_cov, 
                        pos_class = st.session_state.pos_class, neg_class = st.session_state.neg_class )

                    st.session_state[CURRENT_STEP] = 1

    with viz_data:
        
        
        for col in df_metadata.columns:
            n_unique = df_metadata[col].nunique()
            if 1 < n_unique < df_metadata.shape[0] - 10:
                fig = px.histogram( 
                    data_frame = df_metadata, 
                    x = col ) 
                fig.update_traces(opacity=0.75)
                fig.update_layout(title_text=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)


    with st.expander("Add Validation Set", expanded = check_step(1) ):

        if ( valids := st.session_state.get("vset") ):
            valids_names = "\n".join( f"* **{vname}**" for vname in valids.keys() )
            st.markdown(f"**{len(valids)}** validation set(s) already added\n{valids_names}")
        else:
            if st.button(f"Skip this step"):
                st.session_state[CURRENT_STEP] = 2

        col, cor = st.columns([0.5, 0.5])
        with col:
            st.file_uploader( 
                label = "Tabular file containing samples' clinical data", 
                type = tab_files_extensions, 
                key = "metadata_validation"
            )
        with cor:
            st.file_uploader( 
                label = "Tabular file containing samples' omics data", 
                key = "omics_validation", 
                type = tab_files_extensions,
                accept_multiple_files = True
            )

        if st.session_state.metadata_validation and st.session_state.omics_validation:
            with st.form("add_val"):
                st.text_input("Validation set name", key="vset_name", placeholder="e.g. Validation" )
                if st.form_submit_button("Add validation set"):
                    vset_name = st.session_state.vset_name.strip()
                    if len( vset_name ) == 0 or vset_name in st.session_state["vset"]:
                        st.error("Validation set with the same name already exists!")
                    else:
                        v_set = wrap_validation_set( 
                            st.session_state.metadata_validation, 
                            st.session_state.omics_validation, 
                            st.session_state.target_covariate, 
                            [ st.session_state.pos_class, st.session_state.neg_class ])
                        if v_set is not None:
                            st.session_state[ "vset" ][ st.session_state.vset_name ] = v_set
                            st.session_state[CURRENT_STEP] = 2

    if True:    
        with st.expander("Data transformation", expanded = check_step(2) ):
            omics_types = st.session_state[ FEATURE_PER_OMIC ].keys()
            tabs_trans = st.tabs( ["Transform data"] + [ f"View {omic_type} distributions" for omic_type in omics_types ] )
            
            
            with tabs_trans[0]:
                with st.form("data_transformation_form", border=False):
                    for omic_type in st.session_state[ FEATURE_PER_OMIC ].keys():
                        st.pills(f"**{omic_type}** data transformation", options=DataTransformation.get_transformation(), default="None", key=f"{omic_type}_transformation")

                    if st.form_submit_button("Select transformations"):
                        transformation_map = dict()

                        for omic_type, omics_features in st.session_state[ FEATURE_PER_OMIC ].items():
                            transformation_map[ omic_type ] = st.session_state[ f"{omic_type}_transformation" ]


                        st.success(f"Transformation list saved: {transformation_map}")
                        st.session_state["transformation_map"] = transformation_map
                        st.session_state[CURRENT_STEP] = 3

            for selected_omic, distr_tab in zip( omics_types, tabs_trans[1:] ):
                with distr_tab:
                    fragment_viz_tranformation(selected_omic)


    with st.expander("Feature graph", expanded = check_step(3)):
        graph_presence = GRAPH_BUILT in st.session_state
        if ZIPPED_DATA in st.session_state:
            
            st.info(f"""Feature graph included in the zip file:\n{st.session_state.feature_graph}""")

            with open( st.session_state[ ZIPPED_DATA ], "rb" ) as fp:
                

                st.download_button(
                    f"Download ML data", 
                    data = fp,
                    file_name = "data_for_ML.zip", #fp.name, 
                    mime = "application/zip"
                ) 
        with st.form("feature_graph_form"):
            st.subheader("Run this once you completed data manipulation steps...")
            form_submit_label = "Build feature graph" if not graph_presence else "**Re**build feature graph"

            st.number_input(f"Significance threshold (adj. p-value)", key="signif_threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3g" )


            if st.form_submit_button(form_submit_label):
                with st.spinner():
                    st.write("Applying data transformations...")
                    
                    ### TODO: apply data transformations to validation sets

                    feature_groups = st.session_state[ FEATURE_PER_OMIC ]
                    train_test_samples = st.session_state[ TRAIN_TEST ]
                    transformation_map = st.session_state.get( "transformation_map", dict() )
                    whole_data = apply_data_transformations( st.session_state[ THE_DATA ], feature_groups, transformation_map )
                    training_data, test_data = prepare_train_test_data( whole_data, train_test_samples[ "training_set" ], train_test_samples[ "test_set" ] )

                    train_test = dict( training_set = training_data, test_set = test_data )

                    omics_data_train = {
                        omic_name: training_data[ feature_list ].copy()
                            for omic_name, feature_list in feature_groups.items()
                    }
                    raw_metadata_train = st.session_state[ THE_METADATA ].loc[ training_data.index ].copy()

                    ########################################
                    task_info = mlu.TaskParameters(
                        target_cov = st.session_state.task_data["target"],
                        pos_class = st.session_state.task_data["pos_class"],
                        neg_class = st.session_state.task_data["neg_class"] )

                    data_identifier = task_info.get_unique_study_identifier( st.session_state[ FEATURE_PER_OMIC ] )

                    #########################################

                    net_found = mongoml.search_graph_with_metadata( data_identifier, p_threshold=st.session_state.signif_threshold)
                    save_net_in_db = True 

                    if net_found is None:

                        st.write("Starting...")
                        gbuilder = OmicsGraphBuilder( omics_data_train, Coefficients.SPEARMAN.value )
                        gbuilder.init_vertices()
                        st.write("Vertices created. What about the edges?")
                        
                        gbuilder.build_edges(
                            df = raw_metadata_train, 
                            strat=st.session_state.target_covariate, 
                            intra_list=list( omics_data_train.keys() ), 
                            inter_list=list( combinations( omics_data_train.keys(), 2 ) )
                        )
                        st.write("We got the edges!")
                        gbuilder.create_edges() 
                        st.write("Edges created. Now we compute the edge weights")
                        
                        feature_graph = gbuilder.graph
                        if st.session_state.signif_threshold < 1.0:
                            st.write(f"Dropping edges associated to non-significant correlations (adj. pvalue > {st.session_state.signif_threshold})")
                            OmicsGraphFilter.apply_graph_edge_filtering(
                                mog = feature_graph, 
                                key = "WholeData", 
                                corr_thr=0, 
                                pv_thr=st.session_state.signif_threshold, 
                                use_adj_pvalue = True )

                            feature_graph = gt.Graph( g = feature_graph, prune = True )

                        
                        raw_omics_data = pd.concat( omics_data_train.values(), axis = 1 )
                        raw_omics_data = pd.DataFrame(
                            data = StandardScaler().fit_transform( raw_omics_data ),
                            columns = raw_omics_data.columns,
                            index = raw_omics_data.index
                        )
                        training_data = mlu.LabelledData( 
                            raw_omics_data, 
                            raw_metadata_train[ st.session_state.target_covariate ].isin(st.session_state.pos_class), #== st.session_state.pos_class,
                            None, None 
                        )

                        st.write(f"Computing collection of vertex weights ;)")
                                
                        st.write(f"* AUC-based vertex weights...")
                        start_time = time()
                        df_aucs = FeatureGraphEnricher.compute_AUC_vertex_weights( feature_graph, training_data )
                        print(f"######### Done in {time() - start_time:.2f} seconds")
                        st.write(f"* Stat-based vertex weights...")
                        df_scores = FeatureGraphEnricher.compute_vertex_weights( feature_graph, training_data )

                        st.write(f"Saving vertex weights in the feature graph ;)")
                        df_weights_normalized = st.session_state[ "df_weights" ] = pd.concat([ df_scores, df_aucs ], axis=1)
                        numeric_cols = df_weights_normalized.columns.tolist()[1:]
                        FeatureGraphEnricher.set_vertex_weights( feature_graph, st.session_state.target_covariate, df_weights_normalized[ numeric_cols ] )
                    
                    else:
                        st.write(f"A feature graph with the same identifier already exists in the database. Retrieving it ({net_found})...")
                        feature_graph = mongoml.retrieve_graph_from_mongo__with_id( net_found[0] )
                        save_net_in_db = False

                        if st.session_state.signif_threshold < net_found[1]:
                            st.write(f"Graph retrieved: {feature_graph} --  applying p-value filtering with threshold {st.session_state.signif_threshold}...")
                            OmicsGraphFilter.apply_graph_edge_filtering(
                                mog = feature_graph, 
                                key = "WholeData", 
                                corr_thr=0, 
                                pv_thr=st.session_state.signif_threshold, 
                                use_adj_pvalue = True )
                            st.write(f"Graph filtered. New graph: {feature_graph}")
                            feature_graph = gt.Graph( g = feature_graph, prune = True )
                            save_net_in_db = True

                    st.write("Ok got it. Zipping the output files...")

                    ### TODO: wrap zip file creation in a function
                    task_data = st.session_state[ "task_data" ]
                    
                    id_data_available = dict( 
                        training_set = "training_set", 
                        test_set = "test_set", 
                        validation_set = list( st.session_state[ "vset" ].keys() ) ) 

                    print(f"My data available: {id_data_available}")
                    encodings = dict( 
                        data = id_data_available,
                        task_info = task_data, 
                        **st.session_state[ "__encodings" ] )
                    covariates_and_features = dict(
                        covariates = st.session_state[ THE_METADATA ].columns.tolist(), 
                        **st.session_state[ FEATURE_PER_OMIC ]
                    )
                    tsv_data_available = { k: v for k, v in train_test.items() }

                    ## apply data transformations to validation sets
                    validation_sets = {
                        vset_name: apply_data_transformations( vset_data, feature_groups, st.session_state["transformation_map"] )
                            for vset_name, vset_data in st.session_state[ "vset" ].items()
                    }

                    tsv_data_available.update( validation_sets )

                    st.session_state[ ZIPPED_DATA ] = write_prepdata(
                        tsv_data = tsv_data_available, # s#t.session_state[ TRAIN_TEST ], 
                        feature_sets = covariates_and_features, #st.session_state[ FEATURE_PER_OMIC ], 
                        encodings = encodings
                    )

                    net_metadata = dict(
                        creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        n_nodes = feature_graph.num_vertices(), 
                        n_edges = feature_graph.num_edges(), 
                        omic_layers = sorted( st.session_state[ FEATURE_PER_OMIC ].keys() ), 
                        p = st.session_state.signif_threshold
                    )

                    ### check whether the graph is already in the DB


                    mongoml.save_graph_in_mongo( 
                        g_id = data_identifier, 
                        g = feature_graph, 
                        metadata = net_metadata )

                    st.session_state[ "feature_graph" ] = feature_graph
                    st.session_state[ GRAPH_BUILT ] = True
                    st.rerun()


