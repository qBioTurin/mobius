import streamlit as st 
from typing import List, Tuple, Dict, Optional
import zipfile, tempfile
import graph_tool.all as gt 
import pandas as pd, numpy as np
import plotly.graph_objects as go
from itertools import product
from tools.utils import  load_prepdata, HelpMessage, read_graph_from_zip

from tools.enums import GraphField, EdgeType, GraphFilteringParametrization, FilteringParameters, GraphStats, SessionState
from tools.gtools import prepare_graph, get_input_graph_names, build_g_dataframe, GraphStatsViz, CommunityDetector, compute_centralities, OmicsGraphFilter

import logging


@st.cache_resource
def load_graph_from_zip(input_zipfile, graph_filename) -> gt.Graph:
    
    if input_zipfile is not None:
        return read_graph_from_zip( input_zipfile, graph_filename )

        
def on_change_read_graph_db():
    def preprocess_omics_graph( g: gt.Graph ):
        if g is None:
            return False
        if GraphField.FEATURE_OMIC.value not in g.vp:
            return None
        return list( g.vp[ GraphField.FEATURE_OMIC.value ] )

    # """ Extract metadata (e.g. set of omics) from graph collection"""

    with st.spinner("Reading graph collection metadata"):
        input_graph_archive = st.session_state.input_graphs
        print(f"Uploaded file is {input_graph_archive}")


        if input_graph_archive is not None: 
            omic_set = set()
            default_graph = False 
            archive_filename = input_graph_archive.name 

            if archive_filename.endswith(".zip"):
                _, _, encodings = load_prepdata( input_graph_archive)
                st.session_state[ "palette" ] = encodings.get("colors")

                for gt_net in get_input_graph_names( input_graph_archive ):
                    curr_g = load_graph_from_zip( input_graph_archive, gt_net )
                    match preprocess_omics_graph( curr_g ):
                        case None: 
                            break
                        case False:
                            logging.error(f"Cannot load {gt_net} from source '{input_graph_archive}'\n\n")
                        case preprocessed_data:
                            omic_set = omic_set.union( preprocessed_data )
                            
                else:
                    default_graph = True 

            elif archive_filename.endswith(".gt") or archive_filename.endswith(".gt.xz"):
                curr_g = gt.load_graph( input_graph_archive )
                preprocessed_data = preprocess_omics_graph( curr_g )
                if preprocessed_data:
                    omic_set = set( preprocessed_data )


            st.session_state[ SessionState.AVAIL_OMICS.value ] = tuple( sorted( omic_set ) )
            st.session_state[ SessionState.IS_DEFAULT_GRAPH.value ] = default_graph
            st.session_state[ SessionState.LAST_UPLOADED_FILENAME.value ] = input_graph_archive.name 


def zip_analysis_results():
    with zipfile.ZipFile( tempfile.NamedTemporaryFile(prefix="net_analysis_", suffix=".zip", delete=False), 'w') as zip_out:
        for graph_id, obj_results in st.session_state[ SessionState.NET_RESULTS.value ].items():
            zipped_stuff = obj_results.build_zip()
            zip_out.write( zipped_stuff, f"data_{graph_id}.zip" )
            
        st.session_state[ "zipped_results" ] = zip_out.filename


@st.cache_resource
def network_analysis( _g, g_params: GraphFilteringParametrization, feature_vprop_name: str, color_palette: Dict[str,str] = None ) -> GraphStats:
    v_data, gt_props = get_centrality_measures( _g, g_params, feature_vprop_name )
    e_data = build_edge_based_dataframe( 
        _g, gt_props["edge_betweenness"].a, 
        feature_vprop_name )
    fig_degree_distr, fig_degree_corr = build_degree_figures( _g, g_params.filtering_params.chosen_omics, color_palette )
    fig_avg_cc = GraphStatsViz( _g ).get_clustering_coefficient_figure( 
        gt_props["clustering"].a, 
        get_figure=True, 
        color_palette=color_palette)
    dfff = get_community_detection(_g, g_params, feature_vprop_name )
    
    if ( omic_fieldname := GraphField.FEATURE_OMIC.value ) in v_data.columns:
        # dfff[ GraphField.FEATURE_OMIC.value ] = v_data[ GraphField.FEATURE_OMIC.value ]
        dfff.insert( 0, omic_fieldname, v_data[ omic_fieldname ] )

    return GraphStats( v_data, gt_props, dfff, e_data, fig_degree_distr, fig_degree_corr, fig_avg_cc )


def build_degree_figures( g, omics_list, color_palette: Dict[str,str] ):
    gsv = GraphStatsViz( g )

    fig_degree_distr = go.Figure() 

    if omics_list:
        for omic_data in omics_list:
            fig_degree_distr.add_trace( gsv.get_degree_distribution_figure( omic_data, color_palette=color_palette ) )

    fig_degree_distr.add_trace( gsv.get_degree_distribution_figure() )

    fig_degree_distr.update_layout(
        title="Degree distribution",
        xaxis_title="Degree",
        yaxis_title="Probability",
        legend_title="Graph traces" )

    fig_degree_corr = go.Figure()
    fig_degree_corr.add_trace( gsv.build_degree_correlation_function() )
    fig_degree_corr.update_layout(
        title="Degree correlation function",
        xaxis_title="Degree",
        yaxis_title="knn(k)" )
    
    
    return fig_degree_distr, fig_degree_corr


@st.cache_data
def get_centrality_measures( _g, g_params: GraphFilteringParametrization, vertex_name_prop: str = None) -> Tuple[ pd.DataFrame, Dict ]: 
    def get_neighborhood( _g: gt.Graph, g_params: GraphFilteringParametrization ) -> pd.DataFrame:
        key = g_params.graph_key
        ## col 0: num. pos correlations 
        ## col 1: num. neg correlations 
        ## col 2: avg pos. correlations
        ## col 3: avg neg. correlations
        neighborhood_matrix = np.zeros(shape=(_g.num_vertices(), 5))

        for u, v, w in _g.iter_edges( [_g.ep[ key ]] ):
            index = 0 if w > 0 else 1 
            ## count occurrences 
            neighborhood_matrix[u,index] += 1
            neighborhood_matrix[v,index] += 1
            ## update cumulative sum 
            neighborhood_matrix[u,index+2] += w
            neighborhood_matrix[v,index+2] += w

        ## get average correlation in magnitude 
        degrees = neighborhood_matrix[:,0] + neighborhood_matrix[:,1]
        neighborhood_matrix[:,4] = (neighborhood_matrix[:,2] - neighborhood_matrix[:,3]) / degrees
        ## compute average correlations among neighbors 
        neighborhood_matrix[:,2] /= neighborhood_matrix[:,0]
        neighborhood_matrix[:,3] /= neighborhood_matrix[:,1]

        return build_g_dataframe( _g, columns_data = dict(
            ne_pos = neighborhood_matrix[:,0], 
            ne_neg = neighborhood_matrix[:,1], 
            avg_we_pos = neighborhood_matrix[:,2], 
            avg_we_neg = neighborhood_matrix[:,3],
            avg_we_abs = neighborhood_matrix[:,4] 
        ))


    df, props = compute_centralities( _g, vertex_name_prop )
    df = pd.concat([ df, get_neighborhood( _g, g_params ) ], axis=1)
    return df, props 


@st.cache_resource
def get_community_detection( _g, g_params: GraphFilteringParametrization, vertex_name_prop: str = None ):
    communities, nested_sbm = CommunityDetector.community_detection( _g, vertex_name_prop )
    return communities


def build_edge_based_dataframe( g: gt.Graph, eb: gt.EdgePropertyMap, feature_name_vprop: str ) -> pd.DataFrame:
    def get_significance_symbol(pvalue):
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*" if pvalue < 0.05 else ""

    two_cols_for_index = ["f1", "f2"]
    feature_vprop = feature_name_vprop if feature_name_vprop else GraphField.FEATURE_NAME.value

    if feature_name_vprop is None or feature_vprop == GraphField.FEATURE_NAME.value:
        strata_fields = [ GraphField.GRAPH_REPR, GraphField.PVALUE, GraphField.ADJ_PVALUE ]
        strata = ["WholeData"] + OmicsGraphFilter.get_strata( g ) 
        
        wholeset_ep_fields = [ field.value.format( s ) for s in strata for field in strata_fields ]
        wholeset_ep_fields.insert(0, GraphField.INTRAOMIC_EDGE.value )
        eprops = [ g.ep[ field ] for field in wholeset_ep_fields ]
        edges_data = list( g.iter_edges( eprops ) )


        columns = two_cols_for_index + wholeset_ep_fields
    else:
        edges_data = list( g.iter_edges() )
        columns = two_cols_for_index


    degrees = g.degree_property_map("out").a
    vpn = g.vp[ feature_vprop ]

    edges_df = pd.DataFrame(
        data = edges_data,
        columns = columns
    )
    ### replace integers with feature names
    degree_lists = list()
    for fcol in columns[:2]:
        degree_lists.append( edges_df[ fcol ].map( lambda v_id: degrees[ v_id ] ) )
        edges_df[ fcol ] = edges_df[ fcol ].map( lambda v_id: vpn[ v_id ] )

    edges_df.insert(2, "degree_diff", np.abs( degree_lists[0] - degree_lists[1] ) )
    edges_df.insert(3, "eb", eb )

    padj_cols = filter( lambda col_name: col_name.startswith("padj_"), edges_df.columns.tolist()  )

    for padj_col in padj_cols:
        column_set = edges_df.columns.tolist() 
        edges_df.insert( 
            column_set.index( padj_col ) + 1, 
            f"stat.sign_{padj_col}",
            edges_df[ padj_col ].map( lambda padj: get_significance_symbol( padj ) )
        )

    return edges_df.set_index( two_cols_for_index )


def load_net_UI():

    net_results = st.session_state.get( SessionState.NET_RESULTS.value )

    if net_results is not None:
        with st.container(border=True):
            st.header(f"Download data")
            st.subheader(f"Latest uploaded zip  { st.session_state[ SessionState.LAST_UPLOADED_FILENAME.value ] }")
            with open( st.session_state.zipped_results, "rb" ) as fp:
                st.download_button(
                    "Download analysis results", 
                    data = fp,
                    file_name = "results_network_analysis.zip", 
                    mime = "application/zip" )
    else:
        with st.container( border = True ):
            st.header("Graph loading")
            input_filename = st.file_uploader( 
                label = "Select a graph-tool file", 
                type = ["gt", "zip", "graphml"], 
                key = "input_graphs", 
                on_change = on_change_read_graph_db
            )

        if st.session_state.input_graphs is not None:
            with st.container(border=True):
                st.title("Graph parametrization")

                if st.session_state[ SessionState.IS_DEFAULT_GRAPH.value ]: 

                    st.radio("Choose mode", options = ["Single Graph", "Graph collection"], key="graph_mode", index = 1, horizontal = True )
                    single_graph_mode = st.session_state.graph_mode == "Single Graph"

                    pick_string = "Select a graph" if single_graph_mode else "Select one or more graphs"

                    selected_g_id = st.pills(
                        pick_string, 
                        key = "choose_graph_id",
                        options = get_input_graph_names( input_filename ), 
                        selection_mode = ( "single" if single_graph_mode else "multi" ) )

                    graph_classes = [ "WholeData" ]

                    if single_graph_mode and selected_g_id:
                        my_g = load_graph_from_zip( st.session_state.input_graphs, selected_g_id ) #[0]  )
                        graph_classes.extend(my_g.gp[ GraphField.STRAT_LAYERS.value ].split("$"))
                        selected_g_id = [ selected_g_id ]

                    with st.form("default_graph_form"):

                        pill_label = "Select one or more strata" if single_graph_mode else "Use default stratum"
                        st.pills( pill_label, options = graph_classes, selection_mode ="multi", key = "graph_keyclass", default=graph_classes[0], disabled=( not single_graph_mode))


                        col_l, col_r = st.columns([0.45, 0.55])
                        
                        with col_l:
                            st.pills(
                                "Select omics to be considered", 
                                options = st.session_state[ SessionState.AVAIL_OMICS.value ], 
                                default = st.session_state[ SessionState.AVAIL_OMICS.value ],
                                selection_mode="multi",
                                key = "omics_types"
                            )
                            st.number_input(
                                "Set a correlation threshold x in [0,1)", 
                                key = f"edge_threshold", 
                                min_value=0., max_value=1., value=0.,
                                format="%f", 
                                placeholder="Type a number...", 
                                help="Edges with absolute correlation below this threshold will be removed" )
                            

                        with col_r:
                            st.radio(
                                "Choose the desired type of interactions (i.e. edges) to be considered", 
                                ["All", "Intra-omic", "Inter-omic"], 
                                captions = ["All", "Intra", "Inter"], 
                                key = "edge_type_selector",
                                index = 0, horizontal = True )
                            inner_cols = st.columns([0.7, 0.3])
                            with inner_cols[0]:
                                st.radio(
                                    "select a p-value threshold", 
                                    key = "pvalue_threshold",
                                    help="Edges with p-value above this threshold will be removed",
                                    options = [ None, 0.05, 0.01, 0.001 ], index = 1, horizontal=True ) 
                            with inner_cols[1]:                                
                                st.toggle("Use adjusted p-values", value=True, key="pvalue_use_adj")

                    
                        if st.form_submit_button("Submit", disabled = not bool( selected_g_id )):

                            if not st.session_state.omics_types:
                                st.error("Select at least one omic layer")
                                st.stop()

                            st.write(f"Desired edges: {st.session_state.edge_type_selector}")
                            edge_selector = EdgeType.from_string( st.session_state.edge_type_selector.lower() )
                            omics_types = tuple( sorted( st.session_state.omics_types, reverse=True ) )
                            st.session_state[ SessionState.NET_RESULTS.value ] = net_results = dict() 
                            st.session_state[ SessionState.CURRENT_GRAPH.value ] = curr_graphs = dict() 
                            st.session_state[ SessionState.VPROP_FEATURE_NAME.value ] = GraphField.FEATURE_NAME.value
                            st.session_state[ SessionState.FILTERING_PARAMS.value ] = filt_params = FilteringParameters( 
                                st.session_state.edge_threshold, 
                                st.session_state.pvalue_threshold,
                                st.session_state.pvalue_use_adj, 
                                omics_types, 
                                edge_selector
                            )


                            st.write(selected_g_id)
                            st.write(st.session_state.graph_keyclass    )

                            for chosen_graph, key_class in product( selected_g_id, st.session_state.graph_keyclass ):
                                curr_params = GraphFilteringParametrization( chosen_graph, key_class, filt_params )
                                curr_graph_id = chosen_graph if not single_graph_mode else f"{key_class}__{chosen_graph}"
                                
                                st.write(f"Loading graph {chosen_graph} ({key_class})")
                                my_g = load_graph_from_zip( st.session_state.input_graphs, chosen_graph )
                                st.session_state["my_palette_net"] = st.session_state["palette"]
                                st.write(f"* Preparing graph: (nv: {my_g.num_vertices()}, ne: {my_g.num_edges()})")
                                curr_graphs[ curr_graph_id ] = working_g = prepare_graph( my_g, curr_params, remove_zero_degree_nodes=False )
                                st.write(f"* Performing network analysis: (nv: {working_g.num_vertices()}, ne: {working_g.num_edges()})")
                                net_results[ curr_graph_id ] = network_analysis( working_g, curr_params, GraphField.FEATURE_NAME.value, st.session_state.my_palette_net )
                                st.write(f"* Computing vertex positioning for visualization purposes")
                                st.session_state[ SessionState.V_POS.value.format(curr_graph_id) ] = gt.sfdp_layout( working_g )

                            st.write("Zipping results")
                            zip_analysis_results()

                else:
                    st.error(f"Unsupported graph format: {input_filename.split('.')[-1]}")


st.title("Network Loading")

HelpMessage.load_zip_nets()

load_net_UI()