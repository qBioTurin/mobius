import streamlit as st 

import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots 


from itertools import chain
from typing import List, Tuple, Dict
from dataclasses import asdict

import tools.ml_utils as mlu
from tools.utils import build_tsv_archive, HelpMessage
from tools.gtools import GraphField, CommunityDetector, OmicsGraphFilter, prepare_graph, graph_reweighting, compute_centralities
from tools.enums import FilteringParameters, GraphFilteringParametrization, EdgeType    
import tools.graph_fs as gfs
from tools.gviz import get_graph_figure, PlotlyGraphVizManager
from scipy.cluster import hierarchy
from tools.helpers import HelperGeneticAlgorithm

import scipy.cluster.hierarchy as sch
from time import time
from multiprocessing import cpu_count


@st.cache_data
def prepare_feature_graph( 
        data_id: str, 
        initial_feature_sets: List[List[str]], 
        vertex_threshold: float,
        edge_threshold: float, 
        vfilt_strategy: str ) -> Tuple[mlu.Graph, pd.DataFrame]:
    
    feature_list = mlu.prepare_features( initial_feature_sets, st.session_state )
    st.markdown(f"Selected features: **{initial_feature_sets}** -> n.features: {len(feature_list)}")
    my_graph = OmicsGraphFilter.get_subgraph( st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ], feature_list )
    st.write(f"Initial feature graph -> **nv: {my_graph.num_vertices()}, ne: {my_graph.num_edges()}**")

    if vfilt_strategy in ("Median-based", "Mean-based"):
        st.write(f"Performing vertex filtering based on {vfilt_strategy}...")

        vertex_scores = my_graph.vp[ GraphField.VERTEX_WEIGHT.value.format("avg_vscore")].a
        threshold = np.median(vertex_scores) if vfilt_strategy == "Median-based" else np.mean(vertex_scores)
        vertex_bitmask = (vertex_scores > threshold)

        my_graph = OmicsGraphFilter.set_vertex_filter( my_graph, vertex_bitmask, ret_new_graph=True )
        st.write(f"Filtered graph w.r.t. {vfilt_strategy} -> **nv: {my_graph.num_vertices()}, ne: {my_graph.num_edges()}**")


    if vertex_threshold is not None and vertex_threshold < 1.: 
        vweights = my_graph.vp[ GraphField.WX_PVALUE.value ].a
        my_graph = OmicsGraphFilter.set_vertex_filter( my_graph, vweights < vertex_threshold, ret_new_graph=True )
        st.write(f"Vertex-pruned feature graph -> Wilcoxon p-value < {vertex_threshold}: **nv: {my_graph.num_vertices()}, ne: {my_graph.num_edges()}**")


    ##
    omics_list = OmicsGraphFilter.get_omics_labels( my_graph )
    if edge_threshold is not None and edge_threshold < 1.: 
        filtering_params = FilteringParameters(
            min_corr_threshold=0., pvalue_threshold = edge_threshold, padj_flag=True, chosen_omics=omics_list, edge_type=EdgeType.ALL_EDGES)
        gf_params = GraphFilteringParametrization( 
                    f"Graph Wholeee", "WholeData", filtering_params  )
        my_graph = prepare_graph( my_graph, gf_params, remove_zero_degree_nodes=False )
        st.write(f"Edge-pruned feature graph -> adj.p < {edge_threshold}: **nv: {my_graph.num_vertices()}, ne: {my_graph.num_edges()}**")


    if vfilt_strategy == "Degree-based":
        st.write(f"Performing degree-based filtering ...")
        degrees = my_graph.get_out_degrees( list( my_graph.vertices() ) )
        my_graph = OmicsGraphFilter.set_vertex_filter( my_graph, degrees > 0, ret_new_graph=True )
    

    st.write(f"Performing community detection")
    g_communities, _ = CommunityDetector().community_detection( my_graph, vertex_name_prop=GraphField.FEATURE_NAME.value )
    return (my_graph, g_communities)


def draw_heatmap_population( solutions: List[mlu.FeatureSubset], feature_to_omic: Dict[str, str] ):
    fpool, _ = zip(*sorted( feature_to_omic.items(), key=lambda x: x[1] ))
    fpool = { v:k for k, v in enumerate(fpool) }
    mb = np.zeros((len(solutions), len(fpool)))

    for i, sol in enumerate(solutions):
        indices = np.array([ fpool[f] for f in sol.feature_list ])   #sol["flist"] ])
        mb[i, indices] = 1

    df = pd.DataFrame(mb, columns = fpool.keys() )#, index = [ f"sol_{i+1}" for i in range(len(solutions)) ])

    rgb_lambda = lambda hex_str: ( int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16) )   #f"rgb({int(hex_str[1:3], 16)}, {int(hex_str[3:5], 16)}, {int(hex_str[5:7], 16)})"
    rgb_palette, rgba_palette = dict(), dict()
    for omic, hex_color in st.session_state.my_palette.items():
        r, g, b = rgb_lambda( hex_color )
        rgb_palette[omic] = f"rgb({r},{g},{b})"
        rgba_palette[omic] = f"rgba({r},{g},{b}, 0.2)"

    # Costruiamo la matrice RGB (shape: num_solutions x num_features x 3)
    colors = list() 

    for i, sol in enumerate(solutions):
        row_color = list() 
        for f in fpool:
            index, omic = fpool[f], feature_to_omic[f]
            row_color.append(( rgb_palette[omic] if f in sol.feature_list else rgba_palette[omic])) 
            
        colors.append(row_color)
        
    z = np.zeros_like(df.values)
    nrows, ncols = df.shape  #len(fpool), len(solutions)
    flat_colors = sum(colors, [])  # Flatten riga per riga
    z_flat = list(range(len(flat_colors)))


    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=df.columns.tolist(),
        y=[f"Sol {i+1}" for i in range(df.shape[0])],
        showscale=False,
        hoverinfo='text',
        text=df.astype(str),
        colorscale='gray'  # dummy, verrà sovrascritto sotto
    ))

    fig.data[0].z = np.array(z_flat).reshape(nrows, ncols)
    fig.data[0].colorscale = [[i / (len(flat_colors) - 1), color] for i, color in enumerate(flat_colors)]

    # Aggiungi legenda personalizzata
    for cat, rgb in rgb_palette.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=rgb),
            name=f"{cat}",
            showlegend=True
        ))

    fig.update_layout(
        title="MOEA Population Heatmap",
        xaxis_title="Features",
        yaxis_title="Soluzioni",
        xaxis=dict(tickangle=45),
        legend_title="Omics",
        legend=dict(itemsizing='constant')
    )

    return fig 


#         y_axes_dict[ name ] = y_axis
#             **scatter_args ))

#         y_axes_dict[ omic_id ] = omic_y_axis 
#                 **scatter_args ))
    
#     ##set name of the legend 


@st.fragment 
def explore_ga_generation( ):
    with st.container(border=True):
        h_index, solutions, feature_pool = st.session_state.info_GA_curr
        st.header(f"Explore {h_index+1}-th generation")

        l, c = st.columns([0.7, 0.3])
        with l:
            threshold_min_freq = st.slider("Frequency threshold", 0.0, 1.0, step=0.01)

            cd_algo = st.session_state[ "GA_result" ][ "ga_params" ]["cd_algorithm"] 

            fig_fc, df_barplot = mlu.histogram_feature_counts( 
                mlu.Counter( chain.from_iterable( [ sol.feature_list for sol in solutions ] ) ),
                st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ], 
                st.session_state[ "info_graph_fs" ][1].get(cd_algo),
                st.session_state.my_palette, 
                min_freq=threshold_min_freq, 
                return_df=True )
            df_barplot.set_index("feature", inplace=True)

        with c:
            above_thr = len( df_barplot[ df_barplot["frequency"] >= threshold_min_freq ] )

            st.metric(
                label=f"Num. features above the frequency threshold", 
                value=above_thr, 
                delta=f"{100*(above_thr/len(df_barplot)-1):.2f}%")
            
        with st.expander("View Dataframe", expanded=False):
            st.dataframe(df_barplot, use_container_width=True)
        st.plotly_chart(fig_fc, use_container_width=True)


        ga_params = st.session_state.get("GA_result").get("ga_params")
        cd_algorithm, vscore, pthreshold, starting_features = [ ga_params[key] for key in ["cd_algorithm", "vertex_score", "p_threshold", "feature_pool"]]
        ga_settings = f"GA_gen_{h_index}__{cd_algorithm}_{vscore}"


        c1, c2, c3 = st.columns(3)
        flag_see_gen = False 
        message = None 

        with c1:
            if st.button(f"Save current feature pool!"):
                filtered_pool = df_barplot[ df_barplot["frequency"] >= threshold_min_freq ].index.tolist()
                st.session_state[ mlu.ML_SessionState.FEATURES.value ].add_feature_set( filtered_pool, ga_settings )
                mlu.update_feature_dataframe()
                message = (True, f"Feature pool has been saved!!!")
                

        with c2:
            if st.button("See current generation!"):
                flag_see_gen = True 
        with c3:
            if st.button("Save current generation!"):
                chosen_gen = mlu.GAGeneration( 
                    name = f"pop_{ga_settings}", 
                    feature_pool=df_barplot.index.tolist(), 
                    solutions=[ sol.feature_list for sol in solutions ] ) 
                
                try:
                    if st.session_state[ mlu.ML_SessionState.FEATURES.value ].add_ga_generation( chosen_gen ):
                        message = (True, f"The current generation has been saved as {chosen_gen.name}!")
                    else:
                        message = (False, f"Failed to save the current generation as {chosen_gen.name}.")
                except AssertionError:
                    message = (False, f"Generation {chosen_gen.name} already exists!")

        if flag_see_gen:
            feature_to_omic = { f: o for f, o in zip( df_barplot.index.tolist(), df_barplot.omic.tolist() ) }
            fig = draw_heatmap_population(  solutions, feature_to_omic )
                
                #[sol.feature_list for sol in solutions], feature_to_omic )
            st.plotly_chart(fig, use_container_width=True)

        elif message:
            flag, message = message
            func = st.success if flag else st.warning
            func(message)
            

#             for i, flist in enumerate( solutions )
    

#             # ## retrieve GA params 


#     ## draw convergence plot 


def network_fsel_UI():
    feature_found_df = st.session_state[mlu.ML_SessionState.FEATURE_FOUND.value ]

    with st.form("configure_graph"):
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.multiselect("Choose initial features: ", key="features4net", 
                    options = feature_found_df.index.tolist(), 
                    default = st.session_state.get( mlu.ML_SessionState.SELECTED_FEATURES.value ), 
                    help=HelperGeneticAlgorithm.INITIAL_FEATURES.value)
        with cols[1]:
            st.number_input("Minimum community size", min_value=1, value=3, key="min_community_size", help=HelperGeneticAlgorithm.MIN_COMMUNITY_SIZE.value)

        
        cols = st.columns(3)

        with cols[0]:
            st.number_input("Feature filtering (Wilcoxon p-value)", min_value=0.0, max_value=1.0, step=0.001, format="%0.3g",  value=0.05, key="wx_filt") #, help=HelperGeneticAlgorithm.WILCOXON_PVALUE.value)

        with cols[1]:
            st.number_input(
                "Edge-based pruning (p-value threshold)", 
                min_value=0.0, max_value=1.0, step=0.001, value=0.05, 
                format="%0.3g", key="pt_graph", 
                help=HelperGeneticAlgorithm.P_THRESHOLD.value)
            
        
        with cols[2]:
            st.pills("Vertex-based filtering", 
                     options=["Disabled", "Degree-based"], #, "Median-based", "Mean-based"], 
                     key="vertex_filtering_strategy", default="Disabled", help=HelperGeneticAlgorithm.VERTEX_FILTERING.value)

        if st.form_submit_button("Set current parameters"):
            if not st.session_state.features4net:
                st.error("Select at least one feature set!")
            else:
                with st.status("Network preparation...", expanded = True) as status:
                    
                    st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ] = st.session_state.features4net 

                    curr_g, communities = prepare_feature_graph( 
                        st.session_state[ mlu.ML_SessionState.INIT_SESSION.value ],
                        st.session_state.features4net, 
                        st.session_state.wx_filt,
                        st.session_state.pt_graph, 
                        st.session_state.vertex_filtering_strategy )
                    
                    toremove = [ col for col in communities.columns if communities[col].value_counts().min() < st.session_state.min_community_size ]
                    if toremove:
                        st.warning(f"Removing community detection methods with n.members lower than {st.session_state.min_community_size} -> **{', '.join(toremove)}**")
                        communities.drop(columns=toremove, inplace=True, errors="ignore")
                    

                    st.session_state["info_graph_fs"] = ( curr_g, communities, PlotlyGraphVizManager.get_pos( curr_g ) ) 
                    st.session_state["p_threshold_gviz"] = st.session_state.pt_graph

                    st.session_state["vweights_on_graph"] = vweights_on_graph = list(set(st.session_state.df_weights.columns.tolist()) - { GraphField.FEATURE_OMIC.value })
                    _df_communities = pd.merge( communities.copy(), st.session_state.df_weights[ vweights_on_graph ], left_index=True, right_index=True )

                    adj = np.abs( mlu.gt.adjacency( curr_g, weight=curr_g.ep["WholeData"] ).toarray() )
                    _df_communities.insert( 0,"f_index", range(0, _df_communities.shape[0]))
                    st.session_state["community_info"] = {
                        cd_method: gfs.InfoCommunityDetectionMethod( 
                            adj_matrix = adj, 
                            df_communities = _df_communities, 
                            cd_method = cd_method, 
                            col_feature_index = "f_index", 
                            relevance_col = "avg_vscore" ) 
                        for cd_method in _df_communities.columns.tolist()
                        
                    }

                    st.write(f"Computing centralities...")
                    centralities, _ = compute_centralities( curr_g ) 
                    st.session_state["_centralities"] = pd.concat( [centralities, communities ], axis=1 )

                    _df_communities.drop(columns=["f_index"], inplace=True)
                    _df_communities["omic"] = list(curr_g.vp[ GraphField.FEATURE_OMIC.value ])
                    st.session_state["_df_communities"] = _df_communities        
                    
                    status.update(label="Computation completed!", state="complete", expanded=True) 


    if "info_graph_fs" in st.session_state:
        current_feature_graph, df_communities, vpos = st.session_state["info_graph_fs"]
        
        vweights_on_graph = st.session_state["vweights_on_graph"]

        _df_communities = st.session_state._df_communities
        my_omics = _df_communities.omic.unique().tolist()   
        cd_algorithms = df_communities.columns.tolist()

        avail_omics = sorted( set( _df_communities["omic"].tolist() ) ) 
        avail_cd_methods = df_communities.columns.tolist()

        if len(avail_cd_methods) == 0:
            st.warning("No community detection method available! Please, select a different pvalue threshold or increase the minimum community size.")
            st.stop()
        
        with st.container(border=True):
            st.header("Community Exploration Panel")
            st.markdown(f"The graph has {current_feature_graph.num_vertices()} vertices and {current_feature_graph.num_edges()} edges")
            cd_method = st.pills(
                "Select community detection algorithm", 
                options = avail_cd_methods, 
                key="algo_comm_2viz", default=avail_cd_methods[0] ) or avail_cd_methods[0]
            show_net_figure = st.toggle("Enable network visualization", value=False)

            with st.expander("View Network centralities", expanded=False):
                cd_values = [ val for val in sorted(st.session_state._centralities[ cd_method ].unique().tolist())]
                cd_tab_names = [ f"{cd_method} #{val}" for val in cd_values ]
                tab_centr, *comm_tabs = st.tabs( ["Centralities", *cd_tab_names] )

                with tab_centr:
                    st.dataframe(st.session_state._centralities, use_container_width=True)

                for tab, c_val in zip( comm_tabs, cd_values ):
                    with tab:
                        st.markdown(f"### Features in community {c_val} - {cd_method}")
                        comm_features_df = st.session_state._centralities[ st.session_state._centralities[ cd_method ] == c_val ]
                        st.dataframe( comm_features_df, use_container_width=True )

                
            if False:
                with st.expander("View Dataframe", expanded=False):
                    tab_df, *tabs_boh = st.tabs(["Dataframe", *cd_algorithms ]) #"Boh", "Mamt"])
                    with tab_df:
                        st.dataframe( _df_communities, use_container_width = True )
                        st.dataframe( _df_communities.describe(), use_container_width=  True )

                    corr_matrix = mlu.gt.adjacency( current_feature_graph, weight=current_feature_graph.ep["WholeData"]  ).toarray() #corrs["spearman"].to_numpy() 
                    feature_map = { f: i for i, f in enumerate( current_feature_graph.vp[GraphField.FEATURE_NAME.value] ) }

                    for tab, cm_algo in zip( tabs_boh, cd_algorithms ):
                        with tab:
                            c_info = st.session_state.community_info[cm_algo]
                            comm_info_data = list(map( lambda x: x.asdict(), c_info.community_info ))
                            comm_info_df = pd.DataFrame( comm_info_data ).set_index("c_id")
                            st.dataframe( comm_info_df, use_container_width=True )


            if show_net_figure:
                if st.session_state.p_threshold_gviz and st.session_state.p_threshold_gviz < 1.0:
                    filtering_params = FilteringParameters(
                        min_corr_threshold=0., pvalue_threshold = 0., padj_flag=True, chosen_omics=my_omics, edge_type=EdgeType.ALL_EDGES)
                    try:
                        metric2viz = st.pills("Select importance score", options = vweights_on_graph, key="metric2viz", default="avg_vscore" ) or "avg_vscore"
                        g_fig = get_graph_figure( 
                            current_feature_graph, "", 
                            _df_communities, filtering_params, cd_method, metric2viz, vpos=vpos )
                        st.plotly_chart(g_fig, use_container_width=True)
                    except ValueError as ve:
                        st.warning(f"Graph visualization is not available: {ve}")
                else: 
                    
                    corr_matrix = mlu.gt.adjacency( current_feature_graph, weight=current_feature_graph.ep["WholeData"]  ).toarray() #corrs["spearman"].to_numpy()
                    
                    corr_matrix += np.diag(np.ones(corr_matrix.shape[0]) )
                    feature_order = list( current_feature_graph.vp[ GraphField.FEATURE_NAME.value ] )
                    linkage = sch.linkage(corr_matrix, method='ward')  # Metodo Ward per clustering
                    dendro = sch.dendrogram(linkage, no_plot=True)  # Ottieni l'ordine senza visualizzare il dendrogramma
                    order = dendro['leaves']
                    sorted_corr_matrix = corr_matrix[np.ix_(order, order)]
                    new_cols = [feature_order[i] for i in order]
                    
                    sorted_corr_matrix = pd.DataFrame(sorted_corr_matrix, index=new_cols, columns=new_cols)

                    fig =  go.Figure( go.Heatmap(
                        z=(sorted_corr_matrix), 
                        x=sorted_corr_matrix.index.tolist(), 
                        y=sorted_corr_matrix.columns.tolist(), 
                            colorscale='rdbu', zmin=-1, zmax=1, reversescale=True), 
                    )
                    figsize = 700
                    fig.update_layout(
                        yaxis_scaleanchor="x",     
                        xaxis=dict(
                            tickangle=45,     # ruota le etichette sull'asse X
                            tickfont=dict(size=9),
                            side='bottom'
                        ),
                        yaxis=dict(
                            tickfont=dict(size=9),
                            autorange='reversed'  # per far partire la matrice dall’alto a sinistra
                        ),
                        width=figsize, height=figsize,
                        margin=dict(l=100, r=100, t=100, b=100),)
                    st.plotly_chart( fig, use_container_width=True )

            grouped_df = _df_communities.groupby([cd_method, "omic"]).size().reset_index(name="count")
            sizes = grouped_df.groupby(cd_method)["count"].sum().reset_index(name="total_count")
            st.markdown(
f"""
### Distribution of features per community - {cd_method}
* Number of communities: { _df_communities[cd_method].nunique() }
* Larger and small communities vary from { sizes['total_count'].min()} to { sizes['total_count'].max() } features
* Median community size: { sizes['total_count'].median() }
* Average community size: { np.round( sizes['total_count'].mean(), 2) }
""")
            fig = px.bar( 
                grouped_df, 
                x = cd_method, 
                y = "count", 
                color = "omic", 
                labels = {cd_method: "Community ID", "count": "Number of features"},
                color_discrete_map=st.session_state[ "my_palette" ],
                category_orders={"omic": avail_omics},
                barmode = "stack" )

            st.plotly_chart( fig, use_container_width = True )


        with st.container(border=True):
            st.header(f"Net-based Feature Selection Panel")
            relevance_modes = (
                ("Score-based", "Use univariate feature importance scores to guide the search"),
                ("SHAP-based", "Optimize feature relevance using SHAP values from a classification model"),
            )
            SCORE_BASED, SHAP_BASED = [ mode[0] for mode in relevance_modes ]
            relevance_modes = dict( relevance_modes )

            chosen_algo_mode = st.radio(
                "Relevance evaluation mode", horizontal=True,
                options = list(relevance_modes.keys()),
                captions = list(relevance_modes.values()),
                index = 0, key="GA_algo_mode",
            )
            default_ub_ngen, default_ub_popsize = (1500, 100) #if chosen_algo_mode == SCORE_BASED else (200, 30)

            with st.form("graph-based_fs", border=False):
                with st.container(border=True):
                    st.subheader(f"{chosen_algo_mode} parameters")
                    if chosen_algo_mode == SCORE_BASED:
                        st.pills("Select importance score", options = vweights_on_graph, key = "GA_wv", default="avg_vscore", help=HelperGeneticAlgorithm.IMPORTANCE_METHOD.value )
                    else:
                        st.pills("Underlying model", key="GA_ml_algo",  
                                options = [ mlu.LearningAlgorithm.LOGISTIC_REGRESSION, mlu.LearningAlgorithm.RANDOM_FOREST, mlu.LearningAlgorithm.XGBOOST ],
                                default = mlu.LearningAlgorithm.LOGISTIC_REGRESSION, 
                                selection_mode="single", help=HelperGeneticAlgorithm.HYBRID_GUIDE.value )
                        st.number_input("Number of folds for CV-based relevance estimation", min_value = 2, max_value=10, value=3, key="GA_hybrid_nfolds" )
                     
                     
                cl, cr = st.columns(2)

                with cl:
                    num_cpus = cpu_count() 
                    default_ncpus = min( _df_communities[cd_method].nunique(), num_cpus -1 )
                    st.number_input("Set the number of parallel processes", min_value = 1, max_value=num_cpus-1, value = default_ncpus, key = "GA_nprocs")

                    st.number_input("Set the population size", min_value = 2, value = default_ub_popsize, key = "GA_popsize", help=HelperGeneticAlgorithm.POPULATION_SIZE.value )
                    st.pills(
                        "Select community detection algorithm", 
                        options = df_communities.columns.tolist(), key="id_algo_comms", 
                        default = st.session_state.algo_comm_2viz, help = HelperGeneticAlgorithm.COMM_METHOD.value ) 


                with cr: 
                    st.number_input("Reproducibility seed", min_value = 0, value = 42, max_value=10_000, key = "GA_seed", help=HelperGeneticAlgorithm.RANDOM_SEED.value)
                    st.number_input("Upper bound in the number of generations", min_value = 1, value = default_ub_ngen, key = "GA_ub_ngen", help=HelperGeneticAlgorithm.UB_N_GENERATIONS.value )
            
                    st.toggle("Enable seeded population", key="seeded_population", value=False, help=HelperGeneticAlgorithm.SEEDED_POPULATION.value)

                if st.form_submit_button("Run GA"):
                    cd_algorithm = st.session_state.id_algo_comms
                    if chosen_algo_mode == SCORE_BASED:
                        chosen_score = st.session_state.GA_wv 
                        if chosen_score:
                            chosen_score = GraphField.VERTEX_WEIGHT.value.format( st.session_state.GA_wv )
                    else:
                        chosen_score = st.session_state.GA_ml_algo 

                    error_message = []

                    if cd_algorithm is None:
                        error_message.append( "**community detection algorithm**")
                    if chosen_score is None: 
                        error_message.append( "**importance score**")
                        

                    if not error_message: #all([cd_algorithm, chosen_score]): #chosen_score is not None:
                        with st.status(f"Running Genetic Algorithm using {chosen_score}", expanded = True) as status:
                            st.write(f"Initializing feature selection problem parameterized with {cd_algorithm}....")
                            
                            ga_feature_graph = current_feature_graph
                            
                            is_default_relevance_mode = (chosen_algo_mode == SCORE_BASED)

                            opt_problem = gfs.NetworkMOEAProblem(
                                g = ga_feature_graph, 
                                df_communities = df_communities, 
                                cd_method=cd_algorithm, 
                                score=(chosen_score if is_default_relevance_mode else None), 
                                num_procs = st.session_state.GA_nprocs)
                            
                            if is_default_relevance_mode:
                                opt_problem.set_relevance_function(func="score", score = chosen_score )
                            else:
                                func = "shap"
                                opt_problem.set_relevance_function(
                                    func = func, 
                                    clf = mlu.LearningAlgorithm.get_algorithm( chosen_score ), 
                                    metric = chosen_score.value, 
                                    ncv = st.session_state.GA_hybrid_nfolds,
                                    tr_set = st.session_state[ mlu.ML_SessionState.DISCOVERY_SET.value ]
                                )

                            ga_params = dict(
                                cd_algorithm = cd_algorithm, 
                                vertex_score = chosen_score, 
                                p_threshold = 0.05, 
                                feature_pool = st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ]
                            ) 
                            st.write(f"Executing the genetic algorithm")
                            t = time()
                            ga_results = gfs.GraphBasedFeatureSelection.run_optimization_algorithm( 
                                opt_problem,
                                pop_size=st.session_state.GA_popsize,
                                max_n_gen=st.session_state.GA_ub_ngen, 
                                seed=st.session_state.GA_seed, 
                                seeded_pop=st.session_state.seeded_population,
                                ostream = st.write ) 
                            st.write(f"Time elapsed: {time() - t:.2f} seconds")
                            st.write(f"Post-processing time")
                            t = time()

                            obj = gfs.MOEAHistory( 
                                ga_results, 
                                st.session_state[mlu.ML_SessionState.FEATURE_GRAPH.value], 
                                st.session_state[mlu.ML_SessionState.FEATURE_GROUPS.value]) 

                            st.session_state[ "GA_result" ] = dict( 
                                ga_output = ga_results, 
                                ga_params = ga_params , 
                                moea_data = obj
                            ) 

                            st.write(f"Time elapsed: {time() - t:.2f} seconds")

                            pool = obj.get_feature_pool( obj.n_generations )
                            st.session_state[ "info_GA_curr" ] = (
                                obj.n_generations,
                                obj.get_population( obj.n_generations ),
                                pool.feature_list,
                            )
                            
                        
                            #     ##qua dentro calcoliamo dataframe di tutte le soluzioni esplorate :) 


                            status.update(label="Computation completed!", state="complete", expanded=False) 
                    else:
                        error_message = f"Please select: {', '.join(error_message)}"
                        st.error(f"{error_message.capitalize()}!")


    if "output_GA_archive" in st.session_state:
        with open( st.session_state.output_GA_archive, "rb" ) as fp:
            st.download_button(
                "Download GA results", 
                data = fp,
                file_name = "results_run_GA.zip", 
                mime = "application/zip" )


    if "GA_result" in st.session_state:


        obj = st.session_state["GA_result"]["moea_data"]
        
        df_summary = obj.df_evolution
        st.dataframe( df_summary, use_container_width=True )

        fig = obj.multiomics_convergence_plot( st.session_state.my_palette ) 
        st.plotly_chart( fig, use_container_width=True )


        st.plotly_chart( obj.complexity_reduction_plots(), use_container_width=True )

        st.plotly_chart( obj.heatmap_evolution_plot(), use_container_width=True )


        with st.form("explore_generation"):
            st.slider("Choose a generation", key="ga_it", step=1, min_value = 1, max_value = obj.n_generations, value=obj.n_generations )
    
            if st.form_submit_button("Explore generation"):
                ### get_ga_info 
                pool = obj.get_feature_pool( st.session_state.ga_it - 1 )
                st.session_state[ "info_GA_curr" ] = (
                    st.session_state.ga_it - 1,
                    obj.get_population( st.session_state.ga_it ),
                    pool.feature_list,
                )

        if "info_GA_curr" in st.session_state:
            explore_ga_generation(  )


st.title(f"Network-based Feature Selection")

HelpMessage.net_fs()

mlu.ml_guard(st.session_state)

network_fsel_UI() 
