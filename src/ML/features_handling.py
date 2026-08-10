import streamlit as st
from itertools import chain
import tools.ml_utils as mlu
import graph_tool.all as gt, networkx as nx
import pandas as pd, numpy as np
from itertools import chain, product
from typing import List, Dict
from collections import Counter
import plotly.express as px 
import plotly.graph_objects as go
from tools.gtools import GraphDistanceManager, OmicsGraphFilter, GraphField
from tools.ml_utils import ML_SessionState, ml_guard, update_feature_dataframe
from tools.utils import build_tsv_archive, HelpMessage

@st.cache_data
def get_nx_graph( fset: List[str], feature_space: Dict[str, int] ) -> nx.Graph:
    whole_g = st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ]
    curr_subgraph = OmicsGraphFilter.get_subgraph( whole_g, fset )
    curr_feature_space = { i: feature_space[ f_name ] for i, f_name in enumerate( list( curr_subgraph.vp[ GraphField.FEATURE_NAME.value ])) }
    new_adj = np.zeros( shape = (len( feature_space), len( feature_space)) )

    for i, j in zip( *gt.adjacency( curr_subgraph ).nonzero() ) :
        new_adj[ curr_feature_space[i], curr_feature_space[j] ] = 1.

    new_adj += new_adj.T
    return nx.from_numpy_array( new_adj )


def get_nx_collection( feature_pool: List[str], fset_collection: Dict[str, List[str]] ) -> Dict[str, nx.Graph]:
    feature_space = { feature: i for i, feature in enumerate( sorted( feature_pool ) ) }
    return {
        fset_id: get_nx_graph( fset_list, feature_space )
            for fset_id, fset_list in fset_collection.items()
    }


def get_features_from_hierarchical():
    if ( hc_ga := st.session_state.get("hc_ga") ):
        ga_pop_id, g_metric, df_hc, figures = hc_ga
        univocal_sets = set()


        st.header(f"Dendrogram for feature set similarity w.r.t. {g_metric}")
        for f in figures:
            st.image(f)


        with st.form("choose_nc"):
            cols = st.columns(2)
            with cols[0]:
                options = ["Visualize features", "Save featuress"]
                st.pills(f"Choose operation", options=options, default=options[0], key="see_or_act" )
            with cols[1]:
                st.slider(f"Upper bound in the number of feature sets:",  min_value=2, max_value=10, key="slider_nc")

            if st.form_submit_button("Perform operation"):
                communities = { f"c_{x}": subdf.index.tolist() for x, subdf in df_hc.groupby( str(st.session_state.slider_nc) ) }
                save_mode = st.session_state.see_or_act == options[-1] #st.session_state.see_or_act.startswith("Save")

                for c_id, c_list in communities.items():
                    feature_pool = sorted( set( chain.from_iterable([
                        ga_pop_id.solutions[i] for i in c_list
                    ])))
                    univocal_sets.add( tuple(feature_pool ) )

                st.write(f"Number of fsets: {len(univocal_sets)}")    
                current_fsets = {
                    f"{ga_pop_id.name}__hc_{i}": x for i, x in enumerate( univocal_sets )
                }
            
            
                universe = sorted( set( chain.from_iterable( current_fsets.values() )) )
                b_matrix = np.array([ 
                    [ 1 if f in fset else 0 for f in universe ] 
                        for fset in current_fsets.values()
                ])

                pl_colorscale = ["gray", "black"]
                heatmap = go.Heatmap(
                    z=b_matrix, 
                    x = universe, y = list( range( b_matrix.shape[0] ) )[::-1],
                    colorscale=pl_colorscale, showscale=False, xgap=1, ygap=1, colorbar_thickness=24)
                legend_heatmap = [
                    go.Scatter( x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=str(i))
                        for i, color in enumerate( pl_colorscale )
                ]
                fig = go.Figure( data=[heatmap, *legend_heatmap] ).update_layout( yaxis=dict(dtick=1) )
                st.plotly_chart( fig, use_container_width = True )
                
                if save_mode:
                    for f_name, fset in current_fsets.items():
                        st.session_state[ mlu.ML_SessionState.FEATURES.value ].add_feature_set(
                            fset = fset, 
                            metadata = f_name
                        )
                    st.write(f"Saved {len(current_fsets)} feature sets")
                    update_feature_dataframe()
                

def features_handling_UI():
    feature_found_df = st.session_state[mlu.ML_SessionState.FEATURE_FOUND.value ]
    initial_features = set(chain.from_iterable(feature_found_df[ feature_found_df.metadata == "user" ].flist.tolist()))
    st.dataframe( feature_found_df )

    feature_manager = st.session_state[ mlu.ML_SessionState.FEATURES.value ]

    tab_creation, tab_merge, tab_removal, tab_import, tab_download = st.tabs( [
        "Add feature set", 
        "Merge feature sets",
        "Remove feature set",
        "Import feature sets", 
        "Download feature sets"
        # "Edit feature set"
    ])


    with tab_creation:
        with st.form("manual_fs_creation"):
            st.text_area("Insert features separated by comma or '\\n'", key="manual_fs_list", height=100)
            st.text_input("Insert optional metadata", key="manual_fs_metadata", placeholder="", value="manual")
            if st.form_submit_button("Add feature set"):
                separator = "\n" if "\n" in st.session_state.manual_fs_list else ","
                fs_list = [ my_s for s in st.session_state.manual_fs_list.split( separator ) if ( my_s := s.strip() )]
                unavailable_features = set( fs_list ) - initial_features

                if len( fs_list ) == 0:
                    st.error("Plz insert some features!")
                elif len( unavailable_features ) > 0:
                    st.error(f"The following features are not available: {unavailable_features} --  you selected {fs_list} features")
                else:
                    fs_metadata = st.session_state.manual_fs_metadata.strip()
                    fs_metadata = "manual" if len( fs_metadata ) == 0 else fs_metadata
                    feature_manager.add_feature_set( fs_list, metadata = fs_metadata )
                    update_feature_dataframe()
                    st.write(f"Added feature set with {len(fs_list)} features")

    with tab_merge:
        with st.form("manual_fs_merge"):
            mergable_fsets = feature_found_df.index.tolist()
            st.multiselect(
                f"Select two or more feature set to merge", 
                options=mergable_fsets, 
                key="fs_to_merge")
            st.text_input("Insert optional metadata", key="merge_fs_metadata", placeholder="", value="merged")
            
            if st.form_submit_button("Merge selected feature set(s)"):
                if len( st.session_state.fs_to_merge  ) < 2:
                    st.error("Plz select at least two feature sets!")
                else:
                    merged_fset = set( chain.from_iterable([
                        feature_found_df.loc[ fs_name].flist for fs_name in st.session_state.fs_to_merge 
                    ]))
                    fs_metadata = st.session_state.merge_fs_metadata.strip()
                    fs_metadata = "merged" if len( fs_metadata ) == 0 else fs_metadata
                    feature_manager.add_feature_set( list(merged_fset), metadata = fs_metadata )
                    update_feature_dataframe()
                    st.write(f"Merged {len(st.session_state.fs_to_merge)} feature sets into one with {len(merged_fset)} features")


    with tab_removal:
        with st.form("manual_fs_removal"):
            erasable_fsets = feature_found_df[ feature_found_df.metadata != "user" ].index.tolist()
            st.multiselect(
                f"Select one or more feature set to remove", 
                options=erasable_fsets, 
                key="fs_to_remove")
            
            if st.form_submit_button("Remove selected feature set(s)"):
                if len( st.session_state.fs_to_remove  ) == 0:
                    st.error("Plz select at least one feature set!")
                else:
                    for fs_name in st.session_state.fs_to_remove :
                        if feature_manager.remove_feature_set( feature_found_df.loc[ fs_name].flist ) == fs_name:
                            st.success(f"Removed feature set {fs_name}")
                        else:
                            st.warning(f"Feature set {fs_name} not found")
                            
                    update_feature_dataframe()
                    
    with tab_import:
        st.file_uploader("Upload a CSV/TSV file with feature sets", type=["csv", "tsv"], key="csv_import")
        if st.session_state.csv_import is not None:
            check_cols = ["metadata", "flist"]
            try:
                sep = "\t" if st.session_state.csv_import.name.endswith(".tsv") else ","
                imported_df = pd.read_csv( st.session_state.csv_import, sep=sep, index_col=0, header=0 )
                imported_df.insert( 0, "to_import", True )

                if not all( [col in imported_df.columns for col in check_cols] ):
                    st.session_state["missing_cols"] = set( check_cols ) - set( imported_df.columns )   
                    raise ValueError(f"Columns {check_cols} not found in the file")

            except Exception as e:
                match e:
                    case pd.errors.ParserError:
                        st.error(f"Error while parsing the file: {e}")
                    case ValueError:
                        st.error(f"Some columns missing among {check_cols}")
            else:
                df_features = st.data_editor( imported_df )
                st.markdown( f"Number of feature sets to import: {len(df_features[ df_features.to_import ])}" )
                if st.button("Import selected feature sets"):
                    reduced_df = df_features[ (df_features.to_import) & (df_features.metadata != "user") ][ check_cols ]
                    st.dataframe(reduced_df) 
                    if reduced_df.shape[0] > 0:
                        for i, row in reduced_df.iterrows():
                            metadata = row.metadata 
                            try:
                                flist = eval( row.flist )
                            except SyntaxError:
                                flist = row.flist.split(",")

                            feature_manager.add_feature_set( flist, metadata = metadata )
                        update_feature_dataframe()
                        st.success(f"Added {reduced_df.shape[0]} feature sets")

    with tab_download:
        st.write("Download feature sets as CSV/TSV file")
        individual_fsets = st.session_state[mlu.ML_SessionState.FEATURE_FOUND.value]
        ens_fsets = list() 
        for ens_id, ens_obj in st.session_state[ mlu.ML_SessionState.FEATURES.value ].ensembles.items():
            df = ens_obj.to_dataframe().reset_index()
            df["uid"] = ens_id
            ens_fsets.append( df.set_index("uid") )
        ens_fsets = pd.concat( ens_fsets, axis=0 ) if len(ens_fsets) > 0 else None 
        sep = "\t" # st.selectbox("Choose separator", options=[",", "\t"], index=0, key="sep_download")
        my_fsets = dict(
            individual = individual_fsets, 
            ens_fsets = ens_fsets
        )
        my_fsets = { k: v for k, v in my_fsets.items() if v is not None }

        if st.button("Create zipped feature sets"):
            zippp_my_fsets = build_tsv_archive( **my_fsets ) 
            with open( zippp_my_fsets, "rb" ) as fp:
                st.download_button(
                    "Download feature sets", 
                    data = fp,
                    file_name = "your_feature_sets.zip", 
                    mime = "application/zip" )


    with st.container(border=True):
        tab_names = [ "Histogram", "Ensembles", "GA" ]
        tabs = st.tabs( tab_names )
        
        with tabs[0]:
            f_groups = st.session_state[ mlu.ML_SessionState.FEATURE_GROUPS.value ]
            f_selected = st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ]

            with st.form("histo_form"):
                initial_sets = set( f_groups.keys() )
                avail_fsets = [fset_id for fset_id in feature_found_df.index.tolist() if fset_id not in initial_sets ]
                st.multiselect("Choose feature sets to be considered", key="fset_histo", options=avail_fsets, default=avail_fsets)
                st.pills("Choose distance metrics", key="graph_distance_metrics", options=sorted(GraphDistanceManager.AVAIL_DISTANCES.keys()), selection_mode="multi")

                ## TODO: set threshold for removing features s.t. n_occ < threshold 


                if st.form_submit_button("Get histogram"):
                    if len( st.session_state.fset_histo ) > 1:
                        with st.status("Wait plz") as status:
                            st.write(f"Computing histogram")
                            counter = Counter( chain.from_iterable([
                                flist for flist in f_selected.loc[ st.session_state.fset_histo ].flist ]) ) 
                            my_features = list( counter.keys() )
                            fgraph = st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ]
                            st.session_state[ "feature_counts_hist" ] = counter 

                            feature_space = { feature: i for i, feature in enumerate(set(my_features))}
                            tot_nf = len( feature_space)
                            nx_collection = dict() 

                            st.write(f"Casting gt.Graphs into nx.Graphs :) ")
                            candidate_sets = { flist_id: list( f_selected.loc[flist_id].flist ) for flist_id in st.session_state.fset_histo }
                            nx_collection = get_nx_collection( my_features, candidate_sets )


                            st.write(f"Computing pairwise graph distances")
                            st.session_state["graph_distances"] = GraphDistanceManager.get_distances( nx_collection, st.session_state.graph_distance_metrics )
                            status.update(label="Computation completed!", state="complete", expanded=False) 

                                
                    else:
                        st.error("Choose 2+ feature sets plz")    
        

            feature_counts = st.session_state.get( "feature_counts_hist" )
            if feature_counts is not None:
                st.subheader("Features count histogram")
                max_v = max(feature_counts.values()) 
                st.slider("Lower bound (as frequency)", min_value = 0.0, max_value=1.0, step=0.01, key="lb_numfeatures", value=0.1)
                fig, df_counts = mlu.histogram_feature_counts( 
                        feature_counts, 
                        st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ], 
                        None,
                        st.session_state.my_palette,
                        min_freq= st.session_state.lb_numfeatures, # /  len( st.session_state.fset_histo ),
                        return_df=True )
                col_small, col_large = st.columns([0.3,0.7])
                with col_small:
                    df_counts.set_index("feature", inplace=True)
                    st.dataframe( df_counts, use_container_width = True )
                with col_large:
                    st.plotly_chart( fig, use_container_width = True )

            ### DISABLED FOR NOW: we have to replace the Clustergram with homemade dendrograms
            if False and ( my_beloved_distances := st.session_state.get("graph_distances") ):
                for a, df in my_beloved_distances.items():
                    st.subheader(a)

                    
                    try:
                        df = df / np.max(df)# * 100
                        fig_cluster = Clustergram(
                            data=df,
                            column_labels=list(df.columns.values),
                            row_labels=list(df.index),
                            height=666,
                            width=666, 
                            line_width=2, 
                            color_map= [
                                [0.0, '#636EFA'],
                                [0.25, '#AB63FA'],
                                [0.5, '#FFFFFF'],
                                [0.75, '#E763FA'],
                                [1.0, '#EF553B']
                            ]
                        )
                        st.plotly_chart( fig_cluster, use_container_width = True )
                    except Exception as e:
                        st.write(f"Exploded for {a} ==> {e}")
                        fig_heatmap = px.imshow(df, text_auto = True ) ## TODO - annotare i quadrati? 
                        st.plotly_chart( fig_heatmap, use_container_width = True )

            
        with tabs[1]:
            ens_fsets = list( st.session_state[ mlu.ML_SessionState.FEATURES.value ].ensembles.items() )

            if ens_fsets:
                for ens_id, ens_obj in ens_fsets:
                    st.write(ens_id)
                    st.dataframe( ens_obj.to_dataframe() )
            else:
                st.write("No ensembles available yet.")


        with tabs[2]:
            ga_gens = st.session_state[mlu.ML_SessionState.FEATURES.value].ga_generations

            with st.form("shrink_ga_gen"):
                st.pills("Select GA generation", options = list( ga_gens.keys() ), key="selected_ga_gen" )
                st.pills("Choose distance metrics", key="distance_metric_ga_gen", options=sorted(GraphDistanceManager.AVAIL_DISTANCES.keys()), selection_mode="single")

                if st.form_submit_button("go"):
                    with st.status("Computing stuff", expanded = True) as status:
                        chosen = ga_gens[ st.session_state.selected_ga_gen ]
                        chosen_metric = st.session_state.distance_metric_ga_gen 

                        st.write(f"Preparing the graph collection of {len(chosen.solutions)} graphs...")
                        nx_graphs = get_nx_collection( 
                            chosen.feature_pool, 
                            { f"{i}": fset for i, fset in enumerate( chosen.solutions ) })
                        
                        st.write(f"Computing {chosen_metric} distance metric")
                        dist_metric = GraphDistanceManager.get_distances( nx_graphs, [chosen_metric] ).pop(chosen_metric)

                        df_hc, figures = mlu.hierarchical_clustering(dist_metric.to_numpy() )

                        st.session_state["hc_ga"] = (chosen, chosen_metric, df_hc, figures)
                        

            get_features_from_hierarchical()
            

st.title(f"Feature Sets Management")


HelpMessage.handle_features()


ml_guard(st.session_state)

features_handling_UI()
