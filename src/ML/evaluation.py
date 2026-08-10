import streamlit as st 
import kaleido
import pandas as pd, numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from typing import Dict, List, Tuple, Optional, Literal
from collections import defaultdict
from itertools import chain, combinations

from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
import tools.ml_utils as mlu
from tools.utils import build_tsv_archive, HelpMessage, generate_hex_palette, write_figure_collection
from tools.helpers import HelperEvaluation

from tools.ml_managers import (
    FeatureEvaluationManager, MongoFEManager, get_tset_performances, build_roc_plot)


import time 


def retrieve_featuresets( 
        list_fset_id: List[str], 
        include_covariates: List[str] = None, 
        cov_mapping: Dict[str, List[str]] = None ) -> Dict[str, List[str]]:
    
    df_fset = st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ]

    
    ret_lit__features_only = {
        fset_id: list( df_fset.loc[ fset_id ].flist ) 
            for fset_id in list_fset_id
    }
    ret_lit = ret_lit__features_only.copy()

    if include_covariates:
        cov_combinations = chain.from_iterable(
            list(combinations( include_covariates, i )) for i in range(1, len(include_covariates) + 1)
        )
    
        for covs_comb in cov_combinations:
            selected_encoded = chain.from_iterable([ cov_mapping[cov] for cov in covs_comb ])
            included_covs = "-".join(sorted(covs_comb))
            ret_lit.update({
                f"{fset_id}_adj.{included_covs}": flist + list( selected_encoded )
                    for fset_id, flist in ret_lit__features_only.items()
            })
            

    return ret_lit


def compute_mean_performance( 
        df_performance: pd.DataFrame, 
        groupby_cols: str = None, 
        agg_functions: str = None ) -> pd.DataFrame:
    """
    Computes the mean performance for each algorithm and feature set combination.
    """
    groupby = ["algo_id", "fset_id"]

    if groupby_cols is not None:
        _groupby_cols = [col for col in groupby_cols.split(",") if col in df_performance.columns ]
        groupby.extend(_groupby_cols)

    if "sampling" in df_performance.columns:
        groupby.append("sampling")

    agg_by = agg_functions if agg_functions is not None else ["mean", "std"]

    return df_performance.groupby(by=groupby).agg( agg_by ) #.reset_index()


def get_cv_object( cv_method: str, n_splits: int, n_repeats: int ):
    if cv_method == "Leave One Out":
        cv_split = LeaveOneOut() 
    else:
        cv_split = RepeatedStratifiedKFold( 
                n_splits = n_splits, 
                n_repeats = n_repeats,
                random_state = 42 ) 
    return cv_split


def make_zip_results( feature_sets: pd.DataFrame, feature_importances: Optional[ pd.DataFrame ], cv_performance: pd.DataFrame, tset_results: List ) -> Dict[str, pd.DataFrame]:
    data_to_zip = dict() 
    try:
        df_ci, df_raw = zip(*tset_results)
        data_to_zip = dict(
            performance_test_set = pd.concat( df_raw ),
            confidence_intervals = pd.concat( df_ci )
        )
    except ValueError:
        st.warning("No test set available for evaluation")
    finally:
        data_to_zip.update(dict(
            performance_table = cv_performance,
            feature_sets = feature_sets #st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ]
        ))
        data_to_zip[ "mean_performance_cv" ] = compute_mean_performance( cv_performance, groupby_cols="n_features" )

        if feature_importances is not None:
            data_to_zip[ "feature_importances" ] = feature_importances
    return build_tsv_archive( **data_to_zip )

def wrap_all_feature_importances( fset_impos: Dict[ str, Dict[str, Tuple[float, float]]] ) -> Dict[str, pd.DataFrame]:
    keys = list( fset_impos.keys() )
    df_list = list()    

    for fset_id in keys:
        if not fset_impos[fset_id]:
            continue

        df_list.append(
            pd.concat( 
                [ fset_impos[ fset_id ][ clf ] for clf in fset_impos[ fset_id ].keys() ], 
                axis=1, 
                keys = fset_impos[fset_id].keys() ) ) 

    if df_list:
        return pd.concat( df_list, axis=1 ) #, keys=keys )


def viz_evaluation( key_id: str, session_keys: Tuple[str] ):
    session_cv_key, session_tset_key, session_zip_out, session_f_importances = session_keys

    if all([ (k in st.session_state) for k in session_keys ]):
        if session_zip_out in st.session_state:
            with st.container(border=True):
                st.markdown(f"## Download results")
                with open( st.session_state[session_zip_out], "rb" ) as fp:
                    st.download_button(
                        "Download ML evaluation results", 
                        data = fp,
                        key = f"download_eval_{key_id}",
                        file_name = f"evaluation_results_{key_id}.zip", 
                        mime = "application/zip" )


        if session_f_importances in st.session_state:
            with st.container(border=True):
                f_importances = st.session_state[ session_f_importances ]
                k_importances = list( f_importances.keys() )

                settings_cols = st.columns(2)
                with settings_cols[0]:
                    st.number_input("Number of features to show", min_value=1, max_value=100, value=10, key=f"{key_id}_n_features_to_show")
                    err_bars = st.toggle("Show error bars", key=f"{key_id}_show_error_bars", value=True, help="Show error bars for feature importances")
                impo_tabs = st.tabs( k_importances )

                for tab, fset_id in zip( impo_tabs, k_importances ):
                    with tab:
                        with st.expander("Show data", expanded=False):
                            try:
                                concazz = pd.concat(
                                    [ f_importances[ fset_id ][ clf ] for clf in f_importances[ fset_id ].keys() ],
                                    axis=1,
                                    keys = f_importances[ fset_id ].keys()
                                )
                                
                                st.dataframe(concazz, use_container_width=True)
                            except ValueError:
                                st.error(f"Feature importances for {fset_id} are not available for the current classifiers")
                                continue

                        st.header(f"Feature importances for **{fset_id}**")
                        fset_importances = f_importances[ fset_id ]
                        
                        fig = make_subplots(
                            rows=1, cols=len(fset_importances), 
                            subplot_titles=list(fset_importances.keys()), 
                            shared_yaxes=True,) 

                        for i, clf in enumerate( fset_importances.keys() ):
                            imp_df = fset_importances[ clf ]\
                                .sort_values(by="mean", ascending=False)\
                                .reset_index()\
                                .iloc[:st.session_state[f"{key_id}_n_features_to_show"]]
                            error_bars = dict(type='data', array=imp_df['std'], visible=True) if err_bars else None 
                            ub = imp_df["mean"].max() + imp_df["std"].max() 
                            fig.add_trace(
                                go.Bar(
                                    orientation="h",
                                    y = imp_df["index"], 
                                    x = imp_df["mean"], 
                                    name = clf, 
                                    hoverinfo="y+name",
                                    error_x=error_bars,  #dict(type='data', array=imp_df['std'], visible=True),
                                ), row=1, col=i+1
                            )
                            
                            fig.update_xaxes(title_text="Importance", range=[0,ub],  row=1, col=i+1)
                        fig.update_yaxes(title_text="Features",  
                                tickmode='array',
                                tickfont=dict(size=10), 
                                automargin=True)
                        st.plotly_chart( 
                            
                            fig, 
                            use_container_width= True, key=f"{key_id}_fset_importances_{fset_id}" )


        with st.container(border=True):
            st.header("Performance metrics on CV")
            tab_tab, plot_tab, radar_tab = st.tabs(["Performance tables", "Box Plots", "Radar Plots"])
            cv_performance = st.session_state[ session_cv_key ].reset_index()
            
            loocv = "loocv__accuracy" in cv_performance.columns
            
            with tab_tab:
                df_to_viz = cv_performance if loocv else compute_mean_performance( cv_performance, groupby_cols="n_features" )
                st.dataframe( df_to_viz, use_container_width = True )
            with plot_tab:
                cv_performance_to_plot = cv_performance#.reset_index()#.groupby(by=["algo_id", "fset_id", "sampling"]).agg("mean").set_index(["algo_id", "fset_id", "sampling"])
                cols_metrics = cv_performance_to_plot.columns.tolist()
                facet = dict(facet_row="sampling") if "sampling" in cv_performance and cv_performance["sampling"].nunique() > 1 else dict()
                if not loocv: #"roc_auc" in cols_metrics:
                    cv_mode = "K-Fold CV"
                    start_index = cols_metrics.index("roc_auc")
                else:
                    cv_mode = "LOOCV"
                    start_index = cols_metrics.index("loocv__roc_auc")
                    

                col_m, col_group = st.columns([0.3, 0.7])
                with col_group:
                    avail_metrics = cv_performance_to_plot.columns.tolist()[start_index:]
                    st.pills("Metric", options=avail_metrics, default=avail_metrics[0], selection_mode="single", key=f"{key_id}_metric_col_cv")
                with col_m:
                    boxplot_grouper = mlu.algo_fset_grouper(f"{key_id}_groupboxes_cv")
                    groupby, colorby = ("algo_id", "fset_id") if boxplot_grouper == "algorithm" else ("fset_id", "algo_id")

                match cv_mode:
                    case "K-Fold CV":
                        fig = px.box(cv_performance_to_plot, y = groupby, x = st.session_state[f"{key_id}_metric_col_cv"],  orientation="h", color=colorby,  **facet)
                        ## prev viz
                    case "LOOCV": 
                        fig = px.bar(cv_performance_to_plot, x = groupby, y = st.session_state[f"{key_id}_metric_col_cv"], color=colorby, barmode="group", **facet)
                        
                lowb = -1 if st.session_state[f"{key_id}_metric_col_cv"] == "mcc" else 0.
                fig.update_yaxes(range=[lowb,1])
                st.plotly_chart(fig, key=f"{key_id}_plot", use_container_width = True )

                        
            with radar_tab:
                if df_to_viz.index.nlevels > 1:
                    flatt_df = df_to_viz\
                                .drop(columns="std", level=1).droplevel(1, axis=1)\
                                .reset_index()\
                                .drop(columns=["p_threshold", "n_features", "sampling"], errors="ignore")
                    roc_auc_colname = "roc_auc"
                    
                else:
                    flatt_df = df_to_viz.drop(columns=["loocv__p_threshold"])
                    roc_auc_colname = "loocv__roc_auc"
                    
                #.set_index(["algo_id", "fset_id"])
                
                my_cols = flatt_df.columns.tolist()
                my_cols = ["algo_id", "fset_id"] + my_cols[my_cols.index(roc_auc_colname) : ]

                l, r = st.columns([0.3, 0.7])
                with l:
                    boxplot_grouper = mlu.algo_fset_grouper(f"{key_id}_polars")
                    groupby, trace_key = ("algo_id", "fset_id") if boxplot_grouper == "algorithm" else ("fset_id", "algo_id")
                with r: 
                    my_cols = flatt_df.columns.tolist()
                    my_cols = my_cols[my_cols.index(roc_auc_colname) : ]
                    selected_cols = st.pills("Show metrics", options=my_cols, default=my_cols, selection_mode="multi", key=f"{key_id}_show_metrics_pills")


                if len(selected_cols) < 3:
                    st.warning("Select at least 3 metrics to build the radar plot")
                else:
                    my_cols = ["algo_id", "fset_id"] + selected_cols
                    grps = flatt_df[groupby].unique().tolist()

                    fig = make_subplots(
                        rows=1, cols=len(grps), 
                        specs=[[{'type': 'polar'}]*len(grps)], 
                        subplot_titles=grps )
                    
                    traces_id = sorted( flatt_df[trace_key].unique() )
                    n_colors = len( traces_id )
                    color_map = px.colors.sample_colorscale("Plasma", [ i/(n_colors) for i in range(n_colors)])
                    color_map = { name: color for name, color in zip( traces_id, color_map ) }

                    for i, grp in enumerate( grps ):
                        for row in flatt_df[ flatt_df[groupby] == grp ].itertuples():
                            trace_name = getattr(row, trace_key)

                            fig.add_trace( 
                                row=1, col=i+1,
                                trace = go.Scatterpolar( 
                                    r = row[3:], theta = my_cols[2:], 
                                    fill="tonext", 
                                    name=trace_name, 
                                    line=dict(color=color_map[trace_name]),  # <- force consistent color
                                    showlegend=(i == 0), legendgroup=trace_name ),

                            )

                    fig.update_layout( height=600, width=300*len(grps), title_text="Performance metrics radar plots", title_y=1. )
                    st.plotly_chart( fig, use_container_width = True, key=f"{grp}_polar_plot" )
                    
    
        if (test_set_names := list( st.session_state[ session_tset_key ].keys() )):
            res_records = list( st.session_state[ session_tset_key ].values() )
            avail_metrics = res_records[0].raw_performances.columns.tolist() 
            index_begin = avail_metrics.index("auc_score")
            cols = st.columns([0.2, 0.8])
            with cols[0]:
                chosen_group_cov = mlu.algo_fset_grouper(f"groupby_stuff")
            with cols[1]:
                chosen_metric = st.selectbox("Metric", options=avail_metrics[index_begin:]) #, key=f"{key_id}_the_target_metric")
            

            table_set = {
                test_set: st.session_state[ session_tset_key ][ test_set ].ci_performances
                    for test_set in test_set_names
            }
            table_names, table_dfs = zip( *table_set.items() )
            gesucristo = pd.concat( table_dfs, keys=table_names, axis=1 )
            st.dataframe( gesucristo, use_container_width=True, key=f"{key_id}_all_testsets_performances" )


            if True:
                if False:
                    table_set = {
                        test_set: st.session_state[ session_tset_key ][ test_set ].ci_performances
                            for test_set in test_set_names
                    }
                    table_names, table_dfs = zip( *table_set.items() )
                    gesucristo = pd.concat( table_dfs, keys=table_names, axis=1 )
                    st.dataframe( gesucristo, use_container_width=True, key=f"{key_id}_all_testsets_performances" )


                    with st.container(border=True):
                        st.header("Area Under the Curve (AUC) Panel")     
                        visualization_box__roc_auc__multiple_tsets( res_records, key_id = f"{key_id}_multiple_tsets_auc_panel" )


                if True:
                    tabs_te_sets = st.tabs( test_set_names )
                    for tab, test_set in zip( tabs_te_sets, test_set_names ):

                        with tab:
                            visualization_boxplots__fucking_experimental(
                                res = res_records, 
                                test_set_id=test_set, 
                                group_cov=chosen_group_cov,
                                metric=chosen_metric,
                                key_id=f"{test_set}_boxplots_stuff"
                            )


                            res_record = st.session_state[ session_tset_key ][ test_set ]

                            with st.container(border=True):
                                st.header("Area Under the Curve (AUC) Panel")     
                                visualization_box__roc_auc( res_record, key_id = f"{key_id}_{test_set}" )           


@st.fragment 
def visualization_box__ensembles():
    col_algo_id, col_nf_column = "algo_id", "n_features"

    with st.expander("Original table", expanded=False):
        st.dataframe(st.session_state.get( mlu.ML_SessionState.PERFORMANCE_RECORD_ENSEMBLES.value ))

    main_df = st.session_state.get( mlu.ML_SessionState.PERFORMANCE_RECORD_ENSEMBLES.value ).drop(columns=["fset_id"])
    metric_list = main_df.columns.tolist()[ main_df.columns.tolist().index(col_nf_column) + 1 :] 
    metric_list = [ m for m in metric_list if m != "p_threshold" ]
    chosen_metrics = st.pills("Choose reference metrics", options = metric_list, default=metric_list, selection_mode="multi",  key="metric_col_ens_eval")


    if chosen_metrics:
        metrics_to_drop = set(metric_list).difference(chosen_metrics)

        summary_df = main_df.drop(columns=["algo_id", "p_threshold", *metrics_to_drop]).groupby(by=["ensemble_id", "n_features"]).agg(["mean", "median", "min", "max"])
        with st.expander(label="Summary table", expanded=False):
            st.dataframe(summary_df, use_container_width=True)
        

        for ens_id, subdf in summary_df.groupby("ensemble_id"):
            st.markdown(f"#### Ensemble **{ens_id}** evaluation")

            fig = go.Figure() 
            x_num_features = subdf.index.get_level_values(level=1)

            for col in chosen_metrics:
                col_name = (col, "mean")
                fig.add_trace( go.Scatter( x = x_num_features, y = subdf[col_name], name = col ) ) 


            fig.update_yaxes(range=[0, 1], title_text = "Avg. performance" )
            fig.update_xaxes(range=[x_num_features.min()-0.5, x_num_features.max()+0.5])
            fig.update_xaxes(title_text = "Number of features")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"Choose **at least** one metric to continue")


@st.fragment
def visualization_boxplots__fucking_experimental( 
        res: List[ mlu.TestSetPerformance ], 
        test_set_id: str, 
        group_cov: Literal["algorithm", "feature set"], 
        metric: str,
        key_id: str ):
    if res:    
        df_list = list() 
        for x in res:
            if x.test_set_id == test_set_id:
                df = x.raw_performances.copy() 
                df.insert(0, "test_id", x.test_set_id)
                df_list.append( df )
                
        df = pd.concat( df_list )
        
        x_axis, group_axis = ("fset_id", "algo_id") if group_cov == "algorithm" else ("algo_id", "fset_id")

        item_subplots = df[ x_axis ].unique()
        my_palette = generate_hex_palette( len(item_subplots) )
        color_map = { item: color for item, color in zip(item_subplots, my_palette) }

        target_metric = metric 
        min_pm_value = df[ target_metric ].min()

        sorted_feature_sets = df.groupby(group_axis)[target_metric].mean().sort_values(ascending=False).index.tolist()
        
        ### begin smanettamento ### seems to work -- ALL CREDITS TO CHATGPT
        fig = px.box(
            df, 
            x = target_metric, y = x_axis, orientation="h",
            color = x_axis, 
            facet_row = group_axis,
            title = f"Boxplot of {target_metric} grouped by {group_axis}", 
            points="all",
            color_discrete_map=color_map, 
            category_orders={ group_axis : sorted_feature_sets}
        )
        fig.update_xaxes(range=[min_pm_value-0.05,1])
        # hide y tick labels (feature sets)

        # 2) remove repeated y-axis title (e.g., "fset_id")
        fig.for_each_yaxis(lambda ax: ax.update(title_text=None))

        # 3) try to capture existing facet-row annotations for the group_axis (algo names)
        group_prefix = f"{group_axis}="                     # e.g. "algo_id="
        existing_anns = [a for a in fig.layout.annotations if group_prefix in str(a.text)]

        left_annotations = []
        if existing_anns:
            # use those annotations' y positions and text to create left-side labels
            for a in existing_anns:
                val = str(a.text).split("=", 1)[1]          # keep only the value after '='
                y = a.y                                    # paper coords (should be present)
                left_annotations.append(dict(
                    x=-0.02, y=y, xref='paper', yref='paper',
                    text=val, showarrow=False,
                    xanchor='right', align='right',
                    font=dict(size=12)
                ))
            # remove the original facet annotations for group_axis (they are usually on the right)
            fig.layout.annotations = tuple(a for a in fig.layout.annotations if group_prefix not in str(a.text))

        else:
            # fallback: compute vertical centers from yaxis domains and use unique group values
            # useful if the automatic annotations were not present
            y_centers = []
            for name, obj in fig.layout.items():
                if name.startswith("yaxis") and getattr(obj, "domain", None):
                    dom = obj.domain
                    center = (dom[0] + dom[1]) / 2.0
                    y_centers.append(center)
            # unique centers top->bottom
            y_centers = sorted(set(y_centers), reverse=True)

            # determine the ordered unique groups (try to preserve px ordering)
            ordered_groups = list(dict.fromkeys(df[group_axis].tolist()))
            # pair them (if different length, take min)
            for val, y in zip(ordered_groups, y_centers):
                left_annotations.append(dict(
                    x=-0.02, y=y, xref='paper', yref='paper',
                    text=str(val), showarrow=False,
                    xanchor='right', align='right',
                    font=dict(size=12)
                ))

        # 4) add our left-side algorithm annotations
        for ann in left_annotations:
            fig.add_annotation(ann)

        # 5) hide all y tick labels so feature set names disappear
        fig.update_yaxes(showticklabels=False)

        # 6) make room on the left so the new labels are visible
        fig.update_layout(margin=dict(l=220))


        st.plotly_chart(fig, key = f"{key_id} boxplot", use_container_width = True )

        if False:
            for clf, subdf in df.groupby( group_axis ):
                _subdf = subdf.copy()
                mean_auc_per_feature_set = _subdf.groupby(x_axis)[ target_metric ].mean()  # st.session_state.target_metric ].mean()

                # Ordina i Feature Set in base al valore medio di AUC
                sorted_feature_sets = mean_auc_per_feature_set.sort_values(ascending=False).index

                fig = px.box( 
                    _subdf, 
                    x = target_metric, y = x_axis, orientation="h",
                    color=x_axis, 
                    facet_col="test_id",
                    title=clf, 
                    points="all",
                    color_discrete_map = color_map, 
                    category_orders={ x_axis : sorted_feature_sets}
                )
                fig.update_xaxes(range=[min_pm_value-0.1,1])
                # Aggiungi una linea orizzontale a y = 0.5


                st.plotly_chart(fig, key = f"{key_id}_{clf}_boxplot", use_container_width = True )


def visualization_box__roc_auc__multiple_tsets( res: List[mlu.TestSetPerformance], key_id: str ):
    widget_id = hash((r.test_set_id for r in res))
    
    colz = st.columns(3)
    with colz[0]:
        auc_grouper = mlu.algo_fset_grouper(f"{key_id}_groupdata_wrt_{widget_id}")
    with colz[1]:
        auc_thr = st.slider("Threshold for Area Under the Curve", min_value=0., max_value=1., value=0.5, key=f"{key_id}_auc_thr_{widget_id}")
    with colz[2]:
        pr_thr = st.slider("Threshold for Precision-Recall Curve", min_value=0., max_value=1., value=0.5, key=f"{key_id}_pr_thr_{widget_id}")

    match auc_grouper: #st.session_state[ f"{key_id}_groupdata_wrt_{tset_id}"]:
        case "algorithm":
            my_lambda = lambda rec: (rec.algo_id, rec.fset_id)
        case _:
            my_lambda = lambda rec: (rec.fset_id, rec.algo_id)


    map_tset_to_data = { x.name: x for x in st.session_state[ mlu.ML_SessionState.VALIDATION_SETS.value ] } 
    map_records = { r.test_set_id: r for r in res }
    data_collection = defaultdict( lambda: defaultdict( dict ) )
    key_set = set()

    for record_set in res:  #st.session_state[ mlu.ML_SessionState.PERFORMANCE_RECORD.value ]:
        for record in record_set.probs_record.values():
            group, key = my_lambda( record )

            # data_collection[ record_set.test_set_id ][ group ][ key ] = record.y_probs
            data_collection[group][record_set.test_set_id][ key ] = record.y_probs
            key_set.add( key )


    #     data_collection[ group ][ key ] = record.y_probs

    homemade_palette = {
        key: color for key, color in zip( sorted(key_set), generate_hex_palette( len( key_set ) ) )
    }


    for group, multiple_tset_data in data_collection.items():
        with st.container(border=True):
            st.markdown(f"### Group: **{group}**")

            for tset_name, its_data in multiple_tset_data.items():
                st.markdown(f"#### Test Set: **{tset_name}**")
                test_data = map_tset_to_data[ tset_name ]

                for i, (key, values_dict) in enumerate( its_data.items() ):
                    try:
                        my_fig = build_roc_plot( key, test_data.target, values_dict, auc_thr, pr_thr, color_traces=homemade_palette )
                    except ValueError as ve:
                        st.error(f"Cannot build ROC plot for **{key}**. Bug in investigation 🙄...  {ve}")
                        continue
                    else:
                        st.plotly_chart( my_fig, key=f"{key_id}_{key}_rocplot", use_container_width = True )


@st.fragment
def visualization_box__roc_auc( res: mlu.TestSetPerformance, key_id: str ):
    tset_id = res.test_set_id
    current_figures = dict() 
    
    if True:
        colz = st.columns(3)
        with colz[0]:
            auc_grouper = mlu.algo_fset_grouper(f"{key_id}_groupdata_wrt_{tset_id}")
        with colz[1]:
            auc_thr = st.slider("Threshold for Area Under the Curve", min_value=0., max_value=1., value=0.5, key=f"{key_id}_auc_thr_{tset_id}")
        with colz[2]:
            pr_thr = st.slider("Threshold for Precision-Recall Curve", min_value=0., max_value=1., value=0.5, key=f"{key_id}_pr_thr_{tset_id}")
    else:
        auc_grouper = "algorithm"
        auc_thr, pr_thr = 0.5, 0.5

    match auc_grouper: #st.session_state[ f"{key_id}_groupdata_wrt_{tset_id}"]:
        case "algorithm":
            my_lambda = lambda rec: (rec.algo_id, rec.fset_id)
        case _:
            my_lambda = lambda rec: (rec.fset_id, rec.algo_id)

    test_data = [ x for x in st.session_state[ mlu.ML_SessionState.VALIDATION_SETS.value] if x.name == res.test_set_id  ].pop()
    data_collection = defaultdict( dict )


    key_set = set()

    for record in res.probs_record.values():  #st.session_state[ mlu.ML_SessionState.PERFORMANCE_RECORD.value ]:
        group, key = my_lambda( record )

        data_collection[ group ][ key ] = record.y_probs
        key_set.add( key )

    homemade_palette = {
        key: color for key, color in zip( sorted(key_set), generate_hex_palette( len( key_set ) ) )
    }
    
    
    for i, (key, values_dict) in enumerate( data_collection.items() ):
        try:
            current_figures[key] = build_roc_plot( key, test_data.target, values_dict, auc_thr, pr_thr, color_traces=homemade_palette )
        except ValueError as ve:
            st.error(f"Cannot build ROC plot for **{key}**. Bug in investigation 🙄...  {ve}")
            continue

    try:
        l, r = st.columns(2)
        with l:
            st.radio("Save as", options=["PDF", "SVG"], index=0, key="crazy_fmt", horizontal=True)

        with r:
            filename = write_figure_collection( f"{key_id}_rocplots.zip", current_figures, prefix=f"{key_id}_rocplot", fmt = st.session_state.crazy_fmt.lower() )
            st.download_button(
                "Download ROC plots as ZIP file",
                data = open( filename, "rb" ),
                file_name = f"roc_plots_{key_id}.zip",
                mime = "application/zip") 
    except ValueError as ve:
        st.error(f"Cannot save ROC plots. Bug in investigation 🙄...  {ve}")

    for key, fig in current_figures.items():
        st.plotly_chart( fig, key=f"{key_id}_{key}_rocplot", use_container_width = True )


@st.fragment 
def evaluation_params_panel():
    with st.container(border = True):
        st.header("Evaluation parameters")
        avail_algos = [ x for x in mlu.LearningAlgorithm ] #if x != mlu.LearningAlgorithm.LINEAR_SVM ]
        st.pills("Choose learning algorithms", key="clf_to_eval", options = avail_algos, selection_mode = "multi", help=HelperEvaluation.CLF_TO_EVALUATE.value  ) 
        
        cols = st.columns([0.2, 0.4, 0.4])
        with cols[0]:
            st.pills(f"Cross-validation setting", options = ["K-Fold CV", "Leave One Out"], key = "cv_method", default = "K-Fold CV", help=HelperEvaluation.CV_TECHNIQUE.value )
        with cols[1]:
            st.number_input("Choose number of repetitions", min_value=1, max_value=100, value=1, key="n_reps", help=HelperEvaluation.N_REPS.value )
        with cols[2]:
            st.slider("Choose number of folds", 2, 10, 10, key = "k_value", help=HelperEvaluation.N_FOLDS.value )

        no_samples__discovery = st.session_state.get( mlu.ML_SessionState.DISCOVERY_SET.value ).features.shape[0]
        no_samples__fold = no_samples__discovery // st.session_state.k_value
        st.info(f"No. samples in Discovery Set: {no_samples__discovery}. "\
                f"No. samples for training process: {no_samples__fold*(st.session_state.k_value-1)}. " \
                f"No. samples for testing process: {no_samples__fold}." ) 
        sampling_techniques = ["Nope"] + [ sampling_tech.value for sampling_tech in mlu.ImbalancedTechnique ]
        st.pills("Adjust imbalanced datasets", options = sampling_techniques, key = "imbalance_adjust", default = sampling_techniques[0], disabled=False,  help=HelperEvaluation.IMBALANCED_TECHNIQUE.value )


def evaluation_UI():

    with st.container(border=True):
        st.header("Feature sets")
        df_fset = st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ] 
        st.dataframe( df_fset, use_container_width = True )


    evaluation_params_panel()


    main_tabnames = [ "Feature Set Evaluation", "Feature Ensemble Evaluation", "GA Population Evaluation"] #, "Graph-based Feature Evaluation"]
    fset_eval_tab, ensemble_eval_tab, ga_eval_tab = st.tabs( main_tabnames )

    
    with fset_eval_tab:
        with st.form("form_ml_eval"):
            st.multiselect(f"Choose feature sets to evaluate", key="fset_to_eval", options = df_fset.index.tolist() )

            covariate_mappings = st.session_state.cov_mapping #training_and_test_data[0].cov_encoding.categorical_mappings

            st.pills("Include covariates", 
                    key="covariates_for_learning", 
                    options =  list( covariate_mappings.keys() ), 
                    selection_mode="multi")


            if st.form_submit_button("Start evaluation"):
                if not st.session_state.clf_to_eval:
                    st.warning("No learning algorithm selected")
                    st.stop()
                if not st.session_state.fset_to_eval:
                    st.warning("No feature set selected")
                    st.stop()

                with st.status(f"Starting model evaluation...", expanded=True) as status:
                    training_and_test_data = st.session_state[ mlu.ML_SessionState.ALL_DATA.value ]#[:2]
                    k = st.session_state.k_value 

                    cv_split = get_cv_object(
                        st.session_state.cv_method, 
                        n_splits = k, 
                        n_repeats = st.session_state.n_reps
                    )

                    match cv_split:
                        case LeaveOneOut():
                            cv_mode_str, flag_k_fold = st.session_state.cv_method, False
                            sampling_str = ""
                        case _:
                            cv_mode_str, flag_k_fold = f"{st.session_state.n_reps} x {k}-Fold CV", True
                            sampling_str = f"using **{st.session_state.imbalance_adjust}** technique" if st.session_state.imbalance_adjust != "Disabled" else ""
                            

                    built_feature_sets = retrieve_featuresets( 
                        st.session_state.fset_to_eval, 
                        st.session_state.covariates_for_learning, 
                        covariate_mappings )

                    mongo_manager = st.session_state.get( mlu.ML_SessionState.EVAL_MANAGER.value )
                    performance_table, feature_importances, tset_perf_dict, df_list_outfile = mongo_manager.models_evaluation_wrapper(
                        task_info = st.session_state.get( mlu.ML_SessionState.SELECTED_TASK.value ), 
                        input_data = training_and_test_data,
                        fset_collection = built_feature_sets,
                        algos_id = st.session_state.clf_to_eval, 
                        cv_splits = cv_split, #my_scores,
                        flag_db=MongoFEManager.MongoFlag.FEATURE_SET.value,
                        sampler = mlu.get_imbalanced_sampler( st.session_state.imbalance_adjust )
                    )

                    st.session_state[ "perf_tables_cv"] = performance_table  ## TODO: salvarlo in sessione da qualche parte 
                    st.session_state[ "f_importances" ] = feature_importances
                    st.session_state[ "perf_tsets" ] = tset_perf_dict


                    st.write(f"Zipping up the results")
                    

                    st.session_state[ "zip_eval_data" ] = make_zip_results( 
                        feature_sets = st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ], 
                        feature_importances = wrap_all_feature_importances( feature_importances ) if feature_importances else None,
                        cv_performance = performance_table,
                        tset_results = df_list_outfile, 
                        ) 
                   

                    status.update(label="Computation completed!", state="complete", expanded=False) 

        viz_evaluation("ev", ( "perf_tables_cv", "perf_tsets", "zip_eval_data", "f_importances" ) )
        
    with ensemble_eval_tab:
        feature_manager = st.session_state[ mlu.ML_SessionState.FEATURES.value ]
        ensnames = list( feature_manager.ensembles )  #list( map( lambda ens: ens.name, ens_fsets) )
        
        if ensnames:
            with st.form("run_eval_ensemble"):
                st.pills(
                    "Select ensemble set to evaluate", 
                    options=ensnames, 
                    key="ens2run", selection_mode="multi"
                ) #st.selectbox("scegli", options=ensnames, key="ens2run")

                if st.form_submit_button("run"):
                    if not st.session_state.ens2run:
                        st.warning("No ensemble feature set selected for evaluation")
                        st.stop()

                    with st.status(f"Starting model evaluation...") as status:
                        st.write(f"Training models in {st.session_state.cv_method} mode")
                        st.session_state[ "current_ensemble_fs" ] = list( st.session_state.ens2run ) #feature_manager.get_ensemble( chosen_ens )
                        perfs_df_list = list()

                        mongo_manager = st.session_state[ mlu.ML_SessionState.EVAL_MANAGER.value ]
                        
                        for ens_id in st.session_state.ens2run:
                            chosen_ens = feature_manager.get_ensemble( ens_id )
                            st.write(f"Evaluating ensemble feature set **{chosen_ens.name}**")
                            

                            perfs_df = mongo_manager.ensemble_evaluation(
                                st.session_state[ mlu.ML_SessionState.ALL_DATA.value ],#[:2],
                                chosen_ens, 
                                st.session_state.clf_to_eval, #my_scores, 
                                k_folds=st.session_state.k_value,
                                n_repeats=st.session_state.n_reps
                            )
                            perfs_df = compute_mean_performance( perfs_df, groupby_cols="n_features", agg_functions="mean" ).reset_index() #.reset_index()  #.groupby(by=["algo_id", "fset_id", "n_features"]).agg("mean").reset_index()
                            perfs_df.insert(0, "ensemble_id", ens_id)
                            perfs_df_list.append( (chosen_ens.to_dataframe(), perfs_df) )
                        else:
                            st.write(f"Zipping up the results")
                            ens_fsets_df, perfs_df = [ pd.concat(df_list, axis=0) for df_list in zip(*perfs_df_list)]
                            st.session_state[ mlu.ML_SessionState.PERFORMANCE_RECORD_ENSEMBLES.value ] = perfs_df
                            
                            st.session_state[ "ensemble_zip_filename" ] = build_tsv_archive(
                                ensemble_performance_cv = perfs_df,
                                ensemble_features = ens_fsets_df )
                            status.update(label="Computation completed!", state="complete", expanded=False) 

               
        else:
            st.warning("No ensemble feature sets available. You need to create an ensemble feature set first!")


        if "current_ensemble_fs" in st.session_state:
            with st.container( border = True):
                st.subheader("Evaluation ensemble feature set")
                visualization_box__ensembles()

                if "ensemble_zip_filename" in st.session_state:
                    with st.container(border=False):
                        st.subheader("Download results")
                        with open( st.session_state.ensemble_zip_filename, "rb" ) as fp:
                            st.download_button(
                                "Download ML ensemble evaluation results", 
                                data = fp,
                                file_name = "ensemble_evaluation_results.zip", 
                                mime = "application/zip" )

                with st.container(border=True):
                    st.subheader("Select the feature sets to be saved")
                    df_edited = list()
                    for ensemble_name in st.session_state[ "current_ensemble_fs" ]:
                        my_ensemble_df = st.session_state[ mlu.ML_SessionState.FEATURES.value ]\
                            .get_ensemble( ensemble_name )\
                            .to_dataframe().drop(columns=["metadata"]).reset_index()
                        my_ensemble_df["uid"] = f"{ensemble_name}_" + my_ensemble_df["composition"]  #ensemble_name
                        my_ensemble_df.set_index("uid", inplace=True)
                        blocked_columns = my_ensemble_df.columns.tolist()   
                        my_ensemble_df.insert(0, "to_select", False)
                        df_edited.append( my_ensemble_df )

                    df_edited = st.data_editor(
                        pd.concat( df_edited, axis=0 ), 
                        disabled=blocked_columns, 
                        use_container_width = True, 
                        key = "ensemble_fset" )
                

                    if st.button("Save selected feature sets"):
                        df_selected = df_edited[ df_edited.to_select ]
                        n_selected = df_selected.shape[0]
                        if n_selected == 0:
                            st.warning("No feature set selected")
                            st.stop()
                        
                        for _, row in df_selected.iterrows():
                            st.session_state[ mlu.ML_SessionState.FEATURES.value ].add_feature_set(
                                fset = row.flist, metadata = f"ensemble_{ensemble_name}" ) 
                        else:
                            st.success(f"{n_selected} feature sets saved!")
                            mlu.update_feature_dataframe()

    with ga_eval_tab:
        ga_gens = st.session_state[ mlu.ML_SessionState.FEATURES.value ].ga_generations
        if not ga_gens:
            st.warning("No GA generations available. You need to run a GA first!")
            st.stop()

        with st.form("run_eval_ga"):
            st.pills(
                "Select the population to evaluate", options=list( ga_gens.keys() ), key="population_to_eval")
            
            if st.form_submit_button("Evaluate population"):
                if ( id_chosen_pop := st.session_state.population_to_eval ) is None:
                    st.warning("No population selected!!")
                    st.stop() 
                
                
                with st.status(f"Starting model evaluation...") as status:
                    st.write(f"Training models in {st.session_state.cv_method} mode")
                    chosen_pop = ga_gens[ id_chosen_pop ]
                    training_and_test_data = st.session_state[ mlu.ML_SessionState.ALL_DATA.value ]
                    
                    cv_split = get_cv_object(
                        st.session_state.cv_method, 
                        n_splits = st.session_state.k_value, 
                        n_repeats = st.session_state.n_reps )

                    st.session_state[ "eval_population_ga" ] = { f"#{i}_{id_chosen_pop}": sol for i, sol in enumerate( chosen_pop.solutions, 1 ) }

                    if "old" and False:
                        performance_table, probs_data, feature_importances = FeatureEvaluationManager.models_evaluation_wrapper(
                            training_and_test_data,
                            st.session_state[ "eval_population_ga" ],
                            st.session_state.clf_to_eval, 
                            cv_split, #my_scores
                        )

                        st.session_state[ "perf_tables_ga" ] = performance_table
                        tset_perf_dict = get_tset_performances(
                            task = st.session_state.get( mlu.ML_SessionState.SELECTED_TASK.value ), 
                            probs_data=probs_data, 
                            test_set_list=st.session_state[ mlu.ML_SessionState.VALIDATION_SETS.value ] )
                        df_list_outfile = [ cp.get_outfiles() for cp in tset_perf_dict.values() ]
                        st.session_state[ "perf_tsets_ga" ] = tset_perf_dict

                    mongo_manager = st.session_state[ mlu.ML_SessionState.EVAL_MANAGER.value ]
                    ga_eval_results = mongo_manager.models_evaluation_wrapper(
                        task_info = st.session_state.get( mlu.ML_SessionState.SELECTED_TASK.value ),
                        input_data = training_and_test_data,
                        fset_collection = st.session_state[ "eval_population_ga" ],
                        algos_id = st.session_state.clf_to_eval,
                        cv_splits = cv_split, #my_scores,
                        flag_db=MongoFEManager.MongoFlag.GA_GENERATION.value
                    )
                    performance_table, f_importances_ga, tset_perf_dict, df_list_outfile = ga_eval_results
                    st.session_state[ "perf_tables_ga" ] = performance_table
                    st.session_state[ "perf_tsets_ga" ] = tset_perf_dict
                    st.session_state[ "f_importances_ga" ] = f_importances_ga

                    st.session_state[ "zip_eval_ga" ] = make_zip_results(
                        feature_sets = st.session_state[ mlu.ML_SessionState.FEATURE_FOUND.value ], 
                        cv_performance = performance_table,
                        tset_results = df_list_outfile, 
                        feature_importances = None )

        if (ga_features := st.session_state.get("eval_population_ga") ):
            ga_features = st.session_state[ "eval_population_ga" ]
            fsubsets = list() 
            for fset_id, fset in ga_features.items():
                tokens = fset_id.split("_")
                int_id, metadata = tokens[0], "_".join(tokens[1:])
                fsubsets.append( mlu.FeatureSubset(
                    feature_list=fset, 
                    metadata=metadata,
                    id_count=int( int_id[1:]), #remove the leading '#' from the id
                    feature_graph=st.session_state[ mlu.ML_SessionState.FEATURE_GRAPH.value ],
                    feature_groups=st.session_state[ mlu.ML_SessionState.FEATURE_GROUPS.value ]
                ).to_dict() ) 


            st.write(pd.DataFrame( fsubsets ))

            viz_evaluation("ga", ( "perf_tables_ga", "perf_tsets_ga", "zip_eval_ga", "f_importances_ga" ) )


st.title(f"Models & Features Evaluation")

HelpMessage.model_training()


mlu.ml_guard(st.session_state)
evaluation_UI()

