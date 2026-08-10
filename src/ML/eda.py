from itertools import combinations
import streamlit as st 
import pandas as pd, numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import scipy.stats as stats

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap 
from scipy.cluster.hierarchy import linkage, dendrogram

from tools.ml_utils import ML_SessionState, LabelledData, prepare_features, ml_guard, WilcoxonTest
from tools.utils import HelpMessage
from tools.gtools import OmicsGraphFilter, prepare_graph, compute_centralities
from tools.gviz import get_graph_figure
from tools.enums import GraphFilteringParametrization, FilteringParameters, EdgeType, GraphField
import graph_tool.all as gt 


FSET_EDA = "fset_eda"


AXES_NAMES = {
    "PCA": ["PC1", "PC2"],
    "KernelPCA": ["KernelPCA ax1", "KernelPCA ax2"],
    "t-SNE": ["t-SNE ax1", "t-SNE ax2"],
    "UMAP": ["UMAP ax1", "UMAP ax2"],
    "Isomap": ["Isomap ax1", "Isomap ax2"]
}


def boxplot_trends( feature_set ):
    bx_nmax_features = 30 
    make_boxplot_flag = len(feature_set) <= bx_nmax_features

    with st.container(border=True):
        st.subheader("Feature trends")
        
        task = st.session_state[ ML_SessionState.SELECTED_TASK.value ]
        feature_groups = st.session_state[ ML_SessionState.FEATURE_GROUPS.value ]
        my_feature_groups = {
            group_name: group_list 
                for group_name, feature_list in feature_groups.items()
                    if ( group_list := list( set(feature_list).intersection( feature_set ) ) )
        }
        omics_groups = sorted( my_feature_groups.keys() )
        m2o = { f: g for g, flist in my_feature_groups.items() for f in flist }

        l1, l2 = task.get_class(True), task.get_class(False)   #   posneg_dict.keys() 

        df_wx = st.session_state[ ML_SessionState.WX_TEST.value ].get_wilcoxon_results( feature_set )
        df_wx.insert(0, "omic", [ m2o[f] for f in df_wx.index.tolist()] )

        st.markdown(f"**Wilcoxon signed-rank test for {l1} vs {l2}**")
        st.dataframe( df_wx, use_container_width=True )


        if not make_boxplot_flag:
            st.warning("⚠️ Too many features selected, boxplots may be unreadable. Consider selecting a smaller feature set.")            
            return 

        tabs = st.tabs(["All features", *omics_groups])

        with tabs[0]:
            st.plotly_chart(
                st.session_state[ ML_SessionState.WX_TEST.value ].get_boxplot_trends( feature_set ), 
                key=f"boxplot_trends_all",
                use_container_width=True)
            
        for i, omic in enumerate(omics_groups):
            with tabs[i + 1]:
                st.plotly_chart(
                    st.session_state[ ML_SessionState.WX_TEST.value ].get_boxplot_trends( my_feature_groups[omic] ), 
                    key=f"boxplot_trends_{omic}",
                    use_container_width=True)


def boxplot_trends__old( feature_set ):
    with st.container(border=True):
        st.subheader("Feature trends")
        my_data = st.session_state[ ML_SessionState.DISCOVERY_SET.value ]#.select_features( sorted_feature_set )
        scaled_df = my_data.select_features( feature_set ).scale_data().features
        df = scaled_df.reset_index().melt(id_vars=["index"])
        
        task = st.session_state[ ML_SessionState.SELECTED_TASK.value ]
        samples = {
            task.get_class(True): my_data.metadata[ my_data.metadata[ task.target_cov ].isin(task.pos_class) ].index.tolist(),
            task.get_class(False): my_data.metadata[ my_data.metadata[ task.target_cov ].isin(task.neg_class) ].index.tolist()
        }

        feature_groups = st.session_state[ ML_SessionState.FEATURE_GROUPS.value ]
        my_feature_groups = {
            group_name: group_list 
                for group_name, feature_list in feature_groups.items()
                    if ( group_list := list( set(feature_list).intersection( feature_set ) ) )
        }
        omics_groups = sorted( my_feature_groups.keys() )

        tabs = st.tabs(["All features", *omics_groups])

        
        with tabs[0]:
            fig = go.Figure()
            posneg_dict = dict() 
            for sample_type, sample_list in samples.items():
                subdf = df[ df["index"].isin( sample_list ) ]
                posneg_dict[sample_type] = my_data.features[ feature_set ].loc[ sample_list ].copy()
                
                fig.add_trace( go.Box(
                    x = subdf["variable"], 
                    y = subdf["value"], 
                    name=sample_type, 
                    boxpoints='all',  # represent all points
                    jitter=0.3,
                    pointpos=-1.8,
                    text=subdf["index"],
                    hovertemplate="<b>%{text}</b><br>Value: %{y}<extra></extra>", 
                ))
            
            l1, l2 = posneg_dict.keys() 
            wx_data = list()

            for feature in feature_set:
                w_test, w_p = stats.ranksums(
                    posneg_dict[l1][feature].to_numpy(),
                    posneg_dict[l2][feature].to_numpy() )   
                wx_data.append( (feature, w_test, w_p) )
            
            lcol, rcol = st.columns(2)
            fig.update_layout( 
                title=f"Feature trends w.r.t. {task.target_cov} covariate", 
                xaxis_title="Feature",
                yaxis_title="Z-score",
                boxmode='group', showlegend=True )
            with lcol:
                st.plotly_chart(fig, use_container_width=True)
            with rcol:
                st.markdown(f"**Wilcoxon signed-rank test for {l1} vs {l2}**")
                st.dataframe( pd.DataFrame(
                    data = wx_data,
                    columns = ["feature", "w_stat", "p_value"]
                ).set_index("feature").sort_values("p_value", ascending=True), use_container_width=True )

        for i, omic in enumerate(omics_groups):
            with tabs[i + 1]:
                fig = go.Figure()
                for sample_type, sample_list in samples.items():
                    subdf = df[ df["index"].isin( sample_list ) & df["variable"].isin( my_feature_groups[omic] ) ]
                    
                    fig.add_trace( go.Box(
                        x = subdf["variable"], 
                        y = subdf["value"], 
                        name=sample_type, 
                        boxpoints='all',  # represent all points
                        jitter=0.3,
                        pointpos=-1.8,
                        text=subdf["index"],
                        hovertemplate="<b>%{text}</b><br>Value: %{y}<extra></extra>", 
                    ))

                fig.update_layout( 
                    title=f"Feature trends for {omic} w.r.t. {task.target_cov} covariate", 
                    xaxis_title="Feature",
                    yaxis_title="Z-score",
                    boxmode='group', showlegend=True )
                st.plotly_chart(fig, use_container_width=True, key=f"ftrends_{omic}")


# Wrapper per TSNE
class TSNEWrapper(TSNE):
    def transform(self, X):
        return self.fit_transform(X)


@st.fragment
def dimensionality_reduction_frame( feature_set ):
    with st.container(border=True):
        original_data = [ d.select_features( feature_set ) for d in st.session_state[ ML_SessionState.ALL_DATA.value] ]

        meta_fucking_data = original_data[0].metadata  #st.session_state.valid_input
        curr_params = st.session_state[ ML_SessionState.SELECTED_TASK.value ]

        dim_red_technique = st.pills(
            "Dimensionality reduction technique", 
            options = ["PCA", "KernelPCA", "t-SNE", "UMAP", "Isomap"], 
            key="dim_red_technique", default = "PCA" )
        cov_color = st.pills("Colour points wrt", options = meta_fucking_data.columns.tolist(), key="cov_pca_color", default = curr_params.target_cov ) #curr_params[0])

        try:
            fig, reduced_data = apply_dimensionality_reduction( original_data, feature_set, cov_color, mode = dim_red_technique )
            st.plotly_chart( fig, use_container_width=True )
        except KeyError as e:
            st.error(f"☠️ **{type(e)}**: cannot visualize **{dim_red_technique}** plot because the **{cov_color}** covariate has too many values (more than the colors in the palette LOL). ")
        except TypeError as ve:
            st.error(f"☠️ Unexpected error **{type(ve)}**: {ve}")
        else:
            if False and "pca" in dim_red_technique.lower():
                pca_cols = st.columns( len( original_data ) )
                for i, key in enumerate( reduced_data.keys() ):
                    with pca_cols[i]:
                        pc_scores = reduced_data[ key ][["x", "y"]]
                        df_scaled = original_data[ i ].features
                        df_shape = df_scaled.shape

                        df_corr = pd.DataFrame( 
                            data = np.corrcoef(df_scaled.T, pc_scores.T)[:df_shape[1], df_shape[1]:],
                            index=df_scaled.columns, 
                            columns=['PC1', 'PC2'])
                        
                        heatmap_args = dict( colorscale="rdbu", zmin = -1, zmax = 1, reversescale=True )
                        fig = go.Figure(
                            go.Heatmap( 
                                z = df_corr.to_numpy(),
                                x = df_corr.columns.tolist(), y = df_corr.index.tolist(),
                                **heatmap_args )
                        )
                        fig.update_traces(hoverinfo="x+y+z" )  # Mostra i nomi completi in hover
                        st.plotly_chart( fig, use_container_width=True )


@st.fragment
def parameterized_graph_visualization( feature_graph ) -> Dict[str, gt.Graph]:

    with st.container(border=True):
        st.header("Param box")
        pvalue = st.selectbox(f"Filter by adj. pvalue threshold", options=[None, 0.05, 0.01, 0.001], key="pvalue_fragment")

        omics_list = set(list( feature_graph.vp[ GraphField.FEATURE_OMIC.value ]))
        filtering_params = FilteringParameters(
            min_corr_threshold=0., pvalue_threshold = pvalue, padj_flag=True, chosen_omics=omics_list, edge_type=EdgeType.ALL_EDGES)
        gf_params = GraphFilteringParametrization( 
                        f"Graph Wholeee", "WholeData", filtering_params  )
        my_feature_graph = prepare_graph( feature_graph, gf_params, remove_zero_degree_nodes=False )
        sorted_features = list( my_feature_graph.vp[ GraphField.FEATURE_NAME.value ] )
        vertex_data, _ = compute_centralities( my_feature_graph )
        vertex_weights = st.session_state[ "df_weights" ]
        vertex_weights = vertex_weights[ vertex_weights.columns.tolist()[1:] ]

        vertex_data = pd.merge( left = vertex_data, right=vertex_weights, left_index=True, right_index=True)
        

    heatmap_args = dict( colorscale="rdbu", zmin = -1, zmax = 1, reversescale=True )
    
    with st.container(border=True):
        l, r = st.columns(2)
        with l:
            fig = go.Figure(
                    go.Heatmap( 
                        z = gt.adjacency(my_feature_graph, weight=my_feature_graph.ep.WholeData).toarray(), 
                        x = sorted_features, y = sorted_features,
                        **heatmap_args )
                )
            fig.update_traces(hoverinfo="x+y+z")  # Mostra i nomi completi in hover
            st.plotly_chart(
                fig, use_container_width = True
            )
        with r:
            st.dataframe(vertex_data, use_container_width = True )
    

    with st.container(border=True):
        classes = OmicsGraphFilter.get_strata( feature_graph ) #feature_graph.gp[ GraphField.STRAT_LAYERS.value ].split("$")

        for graph_key in classes:
            l, r = st.columns(2)
            gf_params = GraphFilteringParametrization( 
                f"Graph {graph_key}", graph_key, filtering_params  )
            specific_g = prepare_graph( feature_graph, gf_params, remove_zero_degree_nodes=False )
            vertex_data, _ = compute_centralities( specific_g )
            vertex_data = pd.merge( left = vertex_data, right=vertex_weights, left_index=True, right_index=True)
            #             gf_params.filtering_params, stat_vertex_color, stat_vertex_size, vpos=vpos) 
            with l:
                st.plotly_chart( go.Figure(
                    go.Heatmap( 
                        z = gt.adjacency(specific_g, weight=specific_g.ep[graph_key]).toarray(),
                        x = sorted_features,
                        y = sorted_features,
                        **heatmap_args )), key=f"heatmap_{graph_key}"
                )
            with r:
                st.dataframe( vertex_data, use_container_width = True  )

    
@st.cache_data ##TODO: uncomment this line
def get_projected_data( _data_list: List[ LabelledData ], dim_red: str, features: Tuple[str], random_state: int = 42 ) -> Tuple[ Tuple[ str, pd.DataFrame ]]:  #Dict[str, pd.DataFrame]: #np.array:

    dataset_objs = _data_list

    if False:
        keys = ML_SessionState.DISCOVERY_SET, ML_SessionState.TEST_DATA
        dataset_objs = [ st.session_state[ k.value ].select_features( features ) for k in keys ]
        dataset_names = ("Training", "Test")

    dim_red = dim_red.lower() 
    dr_args = dict( n_components = 2, random_state = random_state )
    pipeline_steps = [("scaler", StandardScaler())]
    
    match dim_red:
        case "pca":
            pipeline_steps.append(("pca", PCA( **dr_args )))
        case "kernelpca":
            pipeline_steps.append(("kernelpca", KernelPCA( kernel="rbf", **dr_args )))
        case "t-sne":
            min_n_samples = min([ data.features.shape[0] for data in dataset_objs ])
            perplexity = min( min_n_samples - 1, 5 )
            pipeline_steps.append(("t-sne", TSNEWrapper(**dr_args, perplexity=perplexity)))
        case "umap":
            pipeline_steps.append(("umap", umap.UMAP(**dr_args)))
        case "isomap":
            pipeline_steps.append(("isomap", Isomap(n_components=2)))
        case _:
            raise ValueError(f"Unknown dimensionality reduction technique: {dim_red}")


    if False:
        if dim_red == "pca":
            dim_red_obj = PCA( **dr_args )
            pipeline_steps.append((dim_red, dim_red_obj))
        else:
            if dim_red == "t-sne":
                min_n_samples = min([ data.features.shape[0] for data in dataset_objs ])
                perplexity = min( min_n_samples - 1, 5 )
                dim_red_obj = TSNEWrapper(**dr_args, perplexity=perplexity)
            elif dim_red == "umap":
                dim_red_obj = umap.UMAP(**dr_args)
            elif dim_red == "isomap":
                dim_red_obj = Isomap(n_components=2)

            if False and len( features ) > 50:
                dim_red_obj = Pipeline([
                    (dim_red, dim_red_obj )
                ])

    dim_red_obj = Pipeline(pipeline_steps).fit( dataset_objs[0].features )

    return tuple([
        (dataset.name, pd.DataFrame( data = dim_red_obj.transform( dataset.features ), columns = ["x", "y"], index = dataset.features.index ))
            for i, dataset in enumerate( dataset_objs )
    ])


    if False:
        return tuple([
            (dataset_names[i], pd.DataFrame( data = dim_red_obj.transform( dataset.features ), columns = ["x", "y"], index = dataset.features.index ))
                for i, dataset in enumerate( dataset_objs )
        ])


def apply_dimensionality_reduction( data_list: List[ LabelledData ], feature_set, color_cov: str, mode: str = "PCA" ) -> Tuple[ go.Figure, Dict[str, pd.DataFrame] ]:
    twodim_data = get_projected_data( data_list, mode, feature_set )
    if False:
        keys = ML_SessionState.DISCOVERY_SET, ML_SessionState.TEST_DATA
        metadata_objs = [ st.session_state[ k.value ].metadata for k in keys ]
    metadata_objs = [ d.metadata for d in st.session_state[ ML_SessionState.ALL_DATA.value] ]
    color_list = px.colors.qualitative.Dark24
    if color_cov:
        cov_values = [ set(metadata_obj[color_cov].unique()) for metadata_obj in metadata_objs ]
        cov_values = cov_values[0] if len(cov_values) == 1 else set(cov_values[0]).union( *cov_values[1:] )

    else:
        cov_values = { 0 }

    target_colors = { value: color for value, color in zip( cov_values, color_list )}
    dataset_names, dataset_matrices = zip( *twodim_data )
    
    fig = make_subplots( rows = 1, cols = len( twodim_data ), subplot_titles=dataset_names )
    axes_names = AXES_NAMES[mode]

    for index, df in enumerate( dataset_matrices ):
        df["target"] = metadata_objs[index][ color_cov ] if color_cov else 0

        for target, subdf in df.groupby("target"):
            trace_name = f"{color_cov}='{target}'"
            fig.add_trace(
                go.Scatter( 
                    x = subdf.x, 
                    y = subdf.y, 
                    customdata = subdf.index, 
                    hovertemplate = "<b>%{customdata}</b><br>(%{x},%{y})",
                    mode = "markers",  
                    marker=dict(color=target_colors[target]), 
                    legendgroup=trace_name, name=trace_name,
                    showlegend=bool(index == 0) ), 
                row = 1, col = 1 + index 
            )
            fig.update_xaxes(title_text=axes_names[0], row=1, col=1 + index)
            fig.update_yaxes(title_text=axes_names[1], row=1, col=1 + index)
            # fig

    figsize = 500
    fig.update_layout(
        yaxis_scaleanchor="x",  #  autosize=False, 
        title_text=f"{mode} coloured by {color_cov}", 
        width=figsize, height=figsize,
        margin=dict(l=100, r=100, t=100, b=100),)
    return fig, dict( zip( dataset_names, dataset_matrices ) )


def prepare_dataframe( lab_data: LabelledData ):
    df = lab_data.features.copy()
    df["target"] = lab_data.target
    df["sample"] = df.index.tolist()  #df_features.index.tolist()
    return pd.melt( 
        df, 
        id_vars=["target", "sample"], 
        value_vars = lab_data.features.columns.tolist(), 
        var_name="feature", 
        value_name="value" )


def interpret_pca(pca, feature_names, labels=None, top_pc = 3, top_n=3):
    """
    Generate an automated textual interpretation of PCA results.
    
    Parameters
    ----------
    pca : fitted sklearn PCA object
    feature_names : list of str
        Names of the original variables.
    labels : array-like, optional
        Sample labels (e.g., groups) for supervised hints.
    top_n : int
        Number of top contributing features to list per PC.
    """
    expl_var = pca.explained_variance_ratio_
    loadings = pca.components_.T
    
    report = []
    
    # Per-PC interpretation
    for pc_idx, var_ratio in enumerate(expl_var[:top_pc]):
        top_features_idx = np.argsort(np.abs(loadings[:, pc_idx]))[::-1][:top_n]
        top_features = [feature_names[i] for i in top_features_idx]
        top_loadings = loadings[top_features_idx, pc_idx]
        
        report.append(
            f"* PC{pc_idx+1} explains {var_ratio*100:.1f}% of the variance. "
            f"It is mainly driven by {', '.join([f'{f} ({l:+.2f})' for f, l in zip(top_features, top_loadings)])}."
        )
    
    # Global feature importance
    global_importance = np.sum(np.abs(loadings) * expl_var, axis=1)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Global Importance": global_importance
    }).sort_values("Global Importance", ascending=False)
    
    return "\n".join(report), importance_df


def principal_component_analysis( feature_set ):
    with st.container(border=True):
        st.header("Principal Component Analysis (PCA)")  

        data = st.session_state[ ML_SessionState.DISCOVERY_SET.value ]\
                .select_features( feature_set )\
                .scale_data()
        scaled_data = data.features
        
        color_list = px.colors.qualitative.Dark24
        cov_values = data.target.unique()

        target_colors = { value: color for value, color in zip( cov_values, color_list )}
        
        # Run PCA
        pca = PCA().fit(scaled_data)

        # Create a DataFrame with loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
            index=scaled_data.columns
        )
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        fig = make_subplots( specs=[[{"secondary_y": True}]] )
        x_axis = list(range(1, 1+len(pca.explained_variance_ratio_)))

        fig.add_trace( go.Scatter(
            x = x_axis, y = explained_var,
            mode = "lines+markers", name = "Explained Variance Ratio") )
        fig.add_trace( go.Scatter(
            x = x_axis, y = cumulative_var,
            mode = "lines+markers",
            name = "Cumulative Explained Variance"), secondary_y=True )
        
        fig.update_layout(
            title = "PCA Explained Variance",
            xaxis_title = "Principal Component",
            yaxis_title = "Explained Variance Ratio" ) 

        st.plotly_chart( fig, use_container_width = True )

        st.number_input("Number of principal components to consider", min_value=2, max_value=len(explained_var), value=2, key="n_pca_components")
        pc_pairs = list( combinations( range( st.session_state.n_pca_components ), 2 ) )

        
        scores = pd.DataFrame(
            pca.fit_transform(scaled_data) , 
            index = scaled_data.index, 
            columns = [f"PC{i+1}" for i in range(pca.components_.shape[0]) ]
        )

        st.toggle(f"Show biplot & loaders", key="biplot_enabler", value=True) ## Thx Ame 
        pc_tabs = st.tabs( ["Summary"] + [ f"PC{i+1} vs PC{j+1}" for i,j in pc_pairs ] )

        with pc_tabs[0]:
            interpretation, feature_importance = interpret_pca(pca, scaled_data.columns.tolist(), top_pc=st.session_state.n_pca_components, top_n=5)
            with st.container():
                st.subheader("PCA Interpretation")
                st.info(interpretation)
                st.markdown("**Global feature importance:**")
                st.dataframe( feature_importance.set_index("Feature"), use_container_width = True )


        for tab_idx, (pc1, pc2) in enumerate( pc_pairs, 1 ):
            with pc_tabs[ tab_idx ]:
                st.subheader( f"PC{pc1+1} vs PC{pc2+1}" )
        
                score_scatter = go.Scatter(
                    x=scores.iloc[:, pc1],
                    y=scores.iloc[:, pc2],
                    mode='markers',
                    name='Samples',            
                    marker=dict(size=8, color = [ target_colors[t] for t in data.target ],
                                colorscale="Viridis", opacity=0.7),
                    text=scores.index   ,
                    hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Group: %{text}"
                )
                

                # Arrows for loadings
                loading_arrows = []
                expl_var = pca.explained_variance_ratio_
                
                if st.session_state.biplot_enabler:

                    scaled_loadings = pca.components_.T  #* np.sqrt(expl_var) 
                    m_factor = 5
                    for i, var in enumerate(scaled_data.columns):
                        loading_arrows.append(go.Scatter(
                            x=[0, scaled_loadings[i, pc1]*m_factor],
                            y=[0, scaled_loadings[i, pc2]*m_factor],
                            mode="lines+text",
                            line=dict(color="blue", width=2),
                            text=[None, var],
                            textposition="top center",
                            name=var,
                            showlegend=False
                        ))

                    # Combine everything
                fig = go.Figure([score_scatter] + loading_arrows)
                # Layout
                fig.update_layout(
                    title="PCA Biplot (Samples colored by group)",
                    xaxis_title=f"PC1 ({expl_var[0]*100:.1f}% variance)",
                    yaxis_title=f"PC2 ({expl_var[1]*100:.1f}% variance)",
                    width=800,
                    height=600, 
                )

                st.plotly_chart( fig, use_container_width = False )

# requisiti: plotly, scipy, pandas, numpy
# pip install plotly scipy pandas numpy


def plotly_clustermap(df, row_colors=None, cmap="RdBu", center=0):
    # --- Compute linkages ---
    row_linkage = linkage(df.values, method="average", metric="euclidean")
    col_linkage = linkage(df.values.T, method="average", metric="euclidean")

    # --- Get row/column order ---
    row_dendro = dendrogram(row_linkage, no_plot=True)
    col_dendro = dendrogram(col_linkage, no_plot=True)
    row_order = row_dendro["leaves"]
    col_order = col_dendro["leaves"]

    df = df.iloc[row_order, col_order]
    
    # --- Create figure layout ---
    fig = make_subplots(
        rows=3, cols=3,
        column_widths=[0.8, 0.15, 0.05],
        row_heights=[0.25, 0.05, 0.7],
        specs=[
            [{"type": "xy"}, None, None ],
            [{"type": "heatmap"}, None, None],
            [{"type": "heatmap"}, {"type": "xy"}, {"type": "heatmap"}], 
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # --- Column dendrogram (top) ---
    for xs, ys in zip(col_dendro["icoord"], col_dendro["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none", showlegend=False
            ),
            row=1, col=1
        )

    # --- Row dendrogram (right of heatmap) ---
    for xs, ys in zip(row_dendro["icoord"], row_dendro["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=ys,
                y=[len(df) * 10 - xi for xi in xs],
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none", showlegend=False
            ),
            row=3, col=2
        )


    # --- Main heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale=cmap,
            reversescale=True,
            zmid=center,
            colorbar=dict(
                orientation="v",  # verticale
                x=1.02,           # appena a destra della heatmap
                y=0.5,            # centrata verticalmente
                len=0.6,          # più corta
                thickness=10, 
                title="Z-Score"
            ),
        ),
        row=3, col=1
    )

    # --- Target color strip (right side) ---
    if row_colors is not None:
        # --- Reorder row colors if provided ---
        row_colors = row_colors.reindex(df.index)
        color_vals = row_colors.astype(str).values
        unique_colors = list(pd.unique(color_vals))
        color_map = {v: i for i, v in enumerate(unique_colors)}
        z = np.array([color_map[c] for c in color_vals]).reshape(-1, 1)
        colorscale = [[i / (len(unique_colors) - 1 or 1), c] for i, c in enumerate(unique_colors)]

        fig.add_trace(
            go.Heatmap(
                z=z[::-1],
                y=df.index,
                x=["Target"],
                colorscale=colorscale,
                showscale=False,
                hoverinfo="text",
                text=color_vals
            ),
            row=3, col=3
        )

    # --- Column color strip (bottom) ---
    m2o = { 
        f: g 
            for g, flist in st.session_state[ ML_SessionState.FEATURE_GROUPS.value ].items() 
                for f in flist if f in df.columns.tolist() }
    
    ## add row heatmap for omics types
    column_colors = pd.Series( 
        [ st.session_state.my_palette[ m2o[f] ] for f in df.columns.tolist()], 
        index = df.columns.tolist() )
    column_colors = column_colors.reindex(df.columns)

    color_vals = column_colors.astype(str).values
    unique_colors = list(pd.unique(color_vals))
    color_map = {v: i for i, v in enumerate(unique_colors)}
    z = np.array([color_map[c] for c in color_vals]).reshape(1, -1)
    
    if len(unique_colors) == 1:
        #one-color palette (special case)
        colorscale = [[0, unique_colors[0]], [1, unique_colors[0]]]
    else:
        denom = len(unique_colors) - 1
        colorscale = [[i / denom, c] for i, c in enumerate(unique_colors)]

    
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=False,
            hoverinfo="text",
            text=color_vals
        ),
        row=2, col=1
    )
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    # --- Hide unneeded axes ---
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    fig.update_xaxes(showticklabels=False, row=3, col=2)
    fig.update_yaxes(showticklabels=False, row=3, col=2)
    fig.update_xaxes(showticklabels=False, row=3, col=3)
    fig.update_yaxes(showticklabels=False, row=3, col=3)

    # --- Format main heatmap axes ---
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9), row=3, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=3, col=1)

    # --- Layout tweaks ---
    fig.update_layout(
        width=900,
        height=800,
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=80)
    )

    return fig


#     """
#     df: dataframe (rows = items, cols = features)
#     row_colors: pd.Series indexed like df.index with color strings (e.g. 'blue'/'orange')
#     colorbar_len: fraction of figure width for the horizontal colorbar
#     colorbar_thickness: thickness in px
#     """
#     # 1) compute linkages and orderings
#     # Use correlation distance or euclidean as you prefer; here we use correlation on columns & rows.
#     # For numerical stability, convert to numpy.
#     # linkage for rows
#     # linkage for cols (transpose)

#     # reorder
#     # row labels and col labels ordered

#     # 2) create subplots: arrangement:
#     #    top dendrogram (cols)
#     #    left dendrogram (rows)
#     #    main heatmap center
#     #    optional narrow strip for row colors between left dendrogram and heatmap
#         [{"type": "xy", "colspan": 3}, None, None],   # top dendro spans 3 cols
#         [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],  # left dendro, heatmap, colorbar area

#     # 3) column dendrogram (top) -> plot as trace of lines
#     # We need to map leaf positions to col indices in the reordered df. dendrogram no_plot True returns coordinates in leaf order
#     # The x positions returned are in the leaf ordering 0..n-1, so they already match col_order ordering.

#     # 4) row dendrogram (left). We will flip coordinates so it goes vertically.
#         # swap x/y for left vertical dendro and invert axis later

#     # 5) main heatmap: add to row=2,col=2
#     # choose colorscale; Plotly has built-ins, but you can provide custom.

#     # 6) row color strip (narrow heatmap) if provided
#         # reorder row_colors to match df_r
#         # map distinct colors to numeric values for a categorical narrow heatmap
#         # build colorscale from the unique colors

#     # 7) layout adjustments: hide axes for dendrograms and strip, align flips
#     # Top dendrogram: flip y-axis so the dendrogram grows downward

#     # Left dendrogram: hide axis ticks

#     # Heatmap axes formatting


#     # If you want the column dendrogram to align with the heatmap, set its x-range to match heatmap columns
#     # The coordinates in the dendrogram output are in a 0..(ncols*10) space; scaling so top dendro x spans same width as heatmap:
#     # A simple hack: hide axes and rely on visual alignment via subplot sizing and spacings.


def exploratory_data_analysis_UI():
    df_fset = st.session_state[ ML_SessionState.FEATURE_FOUND.value ] #.set_index("fset_id")
    st.dataframe( df_fset, use_container_width = True )
    
    
    with st.form("mamt"):
        st.multiselect(f"Choose feature sets to evaluate", key="fset_to_eda", options = df_fset.index.tolist() )

        if st.form_submit_button("Visualize"):
            my_features = prepare_features( st.session_state.fset_to_eda, st.session_state )
            key = sorted( st.session_state.fset_to_eda )
            st.session_state[ FSET_EDA ] = ( key, my_features )

 
    if  ( eda_info := st.session_state.get( FSET_EDA) ):  #FSET_EDA in st.session_state:
        fset_id, feature_set = eda_info #st.session_state[ FSET_EDA ]


            # ## resize tickfonts on x and y axes


            # # Resize tick fonts on x and y axes


        with st.container(border=True):
            st.subheader(fset_id)

            with st.container(border=True):
                df = st.session_state[ ML_SessionState.DISCOVERY_SET.value ].features[ feature_set ]
                df = ( df - df.mean() ) / df.std()  # z-score normalization
                target = st.session_state[ ML_SessionState.DISCOVERY_SET.value ].target
                fig = plotly_clustermap( df, row_colors = target.map( {False: "blue", True: "orange"} ) )
                st.plotly_chart( fig, use_container_width = True )
        

            boxplot_trends( feature_set )


            principal_component_analysis( feature_set)


            dimensionality_reduction_frame( feature_set )
            

            if st.toggle("Enable graph visualization"):
                subgraph = OmicsGraphFilter.get_subgraph( st.session_state[ ML_SessionState.FEATURE_GRAPH.value ], feature_set )
                parameterized_graph_visualization( subgraph )


if False:
    def violinplot_feature_distribution( lab_data: LabelledData, showlegend: bool ):
        df = prepare_dataframe( lab_data )

        vio_left = go.Violin(
            x=df['feature'][ df['target'] == 0 ],
            y=df['value'][ df['target'] == 0 ],
            legendgroup='Negative class', scalegroup='Negative class', name='Negative class',
            side = "negative",
            line_color='blue',
            showlegend=showlegend
        )
        vio_right = go.Violin(
            x=df['feature'][ df['target'] == 1 ],
            y=df['value'][ df['target'] == 1 ],
            legendgroup='Positive class', scalegroup='Positive class', name='Positive class',
            side = "positive",
            line_color='orange', 
            showlegend=showlegend
        )

        return vio_left, vio_right


    def feature_distribution( feature_set ):
        keys = ML_SessionState.DISCOVERY_SET, ML_SessionState.TEST_DATA
        dataset_objs = [ st.session_state[ k.value ].select_features( feature_set ) for k in keys ]
        dataset_names = ("Training", "Test")

        fig = make_subplots( rows = 2, cols = 1, subplot_titles=dataset_names )

        for i, data in enumerate( dataset_objs ):
            vleft, vright = violinplot_feature_distribution( data, showlegend = bool(i == 0) )
            fig.add_trace( vleft, row = i + 1, col = 1)
            fig.add_trace( vright, row = i + 1, col = 1)

        fig.update_traces(meanline_visible=True)
        fig.update_layout(
            title_text = "Feature distributions",
            violingap=0, violinmode='overlay')
        
        return fig  
        

    def kde_feature_distribution( feature_set ):
        my_data = st.session_state[ ML_SessionState.DISCOVERY_SET.value ]
        target_cov = st.session_state[ ML_SessionState.SELECTED_TASK.value ][0]

        ## get sample id for each value of target_cov
        sample_sets = [
            (cov_value, my_data.metadata[ my_data.metadata[ target_cov ] == cov_value ].index.tolist() )
                for cov_value in my_data.metadata[ target_cov ].unique() 
        ]
        group_labels = [ cv for cv, _ in sample_sets ]


        for f in feature_set:
            curr_feature = my_data.features[ f ]
            groups = [ curr_feature.loc[ g ].to_numpy() for _, g in sample_sets ]
            st.write(f)
            fig = ff.create_distplot( groups, group_labels, bin_size=.5, curve_type="normal" )
            st.plotly_chart(fig, use_container_width = False )


st.title("Exploratory Data Analysis")
HelpMessage.eda()

ml_guard(st.session_state)
exploratory_data_analysis_UI()
