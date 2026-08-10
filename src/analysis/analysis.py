import streamlit as st 
from tools.enums import SessionState, GraphField
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from tools.gtools import net_guard
from tools.utils import HelpMessage


def viz_centralities(df, cols, g_id, x_name: str = "feature" ):
    chosen_metric = st.pills("Choose metric", options = cols, key=f"metric_{g_id}", default = cols[0])

    lc, rc = st.columns(2)
    with lc:
        viz_mode = st.radio("Choose visualization mode", options = ["Multi-omics","Single-omics"], key = f"cviz_mode_{g_id}", horizontal = True)
    with rc:
        ub_nfeatures = st.slider("Number of features to be shown", key = f"num2show_{g_id}")

    fig = None 
    df.sort_values( by = chosen_metric, ascending = False, inplace = True )

    if viz_mode == "Multi-omics":
        _df = df if ub_nfeatures == 0 else df.iloc[:ub_nfeatures]
        fig = px.bar(
                _df, 
                x = x_name,
                y = chosen_metric, 
                color = "omic", 
                color_discrete_map = st.session_state.my_palette_net, 
                category_orders={"omic": sorted(st.session_state.my_palette_net)} 
                ).update_xaxes(categoryorder="total ascending")
        fig.update_layout(
            title = f"View of {chosen_metric} centrality in {g_id}", 
            xaxis_title = "feature", 
            yaxis_title = chosen_metric, 
            legend_title = "Node type"
        )
    else:
        fig = make_subplots(rows=1, cols=df.omic.nunique())
        
        for i, omic_type in enumerate( sorted( df.omic.unique().tolist() ) ):
            _subdf = df[ df.omic == omic_type ] if ub_nfeatures == 0 else df[ df.omic == omic_type ].iloc[:ub_nfeatures]
            
            trace = go.Bar( 
                x = _subdf[x_name], #subdf.index.tolist(), 
                y = _subdf[chosen_metric].tolist(), 
                name=omic_type, 
                marker=dict(color=st.session_state.my_palette_net[omic_type]) )
            fig.add_trace( trace, row = 1, col = i + 1).update_xaxes(categoryorder="total ascending")

        
    st.plotly_chart(
        fig,
        use_container_width = True, 
        key = f"vizc_{g_id}"
    )


def net_analysis_UI():
    st.subheader(f"Input graph file: { st.session_state[ SessionState.LAST_UPLOADED_FILENAME.value ] }")

    tab_names = list( net_results.keys() )
    
    with st.container( border = True ): 
        st.title("Topological network structure")
        for g_id, tab in zip( tab_names, st.tabs( tab_names ) ):
            g_data = net_results[ g_id ]
            with tab:
                curr_g = avail_graphs[ g_id ]
                nv, ne = curr_g.num_vertices(), curr_g.num_edges()
                density = 2*ne / ( nv * (nv-1) )
                st.markdown(f"* Network size: {nv} x {ne}")
                st.markdown(f"* Network density: {density:.3f}")
                
                cl, cc, cr = st.columns(3)
                with cl:
                    st.plotly_chart( g_data.fig_degree_distr, use_container_width = True, key=f"dd_{g_id}" )
                with cr:
                    st.plotly_chart( g_data.fig_degree_corr, use_container_width = True, key=f"dc_{g_id}" )

                with cc:
                    st.plotly_chart( g_data.fig_avg_cc, use_container_width = True, key=f"cc_{g_id}" )
            

    with st.container( border = True ): 
        st.title("Centrality metrics")

        for g_id, tab in zip( tab_names, st.tabs( tab_names ) ):
            with tab:
                df = net_results[ g_id ].vertex_data 
                st.dataframe( df )
                cols2viz = df.columns.tolist()[ df.columns.tolist().index("degree"):   ] 
                df_for_viz = df.copy()
                df_for_viz["feature"] = df_for_viz.index 
                viz_centralities( df_for_viz, cols2viz, g_id )
                
                
    with st.container(border = True):
        st.title("Communities")
        for g_id, tab in zip( tab_names, st.tabs( tab_names ) ):
            with tab:
                l, r = st.columns(2)
                df_communities = net_results[ g_id].communities 
                with l:
                    st.dataframe( df_communities )
                with r:
                    if True: ## TODO: move in fragment 
                        avail_columns = [ c for c in df_communities.columns if c != GraphField.FEATURE_OMIC.value ]
                        cd_method = st.pills("", options = avail_columns, default = avail_columns[0], key=f"mamt_{g_id}")
                        grouped_df = df_communities.groupby([cd_method, GraphField.FEATURE_OMIC.value]).size().reset_index(name="count")

                        fig = px.bar( 
                            grouped_df, 
                            x = cd_method, 
                            y = "count", 
                            color = GraphField.FEATURE_OMIC.value, 
                            title = f"Distribution of Vertex Count per {cd_method}", 
                            labels = {cd_method: "Community ID", "count": "Number of features"},
                            color_discrete_map=st.session_state[ "my_palette_net" ],
                            category_orders={GraphField.FEATURE_OMIC.value: st.session_state[ "my_palette_net" ].keys()},
                            barmode = "stack" )

                        st.plotly_chart( fig, use_container_width = True, key=f"comm_{g_id}" )

        
    with st.container(border = True):
        st.title("Edge metrics")
        for g_id, tab in zip( tab_names, st.tabs( tab_names ) ):
            with tab:
                st.dataframe( net_results[ g_id ].edge_data )
         

st.title(f"Complex Networks Analysis")

HelpMessage.analyze_nets()


net_results = st.session_state.get( SessionState.NET_RESULTS.value )
avail_graphs = st.session_state.get( SessionState.CURRENT_GRAPH.value )


net_guard(st.session_state)

net_analysis_UI()