import streamlit as st 

from tools.enums import SessionState
from tools.gviz import get_graph_figure
from tools.utils import HelpMessage

def net_visualization_UI():
    st.subheader(f"Input graph file: { st.session_state[ SessionState.LAST_UPLOADED_FILENAME.value ] }")

    d_keys = list( avail_gstats )
    df_stats = set( avail_gstats[ d_keys[0] ].get_vertex_colnames() )
    
    for key in d_keys[1:]:
        df_stats = df_stats.intersection( avail_gstats[ key ].get_vertex_colnames() )


    df_stats = sorted( df_stats, key = lambda s: s.lower() )


    cols = st.columns(2)
    with cols[0]:
        st.selectbox("Vertex color", key="v_color", options=df_stats)
    with cols[1]:   
        st.selectbox("Vertex size", key="v_size", options=df_stats)
    
    for graph_id, g_obj in avail_graphs.items():
        
        annotated_plot = get_graph_figure(
            g = avail_graphs[ graph_id ], 
            g_id = graph_id, 
            vertex_data = avail_gstats[ graph_id ].get_vertex_associated_data(), 
            filtering_params=st.session_state[ SessionState.FILTERING_PARAMS.value], 
            prop_vcolor=st.session_state.v_color, 
            prop_vsize=st.session_state.v_size, 
            feature_name_vprop=st.session_state[ SessionState.VPROP_FEATURE_NAME.value ], 
            vpos = st.session_state[ SessionState.V_POS.value.format( graph_id ) ]
        )


        st.plotly_chart( annotated_plot, use_container_width = True )



st.title(f"Interactive Network Visualization")

HelpMessage.viz_nets()

avail_graphs = st.session_state.get( SessionState.CURRENT_GRAPH.value )
avail_gstats = st.session_state.get( SessionState.NET_RESULTS.value )


if not all( [ avail_graphs, avail_gstats ] ):
    st.error(f"You have to load a graph before!")
    st.stop()


net_visualization_UI()