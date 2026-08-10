import streamlit as st 
import graph_tool.all as gt 
import pandas as pd 
import numpy as np
from typing import Tuple
from itertools import combinations

from tools.enums import SessionState
from tools.gtools import build_edge_dataframe, net_guard
from tools.utils import build_tsv_archive, HelpMessage


TURI_CHECK = "turi_check" ## TODO: change this name + move to enums


def perform_overlap( df_g1: pd.DataFrame, df_g2: pd.DataFrame ):
    edges_g1 = set( df_g1.index.tolist() )
    edges_g2 = set( df_g2.index.tolist() )
    elists = dict(
        common = edges_g1.intersection( edges_g2 ),
        g1_only = edges_g1 - edges_g2,
        g2_only = edges_g2 - edges_g1,
        union = edges_g1.union( edges_g2 )
    )
    return { 
        k: pd.MultiIndex.from_tuples( v, names = ["f1", "f2"]) 
            for k, v in elists.items() }


def graph_overlap( info_g1: Tuple[str, gt.Graph], info_g2: Tuple[str, gt.Graph] ):
    name_g1, g1 = info_g1
    name_g2, g2 = info_g2

    df_1, df_2 = [ build_edge_dataframe( g ) for g in (g1, g2)]
    info_overlap = perform_overlap( df_1, df_2 )

    overlap_results = {
        name_g1: df_1.loc[ info_overlap["g1_only"] ] ,
        name_g2: df_2.loc[ info_overlap["g2_only"] ]
    }
    
    commonset = info_overlap["common"]
    df_common = pd.merge( df_1.loc[commonset], df_2.loc[commonset].drop(columns=["intra-omic"]), left_index=True, right_index=True )
    df_common.insert(1, "concordant", ( np.sign( df_common.rho_x ) == np.sign( df_common.rho_y ) ) )
    df_common.columns = list( map( 
        lambda cname: cname.replace("_x", f"_{name_g1}").replace("_y", f"_{name_g2}"), 
        df_common.columns.tolist() ) )

    overlap_results["common"] = df_common
    return overlap_results 


def net_overlapping_UI():
    nets = st.session_state[ SessionState.CURRENT_GRAPH.value ]


    with st.form("gcompare_form"):
        st.multiselect( 
            "Select 2+ graphs", 
            key = "graph_pair", 
            options = list(nets.keys())  )


        if st.form_submit_button("Run overlap"):
            g_names = st.session_state.graph_pair 

            if len(g_names) >= 2:
                comparisons = combinations( g_names, 2 )
                fucking_comparisons = dict() 

                with st.status(f"Obtaining graph w/ parameters...", expanded=True) as status:
                    for a, b in comparisons:
                        st.write(f"facendo {a} vs {b}")
                        fucking_comparisons[ (a,b) ] = graph_overlap( (a, nets[a]), (b, nets[b]) )
                        
                    st.session_state[ TURI_CHECK ] = fucking_comparisons
                    status.update(label="Computation completed!", state="complete", expanded=False)

        
    if TURI_CHECK in st.session_state:
        for (a,b), overlap_data in st.session_state[TURI_CHECK].items():

            id_pair = (a,b)
            names_pair = name_a, name_b = [ name.replace(".gt", "") for name in id_pair]

            with st.expander(f"Comparison: **{name_a}** vs **{name_b}**", expanded = False):
                common = overlap_data["common"]
                st.subheader(f"Common interactions: {common.shape}")
                st.dataframe( common )

                for i, col in enumerate( st.columns(2) ):
                    with col:
                        df = overlap_data[ id_pair[i] ]
                        st.header(f"Interactions {names_pair[i]}: {df.shape}")
                        st.dataframe( df )

                with open( build_tsv_archive( **overlap_data ), "rb" ) as fp:
                        st.download_button(
                            "Download interactions", 
                            data = fp,
                            key = f"{name_a}_{name_b}",
                            file_name = f"interactions_{name_a}_{name_b}.zip", 
                            mime = "application/zip" )

    
st.title(f"Edge Overlapping Analysis")

HelpMessage.overlap_nets()

net_guard(st.session_state)
net_overlapping_UI() 