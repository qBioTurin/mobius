from typing import List, Dict, Optional, Callable, Tuple, Any
import numpy as np, pandas as pd 
import graph_tool.all as gt 

import plotly.graph_objects as go 
from plotly.validators.scatter.marker import SymbolValidator

from tools.enums import FilteringParameters
from tools.enums import GraphField, EdgeType
from itertools import chain

def init_traces( _g: gt.Graph, g_id: str, g_key: str, filtering_params: FilteringParameters, feature_name_vprop: str = None, vpos = None ):
    print(f"Calcolo per {g_id} w/ {filtering_params}")
    enable_eprops = filtering_params.chosen_omics is not None 
    px_gvmanager = PlotlyGraphVizManager( _g, g_key, pos = vpos, use_edge_props=enable_eprops, feature_name_vprop=feature_name_vprop )
    px_gvmanager.set_vertices_to_be_shown( None )
    edge_traces = [
        px_gvmanager.build_edge_trace( EdgeType.ALL_EDGES, lambda w: w > 0, "#FF0000", "pos" ), 
        px_gvmanager.build_edge_trace( EdgeType.ALL_EDGES, lambda w: w < 0, "#0000FF", "neg" )
    ]
    return (g_id, ( px_gvmanager, edge_traces ) )


def get_graph_figure( 
        g: gt.Graph, 
        g_id: str, 
        vertex_data: pd.DataFrame, 
        filtering_params: FilteringParameters, 
        prop_vcolor: str, 
        prop_vsize: str, 
        feature_name_vprop: str = None, 
        graph_key: str = "WholeData", 
        vpos = None ) -> go.Figure:
    
    g_id, (px_gmanager, edge_traces ) = init_traces( 
        g, g_id, graph_key, filtering_params, feature_name_vprop, vpos = vpos ) 
    
    return px_gmanager.annotate(
        df_stats=vertex_data, 
        vcolor=prop_vcolor, vsize=prop_vsize, 
        edge_traces=edge_traces, title=g_id, feature_propname=feature_name_vprop
    )


class PlotlyGraphVizManager:

    def __init__(self, g: gt.Graph, weight_key: str, pos = None, use_edge_props: bool = True, feature_name_vprop: str = None ):

        self.feature_propname = GraphField.FEATURE_NAME.value if feature_name_vprop is None else feature_name_vprop
        self.__g = g
        self.__wkey = weight_key
        self.__vpair = list()  
        self.__edge_x = list() 
        self.__edge_y = list()
        self.__flag_edge_w = use_edge_props
        self.__edge_w = list()
        self.__edge_types = list()
        self.__enable_subgraph = None 

        self.__pos = pos if pos is not None else self.get_pos( g )
        self.__init_edge_trace( g, self.__pos )
    
    
    @property
    def graph(self) -> gt.Graph:
        return self.__g
    
    @property
    def pos_vertex_property(self) -> gt.VertexPropertyMap:
        return self.__pos
    

    def set_vertices_to_be_shown(self, vlist: List[int]):
        self.__enable_subgraph = vlist
        self.__node_trace = self.get_node_trace( self.__pos, self.__enable_subgraph )
        
        
    def __build_legend( self, samples_id: List[int] ) -> Tuple[ List[ Any ], Any ]:

        markers = list( SymbolValidator().values )[2:][::3]
        markers = [ m for m in markers if not any( [ m.endswith("-open"), m.endswith("-dot") ] )]
        
        if GraphField.FEATURE_OMIC.value in self.__g.vp:
            omic_property = [ self.__g.vp.omic[ i ] for i in samples_id ]
        else:
            omic_property = ["o"] * len(samples_id)  
            
        enum_labels = { 
            omic_type: markers[ i ] 
                for i, omic_type in enumerate( set( omic_property ) ) 
        }
        sample_symbols = [ enum_labels[ l ] for l in omic_property ] 
        legend_data = []
        for omic, shape in enum_labels.items():
            legend_data.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, symbol=shape),
                    name=omic
                )
            )

        return sample_symbols, legend_data

    def annotate(self, df_stats: pd.DataFrame, vcolor: str, vsize: str, edge_traces: List, title: str, feature_propname: str = None):
        legend = self.set_annotation( df_stats, vsize, vcolor )
        return self.make_figure( self.__node_trace, edge_traces, legend, title )


    @classmethod
    def make_figure(cls, node_trace, edge_traces, omics_legend, title: str):
        fig = go.Figure(
            data=[*edge_traces, node_trace, *omics_legend],
            layout=go.Layout(
                title=title,  #'<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="", #"Python code: YES",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
        return fig 

            
    @classmethod
    def get_pos(cls, g: gt.Graph):
        return gt.sfdp_layout( g )

    @classmethod
    def get_node_trace(cls, pos: gt.VertexPropertyMap, indices: Optional[List[int]]) -> go.Scatter:

        coords = list( pos )

        if indices is not None: 
            coords = [ coords[i] for i in indices ]
            
        node_x, node_y = list( zip( *coords ) )
        
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale="Portland",   #='Phase',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    x = -0.2,
                    thickness=15,
                    titleside='bottom'
                ),
                line_width=2)) 


    def __init_edge_trace(self, g: gt.Graph, pos: gt.VertexPropertyMap):       

        eprops = list() 
        if self.__flag_edge_w:
            eprops = [ self.__g.ep[ self.__wkey ], self.__g.ep[ GraphField.INTRAOMIC_EDGE.value] ] #self.__g.ep[ "intra_edge" ] ]):

        for edge_data in g.iter_edges( eprops ):
            u, v = edge_data[:2]
            w1, w2 = edge_data[2:] if self.__flag_edge_w else (1., EdgeType.ALL_EDGES.value ) 
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            self.__vpair.append( (u,v) )
            self.__edge_x.append([ x0, x1, None ])
            self.__edge_y.append([ y0, y1, None ])
            self.__edge_w.append( w1 )
            self.__edge_types.append( EdgeType( w2 ) )


    def __edge_belongingness_check(self, edge_id: int ) -> bool:
        u, v = self.__vpair[ edge_id ]
        return (
            self.__enable_subgraph is None or 
                u in self.__enable_subgraph and v in self.__enable_subgraph
        )
    

    def build_edge_trace( self, edge_type: EdgeType, cmp: Callable, color: str, trace_name: str ):
        set_edge_coords = [
            (self.__edge_x[i], self.__edge_y[i]) 
                for i in range( len( self.__edge_w ) ) 
                    if self.__edge_belongingness_check(i) and cmp( self.__edge_w[i] ) and \
                        (edge_type == EdgeType.ALL_EDGES or edge_type == self.__edge_types[ i ])
        ]

        if len( set_edge_coords ) > 0:
            edge_x, edge_y = ( chain.from_iterable(edge_coords) for edge_coords in zip( *set_edge_coords ) )
        else:
            edge_x, edge_y = list(), list()

        return go.Scatter(
            x = list(edge_x), 
            y = list(edge_y), 
            line = dict( width = 0.5, color = color ), 
            hoverinfo = "none", 
            mode = "lines",
            name = trace_name
        )


    def __format_annot( self, node_id, name1, val1, name2, val2 ):
        return f"{self.__g.vp[ self.feature_propname ][node_id]}: {name1[:3]}={val1:.6f}, {name2[:3]}={val2:.6f}"


    def set_annotation(self, df: pd.DataFrame, vsize_prop: str, vcolor_prop: str ):


        if self.__enable_subgraph:
            samples_id = sorted( self.__enable_subgraph )
            df = df.iloc[ samples_id ]
        else:
            samples_id = list( range( self.__g.num_vertices() ) )

        values_color = df[ vcolor_prop ] #.astype(str)
        values_size = df[ vsize_prop ]

        node_annot = [
            self.__format_annot( node_id, vcolor_prop, pc, vsize_prop, ps )
                for node_id, pc, ps in zip( samples_id, values_color, values_size )
        ]

        if False:
            # https://plotly.com/python/marker-style/#custom-marker-symbols
            markers = list( SymbolValidator().values )[2:][::3]
            markers = [ m for m in markers if not any( [ m.endswith("-open"), m.endswith("-dot") ] )]
            
            if GraphField.FEATURE_OMIC.value in self.__g.vp:
                omic_property = [ self.__g.vp.omic[ i ] for i in samples_id ]
            else:
                omic_property = ["o"] * len(samples_id)  #[ "lel" for _ in samples_id ]
                
            enum_labels = { 
                omic_type: markers[ i ] 
                    for i, omic_type in enumerate( set( omic_property ) ) 
            }
            symbols = [ enum_labels[ l ] for l in omic_property ]

        symbols, plotly_legend = self.__build_legend( samples_id )

        self.__node_trace.text = node_annot

        self.__node_trace.marker.color = values_color
        self.__node_trace.marker.symbol = symbols 


        sizeable_values = np.nan_to_num( values_size.to_numpy() )
        
        m, M = sizeable_values.min(), sizeable_values.max()
        diff = M - m 
        if diff == 0:
            diff = 1
        mi, ma = 30., 100
        power = .5

        my_new_vals = list() 
        for x in sizeable_values:
            new_val = mi + (ma - mi) * ( ( (x - m) / diff ) ** power ) 
            my_new_vals.append( new_val )


        my_new_vals = np.array( my_new_vals )

        ## docs - https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
        self.__node_trace.marker.size = np.array( my_new_vals )
        self.__node_trace.marker.sizemin = 3
        self.__node_trace.marker.sizemode = "diameter" # area or diameter
        self.__node_trace.marker.sizeref = 4

        return plotly_legend


