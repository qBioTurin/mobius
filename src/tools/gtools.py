import streamlit as st 
import graph_tool.all as gt 
import numpy as np 
import pandas as pd 
import zipfile

import netrd, networkx as nx 
import fastcluster
from scipy.cluster import hierarchy

from collections import defaultdict, Counter
from functools import partial 
from itertools import combinations
from typing import Dict, List, Union, Optional, Tuple, Any, Iterable


import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr, kendalltau
from statsmodels.stats.multitest import fdrcorrection
from sknetwork.clustering import Louvain

import plotly.graph_objects as go 
from sklearn.cluster import SpectralClustering

import logging, time 

from tools.enums import GraphField, EdgeType, GraphFilteringParametrization, Coefficients, SessionState
from tools.utils import build_graph_zip


def net_guard(session_state):
    if SessionState.CURRENT_GRAPH.value not in session_state:
        st.error("You have to load a graph before!")
        st.stop()


@st.cache_data
def get_input_graph_names(input_data) -> List[str]:
    filelist = list()
    
    if input_data:
        try:
            with zipfile.ZipFile( input_data, "r" ) as input_zip:
                filelist = list( filter( lambda s: s.endswith(".gt") or s.endswith(".gt.xz"), input_zip.namelist() )  )
        except zipfile.BadZipFile:
            filelist = [ input_data.name ]
    
    return filelist


def prepare_graph( 
        g: gt.Graph, 
        params_filter: GraphFilteringParametrization, 
        remove_zero_degree_nodes: bool = True ) -> gt.Graph:
    
    filt_params = params_filter.filtering_params

    gfilter = OmicsGraphFilter
    gfilter.apply_omic_edge_filtering( g, filt_params.edge_type )
    gfilter.apply_omic_vertex_filtering( g, filt_params.chosen_omics )
    gfilter.apply_graph_edge_filtering( 
        g, 
        params_filter.graph_key, 
        filt_params.min_corr_threshold,
        filt_params.pvalue_threshold, 
        filt_params.padj_flag  )
    if remove_zero_degree_nodes:
        gfilter.apply_graph_vertex_filtering( g )
    
    new_g = gt.Graph( g = g, prune = True )
    g.clear_filters()

    return new_g


def compute_centralities( g, vertex_name_prop: str = None) -> Tuple[ pd.DataFrame, Dict ]: 
    comp, hist = gt.label_components( g, directed=False )
    cc_size = [ hist[ cc_node ] for cc_node in comp.a ]
    value_props = dict( component = comp, degree = g.degree_property_map("out") )

    if g.is_directed():
        value_props[ "degree_in" ] = g.degree_property_map("in")

    value_props[ "betweenness" ], betw_ep = gt.betweenness( g )    
    value_props[ "closeness" ] = gt.closeness( g )
    value_props[ "pagerank" ] = gt.pagerank( g )
    value_props[ "clustering" ] = gt.local_clustering( g )

    cols_value_props = { k: v.a for k, v in value_props.items() }
    columns_data = dict( cc_size = cc_size, **cols_value_props)#, **key_dict )

    df = build_g_dataframe( g, columns_data=columns_data, vertex_name_prop=vertex_name_prop)

    if GraphField.FEATURE_OMIC.value in g.vp:
        df.insert(0, "omic", list( g.vp[ GraphField.FEATURE_OMIC.value ] ) ) 

    dict_gt_props = dict(
        components = comp, 
        degree = value_props["degree"],
        betweenness = value_props["betweenness"],
        edge_betweenness = betw_ep,
        closeness = value_props["closeness"], 
        pagerank = value_props["pagerank"], 
        clustering = value_props["clustering"], 
    )

    if g.is_directed():
        degree_in = "degree_in"
        dict_gt_props[ degree_in ] = value_props[ degree_in ]

    return df, dict_gt_props


def build_g_dataframe( g: gt.Graph, columns_data: Dict[str, np.array], vertex_name_prop: str = None):
    if vertex_name_prop is None: 
        vertex_name_prop = GraphField.FEATURE_NAME.value


    return pd.DataFrame( 
        index = g.vp[ vertex_name_prop ],
        data = { key: value for key, value in columns_data.items() if value is not None }
    )


def build_edge_dataframe( g: gt.Graph, g_key: str = None, vertex_name_prop: str = None ):
    def get_significance_symbol(pvalue):
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*" if pvalue < 0.05 else ""
    
    def get_degree_difference(v1, v2, degrees):
        return np.abs( degrees[v1] - degrees[v2] )
    
    g_key = "WholeData" if g_key is None else g_key
    feature_vprop = vertex_name_prop if vertex_name_prop else GraphField.FEATURE_NAME.value


    if feature_vprop == GraphField.FEATURE_NAME.value:
        field_list = [ GraphField.GRAPH_REPR, GraphField.PVALUE, GraphField.ADJ_PVALUE, GraphField.INTRAOMIC_EDGE ]
        eprops = [
            g.ep[ field.value.format( g_key ) ] for field in field_list 
        ]
        data = [
            ( u, v, t, w, p, adj_p, get_significance_symbol( adj_p ) )
                for u, v, w, p, adj_p, t in g.iter_edges( eprops ) 
        ]
        columns = ["f1", "f2", "intra-omic", "rho", "p-value", "padj-value", "stat.sign" ]
    else:
        data = list( g.iter_edges() )
        columns = ["f1", "f2" ]

    columns.append( "deg_diff" )
    degrees = g.degree_property_map("out").a
    vpn = g.vp[ feature_vprop ]

    return pd.DataFrame(
        data = [ 
            (vpn[row[0]], vpn[row[1]], *row[2:], get_degree_difference( row[0], row[1], degrees )  ) 
                for row in data ], 
        columns = columns 
    ).set_index(["f1", "f2"])


def new_string_gp( g: gt.Graph, prop_name: GraphField, prop_value: str, fmt : str = None ):
    key = prop_name.value.format( fmt ) if fmt else prop_name.value
    g.gp[ key ] = g.new_gp("string")
    g.gp[ key ] = prop_value


def new_string_vp( g: gt.Graph, prop_name: GraphField, prop_value: List[str], fmt : str = None ):
    key = prop_name.value.format( fmt ) if fmt else prop_name.value
    v = g.vp[ key ] = g.new_vp("string")
    v.set_2d_array( np.array( prop_value ) )


def new_numeric_vp( g: gt.Graph, prop_name: GraphField, prop_type: str, prop_values: List[ Union[float, int] ], fmt: str = None):
    key = prop_name.value.format( fmt ) if fmt else prop_name.value
    vp = g.vp[ key ] = g.new_vp( prop_type )
    vp.a = np.array( prop_values )


def graph_reweighting(g: gt.Graph, vscore_name: str, ep_name: str) -> gt.Graph:
    """ Reweight the edges of a graph based on vertex scores with the purpose of 
    creating a new graph with the same vertex scores but different edge weights, 
    to weaken the edges weights between high vertex scores."""

    new_g = gt.Graph(directed=g.is_directed())
    new_g.add_vertex(g.num_vertices())

    new_g.vp[vscore_name] = new_g.new_vertex_property("double", vals = g.vp[vscore_name].a)
    new_string_vp( new_g, GraphField.FEATURE_NAME, list( g.vp[GraphField.FEATURE_NAME.value] ) )
    new_string_vp( new_g, GraphField.FEATURE_OMIC, list( g.vp[GraphField.FEATURE_OMIC.value] ) )

    new_g.add_edge_list(g.edges())

    a_vscores = g.vp[vscore_name].a 
    

    mean_lambda = lambda i, j: 0.5 * (a_vscores[i] + a_vscores[j])
    vp_mean = np.array([ (1-( mean_lambda(u,v) ))**2    for u, v in g.iter_edges() ]) 
    ew = np.abs( g.ep[ep_name].a  )

    new_g.ep[ep_name] = new_g.new_edge_property("double", vals = np.multiply( ew, vp_mean) )

    return new_g


class OmicsGraphBuilder:

    CORR_FUNC_SET = ( 
        Coefficients.PEARSON, 
        Coefficients.SPEARMAN, 
        Coefficients.XI, 
        Coefficients.KENDALL
    )


    def __init__(self, omics: Dict[ str, pd.DataFrame], corr_func: str ) -> None:
        
        self.__g = gt.Graph( g = None, directed = False )
        self.__omics = omics 
        self.__omic_table = pd.concat( omics.values(), axis = 1 )

        self.__edge_cache = defaultdict( list ) 

        self.__feature_set = list() 
        self.__fmap = dict()

        self.__corr_func = None 
        self.__set_correlation_function( corr_func.lower() )


    @property
    def graph(self) -> gt.Graph:
        return self.__g 


    def init_vertices(self) -> Dict[str, str]:
    
        self.__feature_set = [
            ( fname, omic_name )
                for omic_name, df in self.__omics.items()
                    for fname in df.columns.tolist()
        ]
        self.__feature_set.sort( key = lambda pair: pair[0] )
        feature_names, feature_omics = zip( *self.__feature_set )
        ## mapping from feature names to integers 
        self.__fmap = { name: n for n, name in enumerate( feature_names ) }

        ## initialize graph vertices
        self.graph.add_vertex( len( feature_names ) )
        ## save feature names as vertex property
        new_string_vp( self.__g, GraphField.FEATURE_NAME, feature_names )
        new_string_vp( self.__g, GraphField.FEATURE_OMIC, feature_omics )

        return { name: n for n, name in enumerate( feature_names ) }


    def build_edges(self, df: pd.DataFrame, strat: str, intra_list: List, inter_list: List):
        
        global_sampleset = df.index.tolist()
        strat_values = list() 

        logging.info(f"Getting vertex weights for 'WholeData'")
        self.__compute_edge_weights( "WholeData", global_sampleset, intra_list, inter_list )

        for cat, subdf in df.groupby( strat ):
            logging.info(f"Category '{cat}': {subdf.shape}")

            logging.critical(f"Getting vertex weights for '{cat}'")
            self.__compute_edge_weights( cat, subdf.index.tolist(), intra_list, inter_list )
            strat_values.append( cat )

        new_string_gp( self.__g, GraphField.STRAT_LAYERS, "$".join( sorted( strat_values ) ) )
            

    def create_edges(self):
        ## init dictionary of np.array of length N equal to the number of categories
        ordered_keys = sorted( self.__edge_cache.keys() )
        edge_list = defaultdict( partial( np.zeros, 2 * len( ordered_keys ) ) )
                                
        for i, key in enumerate( ordered_keys ):
            for u, v, correlation_score in self.__edge_cache[ key ]:
                dict_entry, j = edge_list[ (u, v) ], 2*i
                dict_entry[j], dict_entry[j+1] = correlation_score 


        ## build n-uples composed of edges (u,v) and the set of weights & labels
        omic = self.graph.vp[ GraphField.FEATURE_OMIC.value ]

        edge_list = np.array([
            np.array([ u, v, *weights, omic[u] == omic[v] ])
                for (u, v), weights in edge_list.items() 
        ])

        ## build edges and save properties
        edge_properties = [ self.graph.new_ep("double") for _ in range( 2*len(ordered_keys) ) ]
        edge_prop_sameomic = self.graph.new_ep("bool") 

        self.graph.add_edge_list( edge_list, eprops = [ *edge_properties, edge_prop_sameomic ] )

        self.graph.ep[ GraphField.INTRAOMIC_EDGE.value  ] = edge_prop_sameomic

        for i, cat_name in enumerate( ordered_keys ):
            ## save correlation coefficient values
            self.graph.ep[ cat_name ] = edge_properties[ 2 * i ]
            ## save p-values
            field_pvalue = GraphField.PVALUE.value.format( cat_name ) 
            self.graph.ep[ field_pvalue ] = edge_properties[2 * i + 1]
            ## compute and save adjusted p-values
            self.__compute_adjpvalues( cat_name )


        print(f"Built graph: {self.graph}") 


    def __get_feature_omictype( self, feature: str ):
        return self.__feature_set[ self.__fmap[ feature ] ][1]


    def __get_fid(self, feature_list: List[str], index: int ):
        return self.__fmap[ feature_list[ index ] ]


    def __compute_edge_weights( self, ep_name: str, sampleset: List, intra_list, inter_list ):
        def get_corr_score( f1, f2, mapping: Dict[str, int], c_matrix: np.array, p_matrix: np.array, set_fconstant: set ):
            if f1 in set_fconstant or f2 in set_fconstant:
                cc, p = 0., 1.
            else:
                k, l = mapping[ f1 ], mapping[ f2 ]
                cc, p = c_matrix[k, l], p_matrix[k, l]

            return (cc, p)


        ## retrieve set of interactions to be computed
        interaction_set = set( inter_list ) 
        interaction_set.update({ (y, x) for x, y in inter_list })
        interaction_set.update({ (x, x) for x in intra_list })
        ## retrieve the list of samples and the number/type of features
        input_data = self.__omic_table.loc[ sampleset ].copy() 
        cols, n_features = self.__omic_table.columns.tolist(), self.__omic_table.shape[1]
        omic_types = [ self.__get_feature_omictype( feature ) for feature in cols ]
        ## check for constant columns to be removed ==> "a priori" uncorrelated
        m = input_data.to_numpy() 
        useless_cols = np.all( m == m[0,:], axis = 0)
        non_constant_cols = [ col for col, is_useless in zip( cols, useless_cols ) if not is_useless ]
        constant_cols = set( cols ) - set( non_constant_cols )
        ### efficient computation of pairwise correlations for non-constant columns only
        corr, pvalues = self.__corr_func( input_data[ non_constant_cols ].copy() )

        assert not( np.any( np.isnan( corr ) ) or np.any( np.isnan( pvalues ) ) ), "WARNING: NaN values in correlation matrix -> check your data"
        ## generate pairs of features to compute correlations
        feature_pair_generator = (
            ( cols[i], cols[j] ) 
                for i, omic_label in enumerate( omic_types )
                    for j in range( i+1, n_features )
                        if ( omic_label, omic_types[j] ) in interaction_set
        )

        ## create a temporary mapping for the non-constant columns
        tmp_fmapping = { f: i for i, f in enumerate( non_constant_cols ) }
        # retrieve the correlation values and p-values for the pairs of features 
        self.__edge_cache[ ep_name ] = [
            ( self.__fmap[ f_1 ], self.__fmap[ f_2 ], get_corr_score( f_1, f_2, tmp_fmapping, corr, pvalues, constant_cols ) ) 
                for f_1, f_2 in feature_pair_generator
        ]


    def __compute_adjpvalues(self, condition):
        logging.info(f"Computing adjusted-pvalues for condition '{condition}")
        pvalue_field, adjpvalue_field = [ enum_val.value.format( condition ) for enum_val in ( GraphField.PVALUE, GraphField.ADJ_PVALUE ) ]
        ep = self.graph.ep[ adjpvalue_field ] = self.graph.new_ep("double")
        _, ep.a = fdrcorrection( np.nan_to_num( self.graph.ep[ pvalue_field ].a ) )


    def __set_correlation_function(self, corr_func: str ): 
        corr_func = corr_func.upper()

        match corr_func:
            case Coefficients.SPEARMAN.value:
                self.__corr_func = spearmanr
            case Coefficients.PEARSON.value:
                self.__corr_func = pearsonr
            case Coefficients.KENDALL.value:
                self.__corr_func = kendalltau
            case Coefficients.XI.value:
                raise NotImplementedError("need to import to proper module")
                self.__corr_func = lambda x, y: compute_xi_correlation(x, y, get_p_values = True )
            case _:
                logging.critical(f"Unknown function: ({corr_func}); correlation function set as 'spearman' by default")
                corr_func = Coefficients.SPEARMAN


        new_string_gp( self.graph, GraphField.CORRELATION_FUNCTION, corr_func )
 

class OmicsGraphFilter:
    
    SEPARATOR='$'

    ENCODED_GRAPH_FIELDNAME = GraphField.ENCODED_GRAPHS.value
    FEATURE_NAME_FIELDNAME = GraphField.FEATURE_NAME.value
    FEATURE_OMIC_FIELDNAME = GraphField.FEATURE_OMIC.value


    @classmethod
    def get_categories( cls, mog: gt.Graph ) -> List[ str ]:
        fieldname = GraphField.ENCODED_GRAPHS.value
        if fieldname in mog.gp:
            gprop = mog.gp[ fieldname ] 
            return gprop.split( cls.SEPARATOR )
    
    @classmethod
    def get_strata( cls, mog: gt.Graph) -> List[str]:
        field = GraphField.STRAT_LAYERS.value
        if field in mog.gp:
            return mog.gp[ field ].split( cls.SEPARATOR )

    @classmethod
    def set_categories(cls, mog: gt.Graph, categories: List[str] ):
        gp = mog.new_graph_property("string", cls.SEPARATOR.join( categories ) )
        mog.graph_properties[ cls.ENCODED_GRAPH_FIELDNAME ] = gp

    @classmethod
    def apply_omic_edge_filtering( cls, mog: gt.Graph, edge_type: EdgeType ):
        if edge_type != EdgeType.ALL_EDGES:
            edge_type_filter = mog.ep[GraphField.INTRAOMIC_EDGE.value].a == edge_type.value #req_label
            mog.set_edge_filter( mog.new_ep( "bool", vals = edge_type_filter ) ) 
 

    @classmethod
    def apply_omic_vertex_filtering( cls, mog: gt.Graph, omics_set):
        if omics_set is not None:
            wholeset = set( cls.get_omics_labels( mog ) )
            omics_set = set( omics_set )
            
            if omics_set != wholeset:
                mask = [ (l in omics_set) for l in mog.vp[ cls.FEATURE_OMIC_FIELDNAME ] ]
                mog.set_vertex_filter( mog.new_vp( "bool", vals = mask ) )


    @classmethod
    def get_omics_labels( cls, mog: gt.Graph ):
        fieldname = cls.FEATURE_OMIC_FIELDNAME 
        if fieldname in mog.vp:
            return sorted( set( mog.vp[ fieldname ] ) )

    @classmethod
    def get_feature_names( cls, mog: gt.Graph, feature_name_fieldname: str = None ):
        if feature_name_fieldname is None:
            feature_name_fieldname = cls.FEATURE_NAME_FIELDNAME 

        return list( mog.vp[ feature_name_fieldname ] )


    @classmethod
    def get_omic_types( cls, mog: gt.Graph ):
        return list( mog.vp[ cls.FEATURE_OMIC_FIELDNAME ] )

    @classmethod
    def set_edge_filter( cls, 
            mog: gt.Graph, 
            bitmask: np.array ):
        
        mog.set_edge_filter( mog.new_ep( "bool", vals = bitmask ) ) 


    @classmethod
    def set_vertex_filter( cls, mog: gt.Graph, bitmask: np.array, ret_new_graph: bool = False ): 
        mog.set_vertex_filter( mog.new_vp( "bool", vals = bitmask ) )
        if ret_new_graph:
            new_g = cls.get_pruned_graph( mog )
            mog.clear_filters()
            return new_g

    @classmethod
    def apply_graph_edge_filtering( cls, 
            mog: gt.Graph, 
            key: str, 
            corr_thr: float, 
            pv_thr: float, 
            use_adj_pvalue: bool,
            return_filter: bool = False ):
        

        field_pvalue = GraphField.ADJ_PVALUE.value if use_adj_pvalue else GraphField.PVALUE.value
        thresholded_edges = np.abs( mog.ep[ key ].a ) > corr_thr

        if pv_thr is not None: 
            pthresholded_edges = mog.ep[ field_pvalue.format( key ) ].a <= pv_thr
            thresholded_edges = thresholded_edges & pthresholded_edges
            
        if return_filter:
            return thresholded_edges
        
        cls.set_edge_filter( mog, thresholded_edges )
             

    @classmethod
    def apply_graph_vertex_filtering( cls, mog: gt.Graph ):
        degrees = mog.get_out_degrees( list( mog.vertices() ) )
        mog.set_vertex_filter( mog.new_vp( "bool", vals = degrees > 0 ) ) 


    @classmethod
    def get_pruned_graph(cls, mog: gt.Graph ) -> gt.Graph:
        return gt.Graph( g = mog, prune = True ) 


    @classmethod
    def data_extraction(cls, mog: gt.Graph):
        encoded_g = cls.get_categories( mog )
        template_attr_list = [ "{}", GraphField.PVALUE.value, GraphField.ADJ_PVALUE.value ]
        attr_list = [ attr.format( g ) for attr in template_attr_list for g in encoded_g ]
        eprop_list = map( lambda ep: mog.ep[ ep ], attr_list )
        edge_set = mog.iter_edges( eprops = list( eprop_list ) )

        f_corr_name = mog.gp[ GraphField.CORRELATION_FUNCTION.value ]
        corr_columns = [ f"{f_corr_name}_{g_name}" for g_name in encoded_g ]
        index = [ "u", "v" ]
        columns = index + corr_columns + attr_list[3:]
        feature_names = dict( enumerate( mog.vp[ GraphField.FEATURE_NAME.value ] ) )
        df = pd.DataFrame( data = list( edge_set ), columns = columns  )
        for col in index:
            df[col].replace( feature_names, inplace = True )

        return df.set_index( keys = index )

    @classmethod
    def get_subgraph(cls, mog: gt.Graph, feature_set: Iterable[str], get_graphview: bool = False) -> gt.Graph:
        feature_set = set( feature_set )
        fucking_bitmask = np.array( [ 
            (feature in feature_set) 
                for feature in  mog.vp[ GraphField.FEATURE_NAME.value ] ] )
        if get_graphview:
            return gt.GraphView( mog, vfilt = mog.new_vp( "bool", vals = fucking_bitmask ) )
        else:
            cls.set_vertex_filter( mog, fucking_bitmask )
            subgraph = cls.get_pruned_graph( mog )
            mog.clear_filters()
            return subgraph

class GraphCoherenceCheck:
    def __init__(
            self, 
            zip_archive: str, 
            label_alphabet: List[str], 
            significance_level: float, 
            use_adj_pvalues: bool ) -> None:
        
        self.__zip_archive = zip_archive
        self.__cohorts = label_alphabet
        self.__significance_lvl = significance_level
        self.__gfield_pvalue = GraphField.ADJ_PVALUE.value if use_adj_pvalues else GraphField.PVALUE.value 

        self.__local_bitmasks = dict()
        self.__feature_names = None 

    
    def __init_feature_names(self, g: gt.Graph):
        if self.__feature_names is None: 
            self.__feature_names = {
                i: name for i, name in enumerate( g.vp[ GraphField.FEATURE_NAME.value ])
            }
            print(f"Feature names initialized -- {len(self.__feature_names)}")
        

    def get_graph_by_name(self, graph_id: str) -> gt.Graph:
        filename = f"graph_{graph_id}.gt"

        with zipfile.ZipFile( self.__zip_archive, "r" ) as zip:
            if filename in zip.namelist():
                return gt.load_graph( zip.open( filename ) )
            
            raise FileNotFoundError(f"WTF bro")

    def get_graph_iterator(self):
        with zipfile.ZipFile( self.__zip_archive, "r" ) as zip:
            graph_namelist = filter( lambda fn: fn.endswith(".gt"), zip.namelist() )  
            for filename in graph_namelist: #zip.namelist():
                condition = filename.replace(".gt", "").split("_")[-1]
                g = gt.load_graph( zip.open( filename ) )
                self.__init_feature_names( g )
                yield g, condition


    def extract_global_datatable(self) -> pd.DataFrame:
        global_mask = self.__compute_global_bitmask( list( self.__local_bitmasks.values() ) )
        
        df_list = [
            self.__build_edge_based_dataframe( g, condition, global_mask, True )
                for g, condition in self.get_graph_iterator() 
        ]
        return pd.concat( df_list, axis = 1 )
    

    def extract_cohort_dependent_datatable(self) -> Dict[str, pd.DataFrame]:
        return {
            condition: self.__build_edge_based_dataframe( g, condition, self.__local_bitmasks[condition], False )
                for g, condition in self.get_graph_iterator()
        }
    

    def save_filtered_graphs(self, zip_filename: str):

        pruned_graphs = dict()

        for g, condition in self.get_graph_iterator():
            OmicsGraphFilter.set_edge_filter( g, self.__local_bitmasks[ condition ] )
            pruned_graphs[ condition ] = gt.Graph( g = g, prune = True, directed = False )


        return build_graph_zip( pruned_graphs, zip_filename )
    

    def __build_edge_based_dataframe( self, g: gt.Graph, g_id: str, bit_filter: np.array, rename_cols: bool ) -> pd.DataFrame:

        if bit_filter is not None: 
            g.set_edge_filter( g.new_ep( "bool", vals = bit_filter ) )
        
        global_value = "WholeData" #GraphField.GRAPH_REPR_WHOLEDATA.value
        cohorts = [ label for label in self.__cohorts if label in g.ep.keys() ]
        cohorts.insert(0, global_value)


        pv_fieldlist = ( GraphField.PVALUE.value, GraphField.ADJ_PVALUE.value )
        pvalues = [ pvalue_field.format(label) for label in cohorts for pvalue_field in pv_fieldlist ]
        
        w_eprops = [ g.ep[ label ] for label in cohorts ]
        p_eprops = [ g.ep[ label ] for label in pvalues ]
        my_eprops = w_eprops + p_eprops

        edge_collection = [
            edge_data for edge_data in g.iter_edges( eprops = my_eprops )
        ]


        index = [ "u", "v" ]
        cols = cohorts + pvalues 
        if rename_cols:
            cols = [ f"{g_id}__{label}" for label in cols ]
            
        cols = index + cols 
        df = None 

        if len(edge_collection) > 0:
            df = pd.DataFrame( data = edge_collection, columns = cols )
            df.replace( { col: self.__feature_names for col in index }, inplace=True )
            #     # df[col].replace( self.__feature_names, inplace=True ) 
            df.set_index(keys=index, inplace=True)
        
        return df 

    def __compute_global_bitmask( self, bitmask_list: List[ np.array ]) -> np.array:
        global_bitmask = np.full( shape = bitmask_list[0].shape, fill_value = False )
        for lb in bitmask_list:
            logging.debug(f"in __compute_global_bitmask -- num. ones in current bitmask: {sum(lb)}")
            global_bitmask = np.bitwise_or( global_bitmask, lb )

        return global_bitmask
    

    def compute_local_bitmasks(self):

        for g, condition in self.get_graph_iterator():
            self.__local_bitmasks[ condition ] = self.__compute_graph_bitmask( g, self.__cohorts )
            logging.debug(f"Current condition '{condition}': {np.sum(self.__local_bitmasks[ condition ])} interactions")


    def __compute_graph_bitmask(self, g: gt.Graph, cohorts: List[str] ):
        cdg_labels = [ label for label in cohorts if label in g.ep.keys() ] #[ GraphField.GRAPH_REPR.value.format(label) for label in cohorts if label in g.vp.keys() ]
        cig_label = "WholeData" #GraphField.GRAPH_REPR_WHOLEDATA.value       ## cohort-independent graph 

        ### 1 . coerenza correlazione 
        cig_signs = np.sign( g.ep[ cig_label ].a )
        ### 2. significatività 
        bitmask_cig_pvalues = g.ep[ self.__gfield_pvalue.format( cig_label ) ].a <= self.__significance_lvl
        
        ## initialize bitmasks 
        bitmask_corr_sign = np.full( shape = cig_signs.shape, fill_value = True )
        bitmask_pvalues = np.full( shape = cig_signs.shape, fill_value = False )

        ### iterate over available cohort-dependent graphs (CDG)
        for cdg_graph in cdg_labels:
            ## keep interactions whose correlation sign is consistent w.r.t. CIG aka Cohort Independent Graph
            coherence_check = np.sign( g.ep[ cdg_graph ].a ) == cig_signs
            bitmask_corr_sign = np.bitwise_and( bitmask_corr_sign, coherence_check )

            ## controllo significatività nel CDG
            bitmask_pvalue_cdg = g.ep[ self.__gfield_pvalue.format( cdg_graph ) ].a <= self.__significance_lvl
            bitmask_pvalue_overlap = np.bitwise_and( bitmask_cig_pvalues, bitmask_pvalue_cdg )
            bitmask_pvalues = np.bitwise_or( bitmask_pvalues, bitmask_pvalue_overlap )

        final_bitmask = np.bitwise_and( bitmask_corr_sign, bitmask_pvalues )
        logging.debug(f"Passing the final check: {np.sum(final_bitmask)}")
        return final_bitmask


class GraphStatsViz:
    def __init__(self, mog: gt.Graph) -> None:
        self.g = mog 
        self.net_degrees = mog.get_total_degrees( mog.get_vertices() )
    

    def __filter_net_degrees(self, omic: Optional[str] = None):
        net_degrees = self.net_degrees
        if omic is not None: 
            omic_prop = self.g.vp[ GraphField.FEATURE_OMIC.value ]
            net_degrees = [
                self.net_degrees[ v ]
                for v in self.g.iter_vertices() if omic_prop[ v ] == omic
            ]
        return net_degrees

    @classmethod
    def cast_to_go_figure( self, scatter_plot, title: str ) -> go.Figure:
        fig = go.Figure( data = scatter_plot.data, layout = scatter_plot.layout )
        fig.update_layout( title = title )
        return fig 

    def get_degree_distribution_figure(self, omic: Optional[ str ] = None, get_figure: bool = True, color_palette: Dict[str,str] = None) -> go.Figure:
        net_degrees = self.__filter_net_degrees( omic )

        k, counts = np.unique( net_degrees, return_counts=True)
        N = len( net_degrees )
        pk = np.array([ nk / N for nk in counts ])

        if get_figure:
            name = omic if omic else "All nodes"
            color_args = dict( line = dict( width = 1 )  )
            if color_palette and omic:
                color = color_palette[ omic ]
                print(f"COLOR for {omic} ==> {color}")
                color_args = dict(
                    marker = dict( color = color ),
                    line = dict( color = color, width = 1 ),
                )
            return go.Scatter(
                x = k, 
                y = pk, 
                mode = "lines+markers", 
                name = name, 
                **color_args
            )
            
        
        return (k, pk)


    def build_degree_correlation_function(self, get_figure: bool = True):
        """
        Compute the degree correlation function knn(k) = Sum_k' k'*P(k'|k), 
        where P(k'|k) is the conditional probability that following a link 
        of a k-degree node we reach a degree-k' node.
        """

        nets_d = self.__filter_net_degrees()
        adj = gt.adjacency( self.g )
        k, counts = np.unique( nets_d, return_counts=True)

        first_index = int( k[0] == 0 )
        max_k = k[-1]
        max_k__plus1 = int( max_k + 1)

        k_vector = k[ first_index: ]
        f_degree_correlation = np.empty( shape = k_vector.shape )

        degree_array = np.arange( start = 1, stop = max_k__plus1, step = 1)

        ## iterate over available degrees
        for (i,), k in np.ndenumerate( k_vector ): 
            ## compute probabilities P(k'|k) by iterate over neighbors of nodes with degree = k
            # count occurrences of neighbors having degree = 1,2,3,...
            probabilities = np.zeros( shape = (max_k,)  ) ##index i -> degree i(+1)

            for v in np.where( nets_d == k )[0]:
                for neighbor in adj[ v, : ].indices: #np.nonzero( adj[:,v] )[0]:
                    probabilities[ int( nets_d[ neighbor ] - 1 ) ] += 1
            
            probabilities /= np.sum( probabilities)
            f_degree_correlation[ i ] = np.dot( degree_array, probabilities )
        
        if get_figure:
            return go.Scatter(
                x = k_vector, y = f_degree_correlation, 
                mode = "markers", 
            )
        
        
        return k_vector, f_degree_correlation


    def get_clustering_coefficient_figure(self, cc_data: np.array, get_figure: bool = True, color_palette: Dict[str,str] = None) -> go.Figure:
        def unroll_dict( xy_dict: Dict[ int, float ] ) -> Tuple[ np.array, np.array ]:
            max_v = max( list( xy_dict.keys() ) )
            _xy_dict = { k: xy_dict.get(k, 0) for k in range( 1, max_v + 1) }
            return np.array( list( _xy_dict.keys() ) ), np.array( list( _xy_dict.values() ) )

        my_data = zip( 
            list( self.g.vp[ GraphField.FEATURE_OMIC.value ]), 
            self.net_degrees, 
            cc_data )
        df = pd.DataFrame( data = list( my_data ), columns=["omic", "k", "cc"] )
        x_vals_glob, y_vals_glob = unroll_dict( df.groupby("k")["cc"].mean().to_dict() )

        df_omics = {
            omic_value: unroll_dict( 
                df[ df["omic"] == omic_value ].groupby("k")["cc"].mean().to_dict() )

            for omic_value in set( df["omic"].tolist() )
        }
        
        if get_figure:
            color_args = dict( line = dict( width = 1 )  )
            fig = go.Figure() 
            fig.add_trace( go.Scatter( 
                x = x_vals_glob, 
                y = y_vals_glob, 
                mode = "lines+markers", 
                name="All omics", **color_args ) ) 
            
            for omic_value, (x_vals, y_vals) in df_omics.items():
                color_args = dict(
                    marker = dict( color = color_palette[omic_value] ),
                    line = dict( color = color_palette[omic_value], width = 1 ),
                )
                fig.add_trace( 
                    go.Scatter( 
                        x = x_vals, y = y_vals, 
                        mode = "lines+markers", 
                        name = omic_value, **color_args ) )
                
            fig.update_layout(
                title="Clustering coefficient function",
                xaxis_title="Degree k",
                yaxis_title="C(k)",
                legend_title="Graph traces" )
            return fig
        
        df_omics["All"] = (x_vals_glob, y_vals_glob)
        return df_omics


class CommunityDetector:

    @classmethod
    def louvain_community_detection(cls, g: gt.Graph, random_state: int = 42 ):
        
        new_ep = None 
        if "WholeData" in g.ep:
            new_ep = g.new_ep("double")
            new_ep.a = np.abs( g.ep["WholeData"].a )

        algorithm = Louvain(random_state=random_state).fit( gt.adjacency( g, weight = new_ep ), force_bipartite = gt.is_bipartite(g) )
        return algorithm.predict() 


    @classmethod
    def get_neighborhoods( cls, g: gt.Graph ):
        return tuple([  
            np.insert( g.get_all_neighbors(v), 0, v ) 
                for v in range( g.num_vertices() )
        ])

    @classmethod
    def ravasz_similarity_matrix( cls, g: gt.Graph ):
        nv = g.num_vertices() 
        similarity_matrix = np.zeros( shape = ( nv, nv ), dtype=np.float64 )
        neighbors = cls.get_neighborhoods( g )
        
        for i in range( nv ):
            neighs_i = neighbors[ i ]
            for j in range( i + 1, nv ):
                neighs_j = neighbors[ j ]
                min_k = min( len(neighs_i), len(neighs_j) )
                d = np.intersect1d( neighs_i, neighs_j, assume_unique=True).size / ( min_k + 1 - int(i in neighs_j) )
                similarity_matrix[i, j] = similarity_matrix[j, i] = d

        return similarity_matrix
    
    @classmethod
    def link_clustering_similarity_matrix( cls, g: gt.Graph ):
        def fix_pair( x, y ):
            return (x, y) if x < y else (y, x)

        nv = g.num_vertices()
        ne = g.num_edges() 

        edge_enc = { 
            fix_pair(u, v): i 
                for i, (u, v) in enumerate( g.iter_edges() ) }

        neighborhoods = cls.get_neighborhoods( g )
        
        MISSING_VALUE = -1.0
        memo_cache = np.full( (nv, nv), MISSING_VALUE )
        similarity_matrix = sp.lil_matrix((ne, ne))
        
        for neighbor_list in neighborhoods: 
            i = neighbor_list[0]
            
            for u, v in combinations( neighbor_list[1:], 2): 
                ## get edges identifiers  &  get indices in distance matrix 
                if memo_cache[( pair_uv := fix_pair(u, v) )] == MISSING_VALUE:
                    n_u, n_v = neighborhoods[u], neighborhoods[v]
                    memo_cache[ pair_uv ] = len( np.intersect1d( n_u, n_v, assume_unique=True) ) / len( np.union1d( n_u, n_v ) )
                    
                j = edge_enc.get( fix_pair(i, u) )
                k = edge_enc.get( fix_pair(i, v) )
                similarity_matrix[j, k] = similarity_matrix[k, j] = memo_cache[ pair_uv ] 

        return similarity_matrix.tocsr()
        

    @classmethod
    def spectral_clustering( cls, g: gt.Graph, nc_max: int = 5, random_state: int = 42 ) -> pd.DataFrame:

        adj = np.abs( gt.adjacency( g, weight = g.ep["WholeData"] ).toarray() )
        predictions = dict() 

        for nc in range(2, nc_max + 1):
            spectral = SpectralClustering(
                n_clusters = nc, 
                eigen_solver="arpack", 
                affinity="nearest_neighbors", 
                random_state=random_state ).fit( adj )
            
            predictions[ f"spectral_{nc}" ] = spectral.labels_

        return build_g_dataframe( g, predictions )


    @classmethod
    def community_detection(cls, g: gt.Graph, vertex_name_prop: str = None, random_state: int = 42) -> Tuple[pd.DataFrame, Any]:
        gt.seed_rng(random_state)
        np.random.seed(random_state)
        state = gt.minimize_nested_blockmodel_dl( g )
        ## TODO - improve state / check degree correction
        df = cls.explore_nested_blockstate( g, state, vertex_name_prop )
        df.insert( 0, "Louvain", cls.louvain_community_detection( g, random_state=random_state ) )
        ravasz_ = cls.hierarchical_clustering( cls.ravasz_similarity_matrix( g ) )
        

        ncols = df.shape[1] 
        for col in ravasz_.columns.tolist()[::-1]:
            df.insert( ncols, col, ravasz_[col].to_numpy() )
        
        try:
            spectral_cl = cls.spectral_clustering( g, random_state=random_state )
        except ValueError as e:
            logging.critical(f"Spectral clustering failed: {e}")
        else:
            df = pd.merge( df, spectral_cl, left_index=True, right_index=True )

        return df, state

    @classmethod
    def hierarchical_clustering(cls, pairwise_dist_m: np.array, nc_max: int = 5):
        clusters_ravasz = fastcluster.linkage( pairwise_dist_m, method = "average" )

        columns = dict() 
        for t_value in list( clusters_ravasz[:,2] )[::-1][:nc_max]:
            prediction = hierarchy.fcluster( clusters_ravasz, t_value, criterion = "distance" )
            nclusters = len( set( prediction ) )
            if nclusters > 1:
                columns[ f"ravasz-{nclusters}" ] = prediction - 1 ##forcing cluster id to start from zero

        return pd.DataFrame( columns )
    

    @classmethod
    def explore_nested_blockstate( cls, g: gt.Graph, state: gt.NestedBlockState, vertex_name_prop: str = None ):
        ### count the number of hierarchical levels of the network
        levels = state.get_levels()

        for lvl, blockstate in enumerate( levels ):
            if blockstate.get_N() == 1:
                max_depth = lvl - 1
                break 

        hierarchy = list()
        blocks = list()

        for lvl in range( max_depth ):
            #get the community id at i-th level of each vertex of the original graph 
            block_prj = state.project_level( lvl )
            hierarchy.append( gt.contiguous_map( block_prj.get_blocks().fa ) )

            c_state = block_prj.copy( b = hierarchy[ -1 ] )
            e = c_state.get_matrix()
            B = c_state.get_nonempty_B()

            blocks.append( e.todense()[:B, :B] )

        df_content = {
            f"SBM_{i}": blocks
                for i, blocks in enumerate( hierarchy )
        }

        return build_g_dataframe( g, df_content, vertex_name_prop )


class GraphDistanceManager:
    AVAIL_DISTANCES = {
        # 'Jaccard':                 netrd.distance.JaccardDistance(), # need the weighted version
        'Hamming':                 netrd.distance.Hamming(),
        'HammingIpsenMikhailov':   netrd.distance.HammingIpsenMikhailov(),
        'Frobenius':               netrd.distance.Frobenius(),
        'PolynomialDissimilarity': netrd.distance.PolynomialDissimilarity(),
        'DegreeDivergence':        netrd.distance.DegreeDivergence(),
        'PortraitDivergence':      netrd.distance.PortraitDivergence(),
        'QuantumJSD':              netrd.distance.QuantumJSD(),
        'NetLSD':                  netrd.distance.NetLSD(),
        'IpsenMikhailov':          netrd.distance.IpsenMikhailov(),
        'DeltaCon':                netrd.distance.DeltaCon(),
        'NetSimile':               netrd.distance.NetSimile()
    }

    @classmethod
    def avail_distances(cls) -> List[str]:
        return sorted( cls.AVAIL_DISTANCES.keys() )
    
    @classmethod
    def get_pairwise_distances(cls, g1: nx.Graph, g2: nx.Graph, distances) -> pd.Series:
        dist_data = {
            dist_name: dist_measure 
                for dist_name in distances 
                    if ( dist_measure := cls.compute_distance( g1, g2, dist_name) )
        }
        return pd.Series( data = dist_data )


    @classmethod
    def get_distances(cls, graph_set: Dict[str, nx.Graph], distances: List[str]) -> Dict[str, pd.DataFrame]:
        g_ordering = sorted( graph_set.keys() )
        map_to_index = { g_name: i for i, g_name in enumerate( g_ordering ) }
        n_graphs = len( g_ordering )
        graph_metrics = {
            dist_metric: np.zeros( shape=( n_graphs, n_graphs ) )
                for dist_metric in distances
        }

        for name1, name2 in combinations( g_ordering, 2 ):
            nx_g1, nx_g2 = graph_set[name1], graph_set[name2] 
            i, j = map_to_index[name1], map_to_index[name2] 
            for index, value in cls.get_pairwise_distances( nx_g1, nx_g2, distances).items():
                m = graph_metrics[ index ]
                m[i, j] = m[j, i] = value 

        graph_metrics = {
            dist_metric: pd.DataFrame( data = dist_matrix, columns = g_ordering, index = g_ordering )
                for dist_metric, dist_matrix in graph_metrics.items()
        }
        return graph_metrics


    @classmethod
    def compute_distance(cls, g1, g2, distance_name: str):
        
        if ( func := cls.AVAIL_DISTANCES.get( distance_name ) ):
            try:
                return func.dist( g1, g2 )
            except Exception as e:
                logging.critical(f"Distance name {distance_name} exploded: {e}")

