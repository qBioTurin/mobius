import enum 
from dataclasses import dataclass
from typing import List, Dict
import graph_tool.all as gt, pandas as pd
from tools.utils import build_tsv_archive, write_plotly_in_zip
import plotly.graph_objects as go


class SessionState( enum.Enum ):
    WORKING_GRAPH = "wo_graph"
    CURRENT_GRAPH = "cu_graph"
    NET_RESULTS = "net_analysis"
    AVAIL_OMICS = "avail_omics"
    IS_DEFAULT_GRAPH = "is_default_graph"

    V_POS = "vpos_{}"

    LAST_UPLOADED_FILENAME = "uploaded_filename"
    FILTERING_PARAMS = "graph_filt_params"
    VPROP_FEATURE_NAME = "feature_name_vprop"

class Coefficients( enum.Enum ):
    SPEARMAN = "SPEARMAN"
    PEARSON = "PEARSON"
    XI = "XI"
    KENDALL = "KENDALL"

class GraphField(enum.Enum):
    #### graph properties 
    ENCODED_GRAPHS = "graphs"
    STRAT_LAYERS = "strat_layers"
    CORRELATION_FUNCTION = "corr_func"
    OMICS_ENUM = "omics_list"

    VWEIGHTS_SET = "vweights"
    FEATURE_SCORES = "scores_{}" #scores_<TARGET COVARIATE>

    #### vertex properties
    FEATURE_NAME = "feature"
    FEATURE_OMIC = "omic"
    #### edge properties
    INTRAOMIC_EDGE = "intra"
    GRAPH_REPR = "{}"
    PVALUE = "p_{}"
    ADJ_PVALUE = "padj_{}"
    # vertex weights
    VERTEX_WEIGHT = "vscore_{}"
    WX_PVALUE = "wx_pval"

class EdgeType( enum.Enum ):
    INTER_OMICS = 0 #"inter"
    INTRA_OMICS = 1 #"intra"
    ALL_EDGES = 2 #"all"


    @classmethod
    def from_string(cls, value):
        value = value.lower()
        if value == "all":
            return cls.ALL_EDGES
        else:
            return cls.INTRA_OMICS if "intra" in value else cls.INTER_OMICS


@dataclass(frozen=True)
class GraphParametrization:
    vname_prop: str
    vprop_list: List[ str ]
    eprop_list: List[ str ]


@dataclass(frozen=True)
class FilteringParameters:
    min_corr_threshold: float 
    pvalue_threshold: float 
    padj_flag: bool 
    chosen_omics: List[ str ]
    edge_type: int 


@dataclass(frozen=True)
class GraphFilteringParametrization: 
    graph_id: str
    graph_key: str 
    filtering_params: FilteringParameters
    # graph_key: str 
    # corr_threshold: float 
    # pvalue_threshold: float 
    # padj_flag: bool 
    # chosen_omics: List[ str ]
    # edge_type: int 


@dataclass
class GraphStats:
    vertex_data: pd.DataFrame
    vertex_props: Dict[ str, gt.VertexPropertyMap ]
    communities: pd.DataFrame
    edge_data: pd.DataFrame
    fig_degree_distr: go.Figure
    fig_degree_corr: go.Figure
    fig_avg_cc: go.Figure


    def build_zip(self) -> str:
        zip_out_filename = build_tsv_archive(
            vertex_data = self.vertex_data, 
            communities = self.communities,
            edge_data = self.edge_data
        )

        write_plotly_in_zip( zip_out_filename, self.fig_degree_distr, "degree_distribution")
        write_plotly_in_zip( zip_out_filename, self.fig_degree_corr, "degree_correlation")
        write_plotly_in_zip( zip_out_filename, self.fig_avg_cc, "average_clustering_coefficient")

        return zip_out_filename
    
    def get_vertex_colnames(self) -> List[ str ]:
        return self.vertex_data.columns.tolist() + self.communities.columns.tolist() 
    
    def get_vertex_associated_data(self) -> pd.DataFrame:
        return pd.concat([ self.vertex_data, self.communities], axis=1)


@dataclass
class PlottingGraphData:
    edge_list: list ##pair of vertices
    edge_x: list    ## x coordinates 
    edge_y: list    ## y coordinates
    edge_w: list    ## edge weights
    edge_t: list    ## edge type (intra/inter-omics)